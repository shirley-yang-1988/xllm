#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Standalone benchmark adaptation: Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Standalone transplant of the original vLLM-Ascend greedy verifier.

Source: vllm-project/vllm-ascend@083d66780c0909ae97d654660ab38665ec5aa31a
Path: vllm_ascend/ops/triton/reject_sample.py (Apache-2.0).

The five functions in REFERENCE_FUNCTION_SHA256 are copied byte-for-byte,
including decorators, signatures, comments, and launch parameters. They are the
original grid helper, K=1 kernel, bonus helper, general kernel, and dispatcher;
no V2, random-sampling, argmax, or logits processing is included.

Only non-kernel dependencies are adapted: direct Triton imports, the original
get_element extension/language lookup order, vectorcore discovery through the
same active-driver API as triton_utils.py, and the integer power-of-two helper.
Unused vLLM device/UB configuration is not imported. This is a benchmark
transplant, not an installed or end-to-end vLLM-Ascend integration. The companion
adapter verifies the function-source hashes before any benchmark launch.
"""

from typing import Any

import torch
import triton
import triton.language as tl

REFERENCE_REVISION = "083d66780c0909ae97d654660ab38665ec5aa31a"
REFERENCE_SOURCE_SHA256 = {
    "vllm_ascend/ops/triton/reject_sample.py": "96211ad30ca54dcc4e467d88cccd262fd0f766dd378b6544ed41c094003acba8",
    "vllm_ascend/ops/triton/triton_utils.py": "057221397bdbfa80258e861eb3fc79792af4003917156fd3cd9be8abf4036fc2",
}
REFERENCE_FUNCTION_SHA256 = {
    "cal_grid_and_block_size": "01d8e252eb870f51d59b9661e21867384937fa72bbc81791febba5ea93943de8",
    "rejection_greedy_sample_spec_len_1_triton": "a72e0a76f53e3a6db56d102c4c9d1024ba57d68005dc3db1402db208be5171fc",
    "bonus_renew": "96e38ae4477976648573c0766290dbc5296e9d122a06f91bf106f084572e6916",
    "rejection_greedy_sample_triton": "dd914917db15192ccbcfdc14b3832f0043065c9e65725eb2abf0e7f31052b025",
    "rejection_greedy_sample_with_triton": "f81da42b285322f24207f3b9f0928414ab6ddbb44f6187943463b16d0a6891a1",
}

try:
    import triton.language.extra.cann.extension as _extension_module
except ImportError:
    _extension_module = None


def _resolve_triton_ascend_op(op_name: str) -> Any:
    if _extension_module is not None:
        extension_op = getattr(_extension_module, op_name, None)
        if extension_op is not None:
            return extension_op
    tl_op = getattr(tl, op_name, None)
    if tl_op is not None:
        return tl_op
    raise RuntimeError(
        f"Failed to resolve Triton op '{op_name}': "
        "neither triton.language.extra.cann.extension nor triton.language provides it."
    )


get_element = _resolve_triton_ascend_op("get_element")
_NUM_VECTORCORE = -1


def init_device_properties_triton() -> None:
    global _NUM_VECTORCORE
    if _NUM_VECTORCORE == -1:
        device_properties = triton.runtime.driver.active.utils.get_device_properties(torch.npu.current_device())
        vectorcore_num = device_properties.get("num_vectorcore", -1)
        if not isinstance(vectorcore_num, int) or vectorcore_num <= 0:
            raise RuntimeError(f"Failed to detect vectorcore count: {device_properties}")
        _NUM_VECTORCORE = vectorcore_num


def get_vectorcore_num() -> int:
    if _NUM_VECTORCORE <= 0:
        raise RuntimeError("Device properties not initialized. Call init_device_properties_triton() first.")
    return _NUM_VECTORCORE


def next_power_of_2(n: int) -> int:
    """Keep the integer helper's inclusive power-of-two semantics without vLLM."""
    return 1 if n < 1 else 1 << (n - 1).bit_length()


# These upstream functions are intentionally not reformatted or reannotated.
def cal_grid_and_block_size(batch_size: int):
    vectorcore_num = get_vectorcore_num()
    if batch_size <= vectorcore_num:
        grid = batch_size
        block_size = 1
    else:
        grid = vectorcore_num
        block_size = next_power_of_2(triton.cdiv(batch_size, grid))
    return grid, block_size


@triton.jit(do_not_specialize=["vec_len"])
def rejection_greedy_sample_spec_len_1_triton(
    output_token_ids_ptr,  # [batch_size, 2]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,
    vec_len,
    uniform_probs_ptr,  # [num_tokens] or None (synthetic only)
    synthetic_conditional_rates_ptr,  # [num_speculative_tokens] or None
    SYNTHETIC_MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < vec_len

    draft_token_id = tl.load(draft_token_ids_ptr + offset, mask)
    target_argmax_id = tl.load(target_argmax_ptr + offset, mask)
    bonus_token_id = tl.load(bonus_token_ids_ptr + offset, mask)

    if SYNTHETIC_MODE:
        # Synthetic: accept the draft token with prob conditional_rates[0],
        # regardless of target match. Accepted => emit draft token (pos 0) and
        # the bonus token (pos 1); rejected => emit target_argmax (pos 0).
        uniform_prob = tl.load(uniform_probs_ptr + offset, mask)
        # spec_len == 1 => only position 0.
        rate = tl.load(synthetic_conditional_rates_ptr + 0)
        accepted = (uniform_prob < rate) & (draft_token_id >= 0) & mask
        # Cast both arms to int32: draft_token_id is int32, target_argmax_id is
        # int64 (from argmax); tl.where requires matching dtypes.
        token_id = tl.where(accepted, draft_token_id.to(tl.int32), target_argmax_id.to(tl.int32))
        tl.store(output_token_ids_ptr + offset * 2, token_id, mask)
        accept_mask = accepted
    else:
        tl.store(output_token_ids_ptr + offset * 2, target_argmax_id, mask)
        accept_mask = (draft_token_id == target_argmax_id) & mask
    tl.store(output_token_ids_ptr + offset * 2 + 1, bonus_token_id, accept_mask)


@triton.jit(do_not_specialize=["max_spec_len"])
def bonus_renew(
    bonus_token_ids_ptr,
    position,
    output_token_ids_ptr,
    max_spec_len,
    num_tokens1,
):
    bonus_token_id = tl.load(bonus_token_ids_ptr + position)
    tl.store(output_token_ids_ptr + position * (max_spec_len + 1) + num_tokens1, bonus_token_id)


@triton.jit(do_not_specialize=["vec_len", "max_spec_len"])
def rejection_greedy_sample_triton(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    is_greedy_ptr,  # [batch_size] or None
    vec_len,
    max_spec_len,
    uniform_probs_ptr,  # [num_tokens] or None (synthetic only)
    synthetic_conditional_rates_ptr,  # [num_speculative_tokens] or None
    SYNTHETIC_MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < vec_len

    if is_greedy_ptr is None:
        is_greedy_mask = mask
    else:
        is_greedy = tl.load(is_greedy_ptr + offset, mask=mask, other=0)
        is_greedy_mask = mask & (is_greedy != 0)

    # Mask the load itself: tl.where does not prevent reading before the
    # buffer start (both arms are always evaluated), so lane offset == 0 must
    # be masked out at the load. other=0 keeps num_draft_tokens deterministic
    # (0) for masked lanes, which the per-position loop below relies on to
    # skip them.
    start_idx = tl.load(cu_num_draft_tokens_ptr + offset - 1, mask=is_greedy_mask & (offset > 0), other=0)
    end_idx = tl.load(cu_num_draft_tokens_ptr + offset, is_greedy_mask, other=0)
    num_draft_tokens = end_idx - start_idx

    for pos in tl.range(0, BLOCK_SIZE):
        num_tokens1 = get_element(num_draft_tokens, (pos,))
        rejected = False
        start_idx1 = get_element(start_idx, (pos,))
        is_greedy_mask1 = get_element(is_greedy_mask, (pos,))
        position = block_idx * BLOCK_SIZE + pos
        for i in range(num_tokens1):
            if not rejected:
                draft_token_id = tl.load(draft_token_ids_ptr + start_idx1 + i)
                target_argmax_id = tl.load(target_argmax_ptr + start_idx1 + i)
                if SYNTHETIC_MODE:
                    # Synthetic: accept draft token i with prob
                    # conditional_rates[i], independent of target match. Store
                    # each arm separately (draft on accept, target_argmax on
                    # reject) so the int32 (draft) / int64 (target_argmax)
                    # dtype mismatch is handled by the store's implicit cast --
                    # no ternary, no explicit cast (matches the random kernel's
                    # synthetic branch).
                    uniform_prob = tl.load(uniform_probs_ptr + start_idx1 + i)
                    rate = tl.load(synthetic_conditional_rates_ptr + i)
                    accepted = (uniform_prob < rate) & (draft_token_id >= 0)
                    if accepted:
                        tl.store(
                            output_token_ids_ptr + position * (max_spec_len + 1) + i,
                            draft_token_id,
                        )
                    else:
                        tl.store(
                            output_token_ids_ptr + position * (max_spec_len + 1) + i,
                            target_argmax_id,
                        )
                        rejected = True
                else:
                    tl.store(
                        output_token_ids_ptr + position * (max_spec_len + 1) + i,
                        target_argmax_id,
                    )
                    if draft_token_id != target_argmax_id:
                        # Reject.
                        rejected = True

        if not rejected and is_greedy_mask1:
            bonus_renew(
                bonus_token_ids_ptr,
                position,
                output_token_ids_ptr,
                max_spec_len,
                num_tokens1,
            )


def rejection_greedy_sample_with_triton(
    output_token_ids,
    num_draft_tokens,
    cu_num_draft_tokens,
    draft_token_ids,
    target_argmax,
    bonus_token_ids,
    is_greedy,
    max_spec_len,
    grid,
    block_size,
    uniform_probs=None,
    synthetic_conditional_rates=None,
    synthetic_mode=False,
):
    vec_len = output_token_ids.shape[0]

    if min(num_draft_tokens) == 1 and max(num_draft_tokens) == 1 and is_greedy is None:
        rejection_greedy_sample_spec_len_1_triton[(grid,)](
            output_token_ids,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            vec_len,
            uniform_probs,
            synthetic_conditional_rates,
            SYNTHETIC_MODE=synthetic_mode,
            BLOCK_SIZE=block_size,
        )
    else:
        rejection_greedy_sample_triton[(grid,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            vec_len,
            max_spec_len,
            uniform_probs,
            synthetic_conditional_rates,
            SYNTHETIC_MODE=synthetic_mode,
            BLOCK_SIZE=block_size,
        )
