# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exact NPU tests of the actual F1 TileLang builder, without a C++ wrapper.

Select the initial INT64 MTP and empty-input checks with ``-k smoke``.
``-k triton`` independently tests the pinned Triton transplant without importing
TileLang or creating the F1 runner; it is not an F1 fallback or timing evidence.
The test-only launch harness owns output allocation and the B=0 no-launch rule;
these tests do not establish those properties for a future C++ wrapper. Launch
indices, strides and relative spans must fit INT32; native token dtypes remain
INT32/INT64. Pure metadata checks do not establish device correctness. CPU
oracles, snapshots and synchronization are correctness checks, not timing data.
Missing torch_npu, TileLang or an available NPU is an error, never a skipped pass.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from typing import Any

import pytest
import torch

from scripts.logger import logger

_GUARD_SIZE = 16
_CANARY = 0x13579BDF
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1
_ALL_INT64 = (64, 64, 64)


@dataclass(frozen=True)
class _Inputs:
    draft: torch.Tensor
    target: torch.Tensor
    bonus: torch.Tensor
    storages: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class _Outputs:
    full: torch.Tensor
    masked: torch.Tensor | None
    storages: tuple[torch.Tensor, ...]


def _dtype(bits: int) -> torch.dtype:
    if bits not in (32, 64):
        raise ValueError(f"Expected 32 or 64 input bits, got {bits}")
    return torch.int32 if bits == 32 else torch.int64


def _checked_span(shape: tuple[int, ...], strides: tuple[int, ...], label: str) -> int:
    if len(shape) != len(strides):
        raise ValueError(f"{label}: shape and stride ranks differ")
    for name, values in (("shape", shape), ("stride", strides)):
        if any(value < 0 or value > _INT32_MAX for value in values):
            raise ValueError(f"{label}: {name} must fit nonnegative INT32, got {values}")
    # Empty tensors have no relative addresses; do not evaluate (size - 1).
    span = 0 if 0 in shape else 1 + sum((size - 1) * stride for size, stride in zip(shape, strides))
    if span > _INT32_MAX:
        raise ValueError(f"{label}: physical span {span} exceeds INT32")
    return span


def _flat_storage_view(tensor: torch.Tensor) -> torch.Tensor:
    span = _checked_span(tuple(tensor.shape), tensor.stride(), "input")
    offset = tensor.storage_offset()
    storage_size = tensor.untyped_storage().nbytes() // tensor.element_size()
    if offset < 0 or (span != 0 and offset + span > storage_size):
        raise ValueError(f"input: offset={offset}, span={span} exceeds storage size {storage_size}")
    # data_ptr already includes storage_offset; it is not an INT32 kernel index.
    flat = tensor.as_strided((span,), (1,), offset)
    assert flat.storage_offset() == tensor.storage_offset()
    assert flat.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
    if tensor.numel() != 0:
        assert flat.data_ptr() == tensor.data_ptr()
    return flat


class _KernelRunner:
    def __init__(self, device: torch.device) -> None:
        # Import after the fixture selects the visible device, not at collection.
        tilelang = importlib.import_module("tilelang")
        implementation = importlib.import_module("xllm.python.kernels_npu.tilelang.greedy_prefix_verify")
        self.device = device
        self.default_task_count = implementation.DEFAULT_TASK_COUNT
        self._select_task_count = implementation.select_greedy_prefix_verify_task_count
        self.launch_count = 0
        self.output_allocation_count = 0
        self._kernels: dict[tuple[int, int, int, int], Any] = {}
        self._factory = tilelang.jit(pass_configs=implementation.GREEDY_PREFIX_VERIFY_PASS_CONFIGS)(
            implementation.build_greedy_prefix_verify_kernel
        )

    def _get_kernel(self, inputs: _Inputs, task_count: int) -> Any:
        bits = tuple(tensor.element_size() * 8 for tensor in (inputs.draft, inputs.target, inputs.bonus))
        key = (task_count, *bits)
        if key not in self._kernels:
            self._kernels[key] = self._factory(
                task_count=task_count, draft_bits=bits[0], target_bits=bits[1], bonus_bits=bits[2]
            )
        return self._kernels[key]

    def _allocate_output(self, shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        size = _checked_span(shape, (shape[1], 1), "output")
        storage = torch.full((size + 2 * _GUARD_SIZE,), _CANARY, dtype=torch.int32, device=self.device)
        self.output_allocation_count += 1
        return storage[_GUARD_SIZE : _GUARD_SIZE + size].view(shape), storage

    def _run(self, inputs: _Inputs, mask: bool, task_count: int | None = None) -> _Outputs:
        batch, width = inputs.target.shape
        if task_count is None:
            task_count = self._select_task_count(batch)
        flat_inputs = tuple(_flat_storage_view(tensor) for tensor in (inputs.draft, inputs.target, inputs.bonus))
        full, full_storage = self._allocate_output((batch, width + 1))
        masked = None
        storages = (full_storage,)
        if mask:
            masked, masked_storage = self._allocate_output((batch, width + 1))
            storages += (masked_storage,)
        if batch != 0:
            kernel = self._get_kernel(inputs, task_count)
            # Five buffers and eight INT32 runtime scalars; no input copies/casts.
            kernel(
                *flat_inputs,
                full.view(-1),
                (full if masked is None else masked).view(-1),
                batch,
                width,
                inputs.draft.stride(0),
                inputs.draft.stride(1),
                inputs.target.stride(0),
                inputs.target.stride(1),
                inputs.bonus.stride(0),
                int(mask),
            )
            self.launch_count += 1
        return _Outputs(full, masked, storages)


@pytest.mark.parametrize(
    ("shape", "strides", "span"),
    [
        ((0, 3), (_INT32_MAX, _INT32_MAX), 0),
        ((3, 0), (_INT32_MAX, _INT32_MAX), 0),
        ((2, 1), (_INT32_MAX - 1, 1), _INT32_MAX),
        ((3, 4), (4, 1), 12),
    ],
)
def test_int32_index_metadata_boundaries(shape: tuple[int, ...], strides: tuple[int, ...], span: int) -> None:
    # Metadata-only boundary checks: no large allocation or device claim.
    assert _checked_span(shape, strides, "metadata") == span


@pytest.mark.parametrize(
    ("shape", "strides", "message"),
    [
        ((-1, 3), (3, 1), "shape"),
        ((_INT32_MAX + 1, 1), (1, 1), "shape"),
        ((0, _INT32_MAX + 1), (_INT32_MAX + 1, 1), "shape"),
        ((1, 1), (-1, 1), "stride"),
        ((0, 3), (_INT32_MAX + 1, 1), "stride"),
        ((2, 1), (_INT32_MAX, 1), "physical span"),
        ((2, _INT32_MAX), (_INT32_MAX, 1), "physical span"),
    ],
)
def test_int32_index_metadata_rejects_overflow(shape: tuple[int, ...], strides: tuple[int, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _checked_span(shape, strides, "metadata")


@pytest.fixture(scope="module")
def runner() -> _KernelRunner:
    importlib.import_module("torch_npu")
    assert torch.npu.is_available(), "F1 precision tests require an available Ascend NPU"
    # Logical device zero follows the invocation's ASCEND_RT_VISIBLE_DEVICES.
    torch.npu.set_device(0)
    return _KernelRunner(torch.device("npu:0"))


def _strided_input(
    values: torch.Tensor,
    bits: int,
    device: torch.device,
    strides: tuple[int, int],
    offset: int = _GUARD_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    span = 0 if values.numel() == 0 else 1 + sum((size - 1) * stride for size, stride in zip(values.shape, strides))
    storage_cpu = torch.full((offset + span + _GUARD_SIZE,), _CANARY, dtype=torch.int64)
    storage_cpu.as_strided(values.shape, strides, offset).copy_(values)
    storage = storage_cpu.to(device=device, dtype=_dtype(bits))
    return storage.as_strided(values.shape, strides, offset), storage


def _mtp_inputs(
    draft_cpu: torch.Tensor,
    target_cpu: torch.Tensor,
    bonus_cpu: torch.Tensor,
    device: torch.device,
    bits: tuple[int, int, int] = _ALL_INT64,
) -> _Inputs:
    batch, width = target_cpu.shape
    draft, draft_storage = _strided_input(draft_cpu, bits[0], device, (max(width, 1), 1))
    block_cpu = torch.cat((target_cpu, bonus_cpu), dim=1)
    block, target_storage = _strided_input(block_cpu, bits[1], device, (width + 1, 1))
    target = block[:, :width]
    if bits[1] == bits[2]:
        bonus_block = block
        storages = (draft_storage, target_storage)
    else:
        bonus_block, bonus_storage = _strided_input(block_cpu, bits[2], device, (width + 1, 1))
        storages = (draft_storage, target_storage, bonus_storage)
    bonus = bonus_block.view(-1)[width :: width + 1].view(batch, 1)
    if batch != 0:
        assert target.stride() == (width + 1, 1)
        assert bonus.stride(0) == width + 1
        assert bonus.storage_offset() == bonus_block.storage_offset() + width
    if bits[1] == bits[2]:
        assert target.untyped_storage().data_ptr() == bonus.untyped_storage().data_ptr()
    return _Inputs(draft, target, bonus, storages)


def _logical_ids(
    batch: int, width: int, rejection: str, edge_values: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = torch.arange(batch * (width + 1), dtype=torch.int64).view(batch, width + 1)
    if edge_values:
        palette = torch.tensor(
            [_INT32_MIN, _INT32_MIN + 1, 1 << 24, (1 << 24) + 1, _INT32_MAX - 1, _INT32_MAX, 0, -1, -(1 << 24) - 1],
            dtype=torch.int64,
        )
        block = palette[indices % palette.numel()]
    else:
        block = indices + 101
    target, bonus = block[:, :width], block[:, width:]
    draft = target.clone()
    for row in range(batch):
        first = {
            "first": 0,
            "middle": width // 2,
            "last": max(width - 1, 0),
            "all": width,
            "rematch": width // 3,
            "mixed": row % (width + 1),
        }[rejection]
        if first < width:
            end = first + 1 if rejection in ("rematch", "mixed") else width
            # Adjacent integer mismatches remain in INT32, including its limits.
            draft[row, first:end] = torch.bitwise_xor(draft[row, first:end], 1)
    return draft, target, bonus


def _semantic_oracle(
    draft: torch.Tensor, target: torch.Tensor, bonus: torch.Tensor, mask: bool
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # Compare original integers, not narrowed data or the kernel's bitmask logic.
    assert draft.device.type == target.device.type == bonus.device.type == "cpu"
    batch, width = target.shape
    full = torch.empty((batch, width + 1), dtype=torch.int32)
    masked = torch.empty_like(full) if mask else None
    for row in range(batch):
        first_reject = next(
            (column for column in range(width) if int(draft[row, column]) != int(target[row, column])), width
        )
        for column in range(width + 1):
            value = int(target[row, column]) if column < width else int(bonus[row, 0])
            assert _INT32_MIN <= value <= _INT32_MAX
            full[row, column] = value
            if masked is not None:
                masked[row, column] = value if column <= first_reject else -1
    return full, masked


def _torch_reference(inputs: _Inputs, mask: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    # Test-only transcription of greedy_sample_from_token_ids/build_accepted_mask.
    full = torch.cat((inputs.target, inputs.bonus), dim=-1).to(torch.int32)
    if not mask:
        return full, None
    batch, width = inputs.target.shape
    accepted = (inputs.target == inputs.draft).to(torch.int64)
    bonus_mask = torch.zeros((batch, 1), dtype=torch.int64, device=inputs.target.device)
    combined = torch.cat((accepted, bonus_mask), dim=-1)
    first_rejected = (1 - combined).argmax(dim=1, keepdim=True)
    positions = torch.arange(width + 1, device=inputs.target.device).unsqueeze(0)
    return full, torch.where(positions <= first_rejected, full, torch.full_like(full, -1))


def _context(inputs: _Inputs, mask: bool, task_count: int) -> str:
    tensors = (inputs.draft, inputs.target, inputs.bonus)
    return (
        f"B={inputs.target.shape[0]} K={inputs.target.shape[1]} mask={mask} tasks={task_count} "
        f"dtypes={[str(tensor.dtype) for tensor in tensors]} "
        f"strides={[tensor.stride() for tensor in tensors]} "
        f"offsets={[tensor.storage_offset() for tensor in tensors]}"
    )


def _assert_pair(
    actual: tuple[torch.Tensor, torch.Tensor | None],
    expected: tuple[torch.Tensor, torch.Tensor | None],
    context: str,
) -> None:
    for name, result, reference in zip(("full", "masked"), actual, expected):
        if reference is None:
            assert result is None, f"{context} {name} must be None"
        else:
            assert result is not None, f"{context} {name} is missing"
            assert result.device.type == "npu", f"{context} {name} is not on NPU"
            assert result.dtype == torch.int32, f"{context} {name} has dtype {result.dtype}"
            torch.testing.assert_close(
                result.cpu(), reference, rtol=0, atol=0, msg=lambda detail, name=name: f"{context} {name}\n{detail}"
            )


def _assert_canaries(outputs: _Outputs, context: str) -> None:
    expected = torch.full((_GUARD_SIZE,), _CANARY, dtype=torch.int32)
    for index, storage in enumerate(outputs.storages):
        snapshot = storage.cpu()
        torch.testing.assert_close(snapshot[:_GUARD_SIZE], expected, rtol=0, atol=0, msg=f"{context} prefix {index}")
        torch.testing.assert_close(snapshot[-_GUARD_SIZE:], expected, rtol=0, atol=0, msg=f"{context} suffix {index}")


def _check_case(runner: _KernelRunner, inputs: _Inputs, mask: bool, task_count: int | None = None) -> _Outputs:
    if task_count is None:
        task_count = runner._select_task_count(inputs.target.shape[0])
    context = _context(inputs, mask, task_count)
    logger.info(f"F1 Python precision: {context}")
    snapshots = tuple(storage.cpu() for storage in inputs.storages)
    expected = _semantic_oracle(inputs.draft.cpu(), inputs.target.cpu(), inputs.bonus.cpu(), mask)
    reference = _torch_reference(inputs, mask)
    launches_before = runner.launch_count
    allocations_before = runner.output_allocation_count
    outputs = runner._run(inputs, mask, task_count)
    torch.npu.synchronize()
    assert runner.launch_count - launches_before == int(inputs.target.shape[0] != 0), context
    assert runner.output_allocation_count - allocations_before == 1 + int(mask), context
    _assert_pair((outputs.full, outputs.masked), expected, f"TileLang {context}")
    _assert_pair(reference, expected, f"old Torch {context}")
    _assert_canaries(outputs, context)
    for index, (storage, snapshot) in enumerate(zip(inputs.storages, snapshots)):
        torch.testing.assert_close(storage.cpu(), snapshot, rtol=0, atol=0, msg=f"{context} input storage {index}")
    return outputs


@pytest.mark.parametrize(
    ("batch", "cores", "expected"),
    [(0, 48, 2), (1, 48, 2), (2, 48, 2), (3, 48, 4), (47, 48, 48), (48, 48, 48), (97, 48, 48), (3, 2, 2)],
)
def test_batch_aware_task_count(runner: _KernelRunner, batch: int, cores: int, expected: int) -> None:
    assert runner._select_task_count(batch, cores) == expected


@pytest.mark.parametrize(("batch", "cores"), [(-1, 48), (_INT32_MAX + 1, 48), (1, 0), (1, 3), (1, _INT32_MAX + 1)])
def test_batch_aware_task_count_rejects_invalid(runner: _KernelRunner, batch: int, cores: int) -> None:
    with pytest.raises(ValueError):
        runner._select_task_count(batch, cores)


@pytest.mark.parametrize("batch", [1, 3])
def test_batch_aware_task_count_numerical(runner: _KernelRunner, batch: int) -> None:
    inputs = _mtp_inputs(*_logical_ids(batch, 3, "mixed"), runner.device)
    expected = _semantic_oracle(inputs.draft.cpu(), inputs.target.cpu(), inputs.bonus.cpu(), True)
    outputs = runner._run(inputs, mask=True)
    context = _context(inputs, True, runner._select_task_count(batch))
    _assert_pair((outputs.full, outputs.masked), expected, context)
    _assert_canaries(outputs, context)


def test_kernel_metadata_is_int32(runner: _KernelRunner) -> None:
    implementation = importlib.import_module("xllm.python.kernels_npu.tilelang.greedy_prefix_verify")
    primitive = implementation.build_greedy_prefix_verify_kernel(task_count=2)
    scalar_parameters = [parameter for parameter in primitive.params if str(parameter.dtype) != "handle"]
    assert len(scalar_parameters) == 8
    assert all(str(parameter.dtype) == "int32" for parameter in scalar_parameters)
    buffers = [primitive.buffer_map[parameter] for parameter in primitive.params if str(parameter.dtype) == "handle"]
    assert [str(buffer.dtype) for buffer in buffers] == ["int64", "int64", "int64", "int32", "int32"]
    assert all(str(extent.dtype) == "int32" for buffer in buffers for extent in buffer.shape)
    for buffer in buffers:
        for extent in buffer.shape:
            variables = implementation.tvm.tir.analysis.undefined_vars(extent)
            assert all(any(variable.same_as(parameter) for parameter in scalar_parameters) for variable in variables)


def test_kernel_metadata_generated_abi_is_int32(runner: _KernelRunner) -> None:
    inputs = _mtp_inputs(*_logical_ids(3, 3, "mixed"), runner.device)
    source = runner._get_kernel(inputs, task_count=2).get_kernel_source()
    for function in ("greedy_prefix_verify_kernel", "call"):
        signature = re.search(rf"\b{function}\s*\(([^)]*)\)\s*\{{", source)
        assert signature is not None, f"Missing generated {function} definition"
        parameters = [parameter.strip() for parameter in signature.group(1).split(",")]
        assert len(parameters) == 14, signature.group(0)
        for parameter in parameters[5:13]:
            assert re.fullmatch(r"(?:int32_t|int)\s+\w+", parameter), signature.group(0)
        assert "int64_t" not in ",".join(parameters[5:13]), signature.group(0)
    gathers = re.findall(r"AscendC::Gather\([^;]+;", source)
    assert len(gathers) == 2, source
    assert all(re.search(r",\s*64\s*\);$", gather) for gather in gathers), gathers
    assert not re.search(r"\(\s*int64_t\s*\)", source), "Generated scalar indices must remain INT32"


@pytest.mark.parametrize("mask", [True, False], ids=["masked", "full-only"])
def test_smoke_mtp_int64(runner: _KernelRunner, mask: bool) -> None:
    target = torch.tensor([[11, 25, 33], [101, 102, 103], [201, 202, 203]], dtype=torch.int64)
    draft = torch.tensor([[11, 22, 33], [99, 102, 103], [201, 202, 203]], dtype=torch.int64)
    bonus = torch.tensor([[44], [104], [204]], dtype=torch.int64)
    inputs = _mtp_inputs(draft, target, bonus, runner.device)
    outputs = _check_case(runner, inputs, mask)
    torch.testing.assert_close(
        outputs.full.cpu(),
        torch.tensor([[11, 25, 33, 44], [101, 102, 103, 104], [201, 202, 203, 204]], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    if mask:
        assert outputs.masked is not None
        torch.testing.assert_close(
            outputs.masked.cpu(),
            torch.tensor([[11, 25, -1, -1], [101, -1, -1, -1], [201, 202, 203, 204]], dtype=torch.int32),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("mask", [True, False])
def test_smoke_bonus_only(runner: _KernelRunner, batch: int, mask: bool) -> None:
    inputs = _mtp_inputs(*_logical_ids(batch, 0, "all"), runner.device)
    _check_case(runner, inputs, mask)


@pytest.mark.parametrize("width", [0, 3])
@pytest.mark.parametrize("mask", [True, False])
def test_smoke_empty_batch_no_launch(runner: _KernelRunner, width: int, mask: bool) -> None:
    inputs = _mtp_inputs(*_logical_ids(0, width, "all"), runner.device)
    kernels_before = len(runner._kernels)
    _check_case(runner, inputs, mask)
    assert len(runner._kernels) == kernels_before, "B=0 must not even request a JIT specialization"


@pytest.mark.parametrize(
    ("batch", "width", "rejection"),
    [
        (1, 1, "last"),
        (3, 3, "all"),
        (17, 7, "rematch"),
        (1, 31, "middle"),
        (1, 32, "middle"),
        (1, 33, "middle"),
        (49, 63, "middle"),
        (97, 64, "first"),
        (3, 65, "last"),
        (17, 129, "mixed"),
    ],
)
@pytest.mark.parametrize("mask", [True, False])
def test_int64_boundaries(runner: _KernelRunner, batch: int, width: int, rejection: str, mask: bool) -> None:
    inputs = _mtp_inputs(*_logical_ids(batch, width, rejection, edge_values=True), runner.device)
    _check_case(runner, inputs, mask)


@pytest.mark.parametrize("rejection", ["first", "middle", "last", "all", "rematch"])
def test_int64_prefix_patterns(runner: _KernelRunner, rejection: str) -> None:
    inputs = _mtp_inputs(*_logical_ids(3, 129, rejection, edge_values=True), runner.device)
    _check_case(runner, inputs, mask=True)


@pytest.mark.parametrize(("batch", "width"), [(3, 65), (49, 64), (97, 129)])
@pytest.mark.parametrize("mask", [True, False])
def test_default_task_boundary_subset(runner: _KernelRunner, batch: int, width: int, mask: bool) -> None:
    inputs = _mtp_inputs(*_logical_ids(batch, width, "mixed"), runner.device)
    _check_case(runner, inputs, mask, task_count=runner.default_task_count)


@pytest.mark.parametrize("bits", list(product((32, 64), repeat=3)), ids=lambda bits: "-".join(map(str, bits)))
def test_small_dtype_matrix(runner: _KernelRunner, bits: tuple[int, int, int]) -> None:
    inputs = _mtp_inputs(*_logical_ids(3, 7, "rematch", edge_values=True), runner.device, bits=bits)
    for mask in (True, False):
        _check_case(runner, inputs, mask)
    assert tuple(tensor.element_size() * 8 for tensor in (inputs.draft, inputs.target, inputs.bonus)) == bits


@pytest.mark.parametrize(
    ("width", "column_stride"),
    [(3, 2), (33, 2), (65, 3), (3, 65537)],
    ids=["non-unit", "window-boundary", "multiple-windows", "huge"],
)
@pytest.mark.parametrize("mask", [True, False])
def test_int64_independent_strided_storage(runner: _KernelRunner, width: int, column_stride: int, mask: bool) -> None:
    draft_cpu, target_cpu, bonus_cpu = _logical_ids(3, width, "rematch", edge_values=True)
    row_span = (width - 1) * column_stride
    draft, draft_storage = _strided_input(draft_cpu, 64, runner.device, (row_span + 11, column_stride), offset=17)
    target, target_storage = _strided_input(target_cpu, 64, runner.device, (row_span + 19, column_stride), offset=23)
    bonus, bonus_storage = _strided_input(bonus_cpu, 64, runner.device, (5, 2), offset=19)
    inputs = _Inputs(draft, target, bonus, (draft_storage, target_storage, bonus_storage))
    _check_case(runner, inputs, mask)


@pytest.mark.parametrize("layout", ["all-zero-strides", "broadcast-rows", "read-only-alias"])
@pytest.mark.parametrize("mask", [True, False])
def test_int64_broadcast_and_alias_inputs(runner: _KernelRunner, layout: str, mask: bool) -> None:
    if layout == "all-zero-strides":
        target_base, target_storage = _strided_input(torch.tensor([[16777217]]), 64, runner.device, (1, 1))
        draft_base, draft_storage = _strided_input(torch.tensor([[16777216]]), 64, runner.device, (1, 1))
        bonus_base, bonus_storage = _strided_input(torch.tensor([[_INT32_MAX]]), 64, runner.device, (1, 1))
        inputs = _Inputs(
            draft_base.expand(17, 65),
            target_base.expand(17, 65),
            bonus_base.expand(17, 1),
            (draft_storage, target_storage, bonus_storage),
        )
    else:
        base = _mtp_inputs(*_logical_ids(1, 65, "rematch", edge_values=True), runner.device)
        draft = base.target if layout == "read-only-alias" else base.draft
        inputs = _Inputs(draft.expand(17, 65), base.target.expand(17, 65), base.bonus.expand(17, 1), base.storages)
    _check_case(runner, inputs, mask)


@pytest.mark.parametrize("mask", [True, False])
def test_int64_limits_and_adjacent_ids_are_exact(runner: _KernelRunner, mask: bool) -> None:
    target = torch.tensor(
        [
            [16777217, 16777216, _INT32_MAX, _INT32_MIN],
            [_INT32_MAX, _INT32_MIN, 16777216, 16777217],
            [_INT32_MIN, _INT32_MAX, -16777217, -16777216],
        ],
        dtype=torch.int64,
    )
    draft = target.clone()
    draft[0, 0] = 16777216
    draft[1, 1] = _INT32_MIN + 1
    bonus = torch.tensor([[_INT32_MAX], [_INT32_MIN], [16777217]], dtype=torch.int64)
    _check_case(runner, _mtp_inputs(draft, target, bonus, runner.device), mask)


def test_int64_repeated_calls_reset_prefix_state(runner: _KernelRunner) -> None:
    for index in range(12):
        rejection = ("first", "all", "middle", "last")[index % 4]
        inputs = _mtp_inputs(*_logical_ids(3, 129, rejection, edge_values=True), runner.device)
        _check_case(runner, inputs, mask=index % 3 != 0)


def test_int64_current_nondefault_stream(runner: _KernelRunner) -> None:
    old_values = _logical_ids(17, 65, "all")
    new_values = _logical_ids(17, 65, "rematch", edge_values=True)
    inputs = _mtp_inputs(*old_values, runner.device)
    staged = tuple(tensor.to(runner.device) for tensor in new_values)
    runner._get_kernel(inputs, task_count=2)
    torch.npu.synchronize()
    expected = _semantic_oracle(*new_values, mask=True)
    context = _context(inputs, mask=True, task_count=2)
    stream = torch.npu.Stream(device=runner.device)
    with torch.npu.stream(stream):
        # Producers, F1 and consumers must all obey this stream. No host wait
        # separates the input updates, the kernel, and the output consumers.
        for destination, source in zip((inputs.draft, inputs.target, inputs.bonus), staged):
            destination.copy_(source)
        outputs = runner._run(inputs, mask=True, task_count=2)
        assert outputs.masked is not None
        consumed = (outputs.full.clone(), outputs.masked.clone())
        reference = _torch_reference(inputs, mask=True)
    stream.synchronize()
    _assert_pair(consumed, expected, f"nondefault stream {context}")
    _assert_pair(reference, expected, f"nondefault stream old Torch {context}")
    _assert_canaries(outputs, context)
    for result, planned in zip((inputs.draft, inputs.target, inputs.bonus), new_values):
        torch.testing.assert_close(result.cpu(), planned, rtol=0, atol=0, msg=f"stream input {context}")


@pytest.fixture(scope="module")
def triton_reference_factory() -> tuple[torch.device, Callable[[int, int], Any]]:
    importlib.import_module("torch_npu")
    assert torch.npu.is_available(), "Triton precision tests require an available Ascend NPU"
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    adapter = importlib.import_module("tools.greedy_prefix_verify_triton_reference")

    def _make_reference(batch: int, width: int) -> Any:
        return adapter.TritonGreedyReference(
            root=None, batch=batch, width=width, device=device, metadata_policy="cached"
        )

    return device, _make_reference


@pytest.mark.parametrize(
    ("batch", "width", "rejection", "bits"),
    [
        pytest.param(3, 1, "all", _ALL_INT64, id="k1-small-all-int64"),
        pytest.param(3, 1, "first", (32, 64, 32), id="k1-small-first-native"),
        pytest.param(97, 1, "all", (32, 64, 32), id="k1-large-all-native"),
        pytest.param(97, 1, "first", _ALL_INT64, id="k1-large-first-int64"),
        pytest.param(3, 3, "all", _ALL_INT64, id="general-small-all-int64"),
        pytest.param(3, 7, "first", (32, 64, 32), id="general-small-first-native"),
        pytest.param(97, 3, "rematch", _ALL_INT64, id="general-large-rematch-int64"),
        pytest.param(97, 7, "rematch", (32, 64, 32), id="general-large-rematch-native"),
    ],
)
def test_triton_transplant_exact_outputs(
    triton_reference_factory: tuple[torch.device, Callable[[int, int], Any]],
    batch: int,
    width: int,
    rejection: str,
    bits: tuple[int, int, int],
) -> None:
    device, make_reference = triton_reference_factory
    inputs = _mtp_inputs(*_logical_ids(batch, width, rejection, edge_values=True), device, bits=bits)
    snapshots = tuple(storage.cpu() for storage in inputs.storages)
    expected = _semantic_oracle(inputs.draft.cpu(), inputs.target.cpu(), inputs.bonus.cpu(), mask=True)
    assert expected[1] is not None
    reference = make_reference(batch, width)
    context = (
        f"Triton B={batch} K={width} rejection={rejection} bits={bits} "
        f"strides={[tensor.stride() for tensor in (inputs.draft, inputs.target, inputs.bonus)]} "
        f"kernel={reference.metadata['kernel']} grid={reference.metadata['grid']} "
        f"BLOCK_SIZE={reference.metadata['BLOCK_SIZE']}"
    )
    logger.info(f"Pinned Triton precision: {context}")
    complete = reference.verify(inputs.draft, inputs.target, inputs.bonus)
    torch.npu.synchronize()
    _assert_pair(complete, expected, f"complete {context}")
    for index, (storage, snapshot) in enumerate(zip(inputs.storages, snapshots)):
        torch.testing.assert_close(
            storage.cpu(), snapshot, rtol=0, atol=0, msg=f"complete {context} input storage {index}"
        )

    # The original Triton entrypoints take flat contiguous inputs, unlike F1.
    flat_inputs = tuple(tensor.contiguous().view(-1) for tensor in (inputs.draft, inputs.target, inputs.bonus))
    flat_snapshots = tuple(tensor.cpu() for tensor in flat_inputs)
    size = batch * (width + 1)
    output_storage = torch.full((size + 2 * _GUARD_SIZE,), _CANARY, dtype=torch.int32, device=device)
    output = output_storage[_GUARD_SIZE : _GUARD_SIZE + size].view(batch, width + 1)
    # Reuse output so both launches exercise the required -1 initialization.
    for iteration in range(2):
        output.fill_(-1)
        reference.launch_kernel(output, *flat_inputs)
        torch.npu.synchronize()
        torch.testing.assert_close(
            output.cpu(), expected[1], rtol=0, atol=0, msg=f"native launch {iteration} {context} masked"
        )
        _assert_canaries(_Outputs(output, None, (output_storage,)), context)
    for index, (tensor, snapshot) in enumerate(zip(flat_inputs, flat_snapshots)):
        torch.testing.assert_close(tensor.cpu(), snapshot, rtol=0, atol=0, msg=f"{context} flat input {index}")
    for index, (storage, snapshot) in enumerate(zip(inputs.storages, snapshots)):
        torch.testing.assert_close(storage.cpu(), snapshot, rtol=0, atol=0, msg=f"{context} input storage {index}")
