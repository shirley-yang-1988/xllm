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

"""Test-only adapter for the standalone, pinned original Triton verifier.

No vLLM installation or remote reference checkout is required. A supplied root
is an optional source audit, never an import path or a synchronization claim.
The transplanted function bodies are checked against pinned source hashes.
"""

import ast
import hashlib
import importlib
import subprocess
from pathlib import Path
from typing import Any

import torch

REFERENCE_REVISION = "083d66780c0909ae97d654660ab38665ec5aa31a"


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments], check=True, stdout=subprocess.PIPE, text=True
    ).stdout.strip()


def _audit_transplant(module_path: Path, expected: dict[str, str]) -> dict[str, str]:
    source = module_path.read_text()
    lines = source.splitlines(keepends=True)
    actual = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.FunctionDef) or node.name not in expected:
            continue
        start = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list])
        block = "".join(lines[start - 1 : node.end_lineno])
        actual[node.name] = hashlib.sha256(block.encode()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"The standalone Triton reference differs from its pinned function hashes: {actual}")
    return actual


def _audit_checkout(root: Path | None, expected: dict[str, str]) -> dict[str, Any]:
    if root is None:
        return {"root": None, "checkout_audit": "NOT_REQUESTED"}
    root = root.resolve(strict=True)
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("vllm-ascend-root must be the repository root")
    revision = _git(root, "rev-parse", "HEAD")
    if revision != REFERENCE_REVISION:
        raise RuntimeError(f"Expected vLLM-Ascend {REFERENCE_REVISION}, got {revision}")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all", "--", *expected)
    if dirty:
        raise RuntimeError(f"The pinned Triton reference source is not clean:\n{dirty}")
    actual = {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in expected}
    if actual != expected:
        raise RuntimeError(f"The reference checkout source hashes differ from the pinned source: {actual}")
    return {
        "root": str(root),
        "checkout_audit": "PASS",
        "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "reference_source_clean": True,
    }


class TritonGreedyReference:
    """One fixed B/K case; cache shape-only metadata, never request token data."""

    def __init__(self, root: Path | None, batch: int, width: int, device: torch.device, metadata_policy: str) -> None:
        if batch <= 0 or width <= 0:
            raise ValueError("Triton comparisons require B > 0 and K > 0; empty cases are N/A")
        if batch * width > torch.iinfo(torch.int32).max:
            raise ValueError("The reference cumulative draft offsets must fit INT32")
        if metadata_policy not in ("cached", "per-call"):
            raise ValueError(f"Unknown metadata policy: {metadata_policy}")
        # Keep Triton optional until the benchmark explicitly requests this reference.
        self._module = importlib.import_module("tools.greedy_prefix_verify_triton_kernel")
        if self._module.REFERENCE_REVISION != REFERENCE_REVISION:
            raise RuntimeError("The adapter and transplanted kernel pin different source revisions")
        module_path = Path(self._module.__file__).resolve()
        function_hashes = _audit_transplant(module_path, self._module.REFERENCE_FUNCTION_SHA256)
        checkout_audit = _audit_checkout(root, self._module.REFERENCE_SOURCE_SHA256)
        self._module.init_device_properties_triton()
        self._batch = batch
        self._width = width
        self._device = device
        self._metadata_policy = metadata_policy
        self._cached_metadata = self._make_metadata() if metadata_policy == "cached" else None
        self._grid, self._block_size = self._module.cal_grid_and_block_size(batch)
        self._kernel = (
            self._module.rejection_greedy_sample_spec_len_1_triton
            if width == 1
            else self._module.rejection_greedy_sample_triton
        )
        # Kernel-only timing always prepares shape metadata outside its interval.
        self._kernel_metadata = self._make_metadata()
        triton = self._module.triton
        self.metadata = {
            **checkout_audit,
            "reference_kind": "standalone_transplant_not_full_vllm_integration",
            "revision": REFERENCE_REVISION,
            "imported_module": str(module_path),
            "transplant_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
            "source_sha256": dict(self._module.REFERENCE_SOURCE_SHA256),
            "function_source_sha256": function_hashes,
            "function_source_audit": "PASS",
            "function_hash_scope": "Exact source bytes from first decorator/def through final body line, including newline",
            "dependency_adaptations": [
                "Direct triton and triton.language imports; no vLLM modules",
                "Original get_element extension-then-language lookup order",
                "Original active-driver device query, retaining only vectorcore count",
                "Integer next_power_of_2 helper without the vLLM import",
            ],
            "synchronization": "These xLLM benchmark-tool files are synchronized with xLLM; no vLLM checkout is synchronized.",
            "triton_version": triton.__version__,
            "triton_module": triton.__file__,
            "vectorcore_num": self._module.get_vectorcore_num(),
            "kernel": self._kernel.fn.__name__,
            "jit_cache_key": str(self._kernel.cache_key),
            "grid": [self._grid],
            "BLOCK_SIZE": self._block_size,
            "synthetic_mode": False,
            "is_greedy": None,
            "metadata_policy": metadata_policy,
            "metadata": "num_draft_tokens=[K]*B; cumulative ends=(1..B)*K, INT32, no leading zero",
            "target_int32": "INT32 specialization, not the upstream argmax INT64 production dtype",
        }

    def _make_metadata(self) -> tuple[list[int], torch.Tensor]:
        return [self._width] * self._batch, torch.arange(
            self._width,
            (self._batch + 1) * self._width,
            self._width,
            dtype=torch.int32,
            device=self._device,
        )

    def verify(
        self, draft: torch.Tensor, target: torch.Tensor, bonus: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same INT32 outputs; input layout copies and output casts are timed."""
        flat_draft = draft.contiguous().view(-1)
        flat_target = target.contiguous().view(-1)
        flat_bonus = bonus.contiguous().view(-1)
        metadata = self._cached_metadata if self._cached_metadata is not None else self._make_metadata()
        num_draft_tokens, cumulative = metadata
        output = torch.empty((self._batch, self._width + 1), dtype=torch.int32, device=self._device)
        output.fill_(-1)
        grid, block_size = self._module.cal_grid_and_block_size(self._batch)
        self._module.rejection_greedy_sample_with_triton(
            output,
            num_draft_tokens,
            cumulative,
            flat_draft,
            flat_target,
            flat_bonus,
            None,
            self._width,
            grid,
            block_size,
            uniform_probs=None,
            synthetic_conditional_rates=None,
            synthetic_mode=False,
        )
        return torch.cat((target, bonus), dim=1).to(torch.int32), output

    def launch_kernel(
        self, output: torch.Tensor, flat_draft: torch.Tensor, flat_target: torch.Tensor, flat_bonus: torch.Tensor
    ) -> Any:
        """Launch the dispatcher's original branch, with prepared arguments.

        The caller must fill output with -1 before EVERY launch. For kernel-only
        device timing, report that fill separately: rejected tails are unwritten.
        """
        if self._width == 1:
            return self._kernel[(self._grid,)](
                output,
                flat_draft,
                flat_target,
                flat_bonus,
                self._batch,
                None,
                None,
                SYNTHETIC_MODE=False,
                BLOCK_SIZE=self._block_size,
            )
        return self._kernel[(self._grid,)](
            output,
            self._kernel_metadata[1],
            flat_draft,
            flat_target,
            flat_bonus,
            None,
            self._batch,
            self._width,
            None,
            None,
            SYNTHETIC_MODE=False,
            BLOCK_SIZE=self._block_size,
        )
