#!/usr/bin/env python3
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

"""Profile the test-only callable of the F1 builder with native integer inputs.

Run in the selected NPU container from the exact Git commit pushed for testing,
using a background fork. Compilation, input construction, correctness checks
and initial warmup are outside the active profile interval.
Shapes, strides and relative spans use checked INT32 indices; native token inputs
remain INT32/INT64 and are never narrowed by the launch adapter.

The primary measurement is Duration(us) from NPU kernel_details.csv, collected
independently per implementation/group in AB/BA order. Unknown schemas fail
explicitly, retaining raw evidence. --collect-intervals additionally records
unprofiled event/host/wall intervals; those are not device-kernel durations.
These measurements describe Python/JIT execution, not the C++ AOT registry or
MTP integration. Use the C++ benchmark and service trace for those paths.

--compare-triton opts into the standalone transplant of the pinned default
Triton verifier. --vllm-ascend-root optionally audits its upstream source.
The same-contract adapter includes layout copies, fill(-1), the reference host
dispatcher, and full output construction/conversion to INT32. Native kernel-only rows have different
output work and are not an equal-contract winner/loser comparison. The caller
must check the selected card has >2 GB free HBM and utilization 0 before running;
per-group npu-smi snapshots are retained for interference review, not auto-rated.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import tilelang
import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.logger import logger
from tools.greedy_prefix_verify_triton_reference import TritonGreedyReference
from xllm.python.kernels_npu.tilelang.greedy_prefix_verify import (
    DEFAULT_TASK_COUNT,
    GREEDY_PREFIX_VERIFY_PASS_CONFIGS,
    build_greedy_prefix_verify_kernel,
)

_INT32_MAX = (1 << 31) - 1


@dataclass(frozen=True)
class _Inputs:
    draft: torch.Tensor
    target: torch.Tensor
    bonus: torch.Tensor
    flat_draft: torch.Tensor
    flat_target: torch.Tensor
    flat_bonus: torch.Tensor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--draft-length", type=int, default=3)
    parser.add_argument("--mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--layout", choices=("contiguous", "mtp"), default="mtp")
    parser.add_argument("--draft-bits", type=int, choices=(32, 64), default=64)
    parser.add_argument("--target-bits", type=int, choices=(32, 64), default=64)
    parser.add_argument("--bonus-bits", type=int, choices=(32, 64), default=64)
    parser.add_argument("--rejection", choices=("first", "middle", "last", "all", "rematch", "mixed"), default="mixed")
    parser.add_argument("--id-base", type=int, default=100)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--iterations", type=int, default=100, help="Active calls per isolated operation/group profile")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--groups", type=int, default=3)
    parser.add_argument("--collect-intervals", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--samples", type=int, default=100, help="Supplementary interval samples per paired group")
    parser.add_argument("--amortized-calls", type=int, default=100)
    parser.add_argument("--compare-triton", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--vllm-ascend-root", type=Path, help="Optional upstream audit checkout; requires --compare-triton"
    )
    parser.add_argument("--triton-native-dtype", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metadata-policy", choices=("cached", "per-call"), default="cached")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--task-count", type=int, default=DEFAULT_TASK_COUNT)
    parser.add_argument("--kernel-only", action="store_true", help="Profile only the preallocated native F1 kernel")
    parser.add_argument("--compare-task-count", type=int, help="Pair kernel-only runs with another task count")
    parser.add_argument("--baseline-revision", help="Pair kernel-only runs with the builder from this Git revision")
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument(
        "--aot-source", type=Path, help="Optional generated AOT source for a separately identified comparison"
    )
    args = parser.parse_args()
    if args.batch_size < 0 or args.draft_length < 0:
        parser.error("batch-size and draft-length must be nonnegative")
    if min(args.iterations, args.groups, args.samples, args.amortized_calls) <= 0 or args.warmup < 0 or args.device < 0:
        parser.error("iterations/groups/samples/amortized-calls must be positive; warmup/device must be nonnegative")
    if not 0 < args.task_count <= _INT32_MAX or args.task_count % 2 != 0:
        parser.error("task-count must be a positive even INT32 integer")
    if args.kernel_only and (args.compare_triton or args.collect_intervals):
        parser.error("kernel-only cannot be combined with compare-triton or collect-intervals")
    if args.compare_task_count is not None:
        if not args.kernel_only:
            parser.error("compare-task-count requires kernel-only")
        if not 0 < args.compare_task_count <= _INT32_MAX or args.compare_task_count % 2 != 0:
            parser.error("compare-task-count must be a positive even INT32 integer")
        if args.compare_task_count == args.task_count:
            parser.error("compare-task-count must differ from task-count")
    if args.baseline_revision is not None:
        if not args.kernel_only or args.compare_task_count is not None:
            parser.error("baseline-revision requires kernel-only without compare-task-count")
        if len(args.baseline_revision) != 40 or any(c not in "0123456789abcdef" for c in args.baseline_revision):
            parser.error("baseline-revision must be a full lowercase Git commit SHA")
    max_id = args.id_base + args.batch_size * (args.draft_length + 1)
    if not 0 < args.vocab_size <= torch.iinfo(torch.int32).max + 1 or args.id_base < 0 or max_id >= args.vocab_size:
        parser.error("generated IDs, including mismatches, must lie in the vocabulary and fit in INT32")
    if args.vllm_ascend_root is not None:
        if not args.compare_triton:
            parser.error("vllm-ascend-root is audit-only and requires --compare-triton")
        if not args.vllm_ascend_root.is_dir():
            parser.error("vllm-ascend-root must name the pinned upstream audit checkout")
    if args.profile_dir.exists():
        parser.error("profile-dir must be a new directory; existing evidence is never overwritten")
    if args.aot_source is not None and not args.aot_source.is_file():
        parser.error("aot-source must name an existing generated source file")
    return args


def _checked_span(shape: tuple[int, ...], strides: tuple[int, ...], label: str) -> int:
    if len(shape) != len(strides):
        raise ValueError(f"{label}: shape and stride ranks differ")
    for name, values in (("shape", shape), ("stride", strides)):
        if any(value < 0 or value > _INT32_MAX for value in values):
            raise ValueError(f"{label}: {name} must fit nonnegative INT32, got {values}")
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
    return tensor.as_strided((span,), (1,), offset)


def _make_inputs(args: argparse.Namespace) -> _Inputs:
    batch_size, width = args.batch_size, args.draft_length
    # Validate before allocating even when B=0; K+1 is an INT32 kernel value.
    output_span = _checked_span((batch_size, width + 1), (width + 1, 1), "output")
    backing_cpu = torch.arange(output_span, dtype=torch.int32).view(batch_size, width + 1)
    backing_cpu += args.id_base
    draft_cpu = backing_cpu[:, :width].clone()
    for row in range(batch_size):
        rejection = {
            "first": 0,
            "middle": width // 2,
            "last": max(width - 1, 0),
            "all": width,
            "rematch": 0,
            "mixed": row % (width + 1),
        }[args.rejection]
        if rejection < width:
            draft_cpu[row, rejection] += 1

    device = torch.device(f"npu:{args.device}")
    draft_dtype = torch.int32 if args.draft_bits == 32 else torch.int64
    target_dtype = torch.int32 if args.target_bits == 32 else torch.int64
    bonus_dtype = torch.int32 if args.bonus_bits == 32 else torch.int64
    draft = draft_cpu.to(device=device, dtype=draft_dtype)
    if args.layout == "mtp":
        backing = backing_cpu.to(device=device, dtype=target_dtype)
        target = backing[:, :width]
        # Mixed target/bonus dtypes need independent storage, but retain strides.
        bonus_backing = backing if target_dtype == bonus_dtype else backing_cpu.to(device=device, dtype=bonus_dtype)
        bonus = bonus_backing.view(-1)[width :: width + 1].view(batch_size, 1)
    else:
        target = backing_cpu[:, :width].contiguous().to(device=device, dtype=target_dtype)
        bonus = backing_cpu[:, width:].contiguous().to(device=device, dtype=bonus_dtype)
    return _Inputs(draft, target, bonus, *(_flat_storage_view(tensor) for tensor in (draft, target, bonus)))


def _semantic_oracle(inputs: _Inputs, mask: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    draft = inputs.draft.cpu()
    target = inputs.target.cpu()
    bonus = inputs.bonus.cpu()
    full = torch.cat((target, bonus), dim=1).to(torch.int32)
    if not mask:
        return full, None
    masked = full.clone()
    for row in range(draft.shape[0]):
        for position in range(draft.shape[1]):
            if int(draft[row, position]) != int(target[row, position]):
                masked[row, position + 1 :] = -1
                break
    return full, masked


def _torch_reference(inputs: _Inputs, mask: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    # Include output conversion in the measured equal-contract interval.
    full = torch.cat((inputs.target, inputs.bonus), dim=-1).to(torch.int32)
    if not mask:
        return full, None
    # Preserve input dtypes for the old Torch equality/mask/index chain.
    accepted = (inputs.target == inputs.draft).to(torch.int64)
    combined = torch.cat((accepted, torch.zeros_like(inputs.bonus, dtype=torch.int64)), dim=1)
    first_reject = (1 - combined).argmax(dim=1, keepdim=True)
    positions = torch.arange(full.shape[1], device=full.device).unsqueeze(0)
    return full, torch.where(positions <= first_reject, full, -1)


def _allocate_outputs(inputs: _Inputs, mask: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    shape = (inputs.target.shape[0], inputs.target.shape[1] + 1)
    _checked_span(shape, (shape[1], 1), "output")
    full = torch.empty(shape, dtype=torch.int32, device=inputs.target.device)
    return full, torch.empty_like(full) if mask else None


def _kernel_args(inputs: _Inputs, full: torch.Tensor, masked: torch.Tensor | None) -> tuple[Any, ...]:
    batch, width = inputs.target.shape
    _checked_span((batch, width + 1), (width + 1, 1), "output")
    for name in ("draft", "target", "bonus"):
        tensor = getattr(inputs, name)
        span = _checked_span(tuple(tensor.shape), tensor.stride(), name)
        if getattr(inputs, f"flat_{name}").numel() != span:
            raise ValueError(f"{name}: flat view does not match checked physical span {span}")
    for tensor in (full,) if masked is None else (full, masked):
        _checked_span(tuple(tensor.shape), tensor.stride(), "output")
    return (
        inputs.flat_draft,
        inputs.flat_target,
        inputs.flat_bonus,
        full.view(-1),
        (full if masked is None else masked).view(-1),
        inputs.target.shape[0],
        inputs.target.shape[1],
        inputs.draft.stride(0),
        inputs.draft.stride(1),
        inputs.target.stride(0),
        inputs.target.stride(1),
        inputs.bonus.stride(0),
        int(masked is not None),
    )


def _candidate(kernel: Any, inputs: _Inputs, mask: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    full, masked = _allocate_outputs(inputs, mask)
    if inputs.target.shape[0] != 0:
        kernel(*_kernel_args(inputs, full, masked))
    return full, masked


def _check_outputs(
    actual: tuple[torch.Tensor, torch.Tensor | None], expected: tuple[torch.Tensor, torch.Tensor | None]
) -> None:
    for result, reference in zip(actual, expected):
        if reference is None:
            if result is not None:
                raise AssertionError("mask=false must not return a second output")
            continue
        if result is None:
            raise AssertionError("mask=true requires both outputs")
        if result.dtype != torch.int32 or reference.dtype != torch.int32:
            raise AssertionError("token outputs and semantic references must be INT32")
        torch.testing.assert_close(result.cpu(), reference, rtol=0, atol=0)


def _tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


@dataclass(frozen=True)
class _Operation:
    name: str
    run: Callable[[], Any]
    check: Callable[[Any], None]
    reset: Callable[[], Any] | None = None


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"Expected nonempty, finite positive duration samples, got {values}")
    ordered = sorted(values)
    return {
        "samples": len(values),
        "median_us": statistics.median(values),
        "p95_us": ordered[math.ceil(0.95 * len(values)) - 1],
    }


def _read_device_kernels(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # This is the torch_npu device export schema, not CPU record_function events.
    paths = sorted(directory.rglob("kernel_details.csv"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one kernel_details.csv in {directory}, found {paths}; raw traces retained")
    path = paths[0]
    required = {"Device_id", "Name", "Type", "Accelerator Core", "Start Time(us)", "Duration(us)"}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if fields is None or len(fields) != len(set(fields)) or not required.issubset(fields):
            raise RuntimeError(f"Unsupported device-kernel schema in {path}: {fields}; raw traces retained")
        records = []
        for line, row in enumerate(reader, start=2):
            if None in row or any(row[field] is None or not row[field].strip() for field in required):
                raise RuntimeError(f"Incomplete device-kernel record at {path}:{line}: {row}")
            try:
                start = Decimal(row["Start Time(us)"].strip())
                duration = float(row["Duration(us)"])
            except (InvalidOperation, ValueError) as error:
                raise RuntimeError(f"Invalid device timestamps/duration at {path}:{line}: {row}") from error
            if not start.is_finite() or start < 0 or not math.isfinite(duration) or duration <= 0:
                raise RuntimeError(f"Invalid device timestamps/duration at {path}:{line}: {row}")
            records.append(
                {
                    "csv_line": line,
                    "device_id": row["Device_id"].strip(),
                    "name": row["Name"].strip(),
                    "type": row["Type"].strip(),
                    "accelerator_core": row["Accelerator Core"].strip(),
                    "start_us": str(start),
                    "duration_us": duration,
                }
            )
    if not records or len({row["device_id"] for row in records}) != 1:
        raise RuntimeError(f"Expected nonempty single-device kernel records in {path}; raw traces retained")
    records.sort(key=lambda row: Decimal(row["start_us"]))
    return {
        "path": str(path),
        "columns": fields,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "duration_column": "Duration(us)",
        "unit": "us",
    }, records


def _kernel_identity(record: dict[str, Any]) -> tuple[str, ...]:
    return tuple(record[field] for field in ("device_id", "name", "type", "accelerator_core"))


def _split_device_calls(records: list[dict[str, Any]], calls: int, label: str) -> list[list[dict[str, Any]]]:
    if not records or len(records) % calls:
        raise RuntimeError(f"{label}: {len(records)} device kernels cannot be attributed to {calls} calls")
    count = len(records) // calls
    chunks = [records[index : index + count] for index in range(0, len(records), count)]
    signature = [_kernel_identity(row) for row in chunks[0]]
    if any([_kernel_identity(row) for row in chunk] != signature for chunk in chunks[1:]):
        raise RuntimeError(f"{label}: device-kernel sequence varies across calls; attribution is ambiguous")
    return chunks


def _device_summary(chunks: list[list[dict[str, Any]]]) -> dict[str, Any]:
    kernels: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for chunk in chunks:
        for row in chunk:
            kernels.setdefault(_kernel_identity(row), []).append(row)
    totals = [sum(row["duration_us"] for row in chunk) for chunk in chunks]
    return {
        "calls": len(chunks),
        "device_kernel_count": sum(len(chunk) for chunk in chunks),
        "kernels": [
            {
                "device_id": identity[0],
                "name": identity[1],
                "type": identity[2],
                "accelerator_core": identity[3],
                "count": len(rows),
                "durations_us": [row["duration_us"] for row in rows],
                "summary": _summary([row["duration_us"] for row in rows]),
            }
            for identity, rows in kernels.items()
        ],
        "device_duration_sum_us_per_call": {"raw": totals, **_summary(totals)},
        "scope": "Sum of device-kernel Duration(us), not wall span, host time, event elapsed or queue gaps.",
    }


def _capture_device_profile(
    args: argparse.Namespace, operation: _Operation, directory: Path, reset_only: bool = False
) -> tuple[dict[str, Any], list[list[dict[str, Any]]]]:
    def run() -> Any:
        if operation.reset is not None:
            operation.reset()
        return None if reset_only else operation.run()

    if reset_only and operation.reset is None:
        raise ValueError(f"{operation.name} has no reset operation")
    for _ in range(args.warmup):
        run()
    torch.npu.synchronize()
    directory.mkdir(parents=True, exist_ok=False)
    label = operation.name + ("/reset_only" if reset_only else "/with_required_reset")
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=args.iterations, repeat=1),
        record_shapes=False,
        profile_memory=False,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(directory)),
    ) as profiler:
        for step in range(args.iterations + 1):
            with torch.autograd.profiler.record_function(f"f1/{label}"):
                output = run()
            # Drain the warmup before active collection and all active work before export.
            if step in (0, args.iterations):
                torch.npu.synchronize()
            profiler.step()
    if not reset_only:
        operation.check(output)
    evidence, records = _read_device_kernels(directory)
    chunks = _split_device_calls(records, args.iterations, label)
    return {"evidence": evidence, "raw_device_kernels": records}, chunks


def _profile_operation(args: argparse.Namespace, operation: _Operation, directory: Path) -> dict[str, Any]:
    reset_chunks = None
    reset_evidence = None
    if operation.reset is not None:
        reset_evidence, reset_chunks = _capture_device_profile(args, operation, directory / "reset_only", True)
    evidence, chunks = _capture_device_profile(args, operation, directory / "operation")
    if reset_chunks is not None:
        # Identify the actual reset records, never subtract an independently measured mean.
        reset_signature = [_kernel_identity(row) for row in reset_chunks[0]]
        count = len(reset_signature)
        if any(
            len(chunk) != count + 1
            or [_kernel_identity(row) for row in chunk[:count]] != reset_signature
            or _kernel_identity(chunk[-1]) in reset_signature
            for chunk in chunks
        ):
            raise RuntimeError(f"{operation.name}: expected reset prefix followed by exactly one verifier kernel")
        evidence["excluded_reset"] = {
            "identity_evidence": reset_evidence,
            "independent_reset_summary": _device_summary(reset_chunks),
            "actual_combined_trace_reset_summary": _device_summary([chunk[:count] for chunk in chunks]),
        }
        chunks = [chunk[count:] for chunk in chunks]
    elif "same_contract_complete" not in operation.name and any(len(chunk) != 1 for chunk in chunks):
        raise RuntimeError(f"{operation.name}: expected exactly one device kernel per native launch")
    return {"status": "COLLECTED", **evidence, **_device_summary(chunks)}


def _collect_device_profiles(
    args: argparse.Namespace,
    pairs: list[tuple[str, _Operation, _Operation]],
    operations: list[_Operation],
    check_inputs: Callable[[], None],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "metric": "NPU profiler kernel_details.csv Duration(us)",
        "sampling_meets_plan_minimum": args.warmup >= 100 and args.groups >= 3 and args.iterations >= 100,
        "scope": "Python callables only; native output work differs, complete adapters share full/optional-masked contract.",
        "interference": "UNREVIEWED: inspect per-group npu-smi snapshots before drawing performance conclusions.",
        "pairs": {},
        "unpaired": {},
    }
    output_path = args.profile_dir / "device-kernel-durations.json"
    if args.batch_size == 0:
        result.update(
            status="N/A",
            reason="B=0 has no candidate device launch; correctness-only case",
            sampling_meets_plan_minimum=False,
        )
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    paired_names = {operation.name for _, left, right in pairs for operation in (left, right)}
    collections = [(name, (left, right)) for name, left, right in pairs]
    collections.extend((operation.name, (operation,)) for operation in operations if operation.name not in paired_names)
    for name, implementations in collections:
        groups = []
        for group in range(args.groups):
            _save_device_snapshot(args.profile_dir, f"device-{name}-group-{group}-before")
            order = implementations if group % 2 == 0 else implementations[::-1]
            measurements = {}
            for operation in order:
                directory = args.profile_dir / "trace" / name / f"group-{group}" / operation.name
                measurements[operation.name] = _profile_operation(args, operation, directory)
                check_inputs()
            _save_device_snapshot(args.profile_dir, f"device-{name}-group-{group}-after")
            groups.append(
                {"group": group, "order": [operation.name for operation in order], "measurements": measurements}
            )
        summaries = {
            operation.name: _summary(
                [
                    value
                    for group in groups
                    for value in group["measurements"][operation.name]["device_duration_sum_us_per_call"]["raw"]
                ]
            )
            for operation in implementations
        }
        collection = {"groups": groups, "device_duration_sum_us_per_call": summaries}
        if len(implementations) == 2:
            left, right = implementations
            collection.update(
                left=left.name,
                right=right.name,
                left_over_right_device_duration_median=summaries[left.name]["median_us"]
                / summaries[right.name]["median_us"],
                equal_output_contract=not name.startswith("native_"),
            )
            result["pairs"][name] = collection
        else:
            result["unpaired"][name] = collection
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        logger.info("Device-kernel durations %s: %s", name, summaries)
    result["status"] = "COLLECTED; interference unreviewed"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _measure_once(operation: _Operation, start: Any, end: Any) -> tuple[dict[str, float], Any]:
    if operation.reset is not None:
        operation.reset()
    # The reset/fill and prior operations complete outside this event interval.
    torch.npu.synchronize()
    start.record()
    begin = time.perf_counter_ns()
    output = operation.run()
    enqueued = time.perf_counter_ns()
    end.record()
    end.synchronize()
    completed = time.perf_counter_ns()
    return {
        "event_us": start.elapsed_time(end) * 1000,
        "host_enqueue_us": (enqueued - begin) / 1000,
        "completed_wall_us": (completed - begin) / 1000,
    }, output


def _save_device_snapshot(directory: Path, label: str) -> None:
    observation = subprocess.run(["npu-smi", "info"], check=True, capture_output=True, text=True)
    (directory / f"{label}-npu-smi.txt").write_text(observation.stdout, encoding="utf-8")


def _collect_timings(
    args: argparse.Namespace,
    pairs: list[tuple[str, _Operation, _Operation]],
    check_inputs: Callable[[], None],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "collection_active": False,
        "sampling_meets_plan_minimum": args.warmup >= 100 and args.groups >= 3 and args.samples >= 100,
        "scope": "Python callable only; C++ AOT and integration conversions are NOT measured.",
        "interference": "UNREVIEWED: inspect per-group npu-smi snapshots; no automatic stable-speedup claim.",
        "pairs": {},
    }
    timing_path = args.profile_dir / "timings.json"
    for name, left, right in pairs:
        groups = []
        for group in range(args.groups):
            _save_device_snapshot(args.profile_dir, f"{name}-group-{group}-before")
            for operation in (left, right):
                if operation.reset is not None:
                    operation.reset()
                operation.check(operation.run())
            check_inputs()
            events = [(torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)) for _ in range(2)]
            for start, end in events:
                start.record()
                end.record()
                end.synchronize()
            samples: dict[str, list[dict[str, float]]] = {left.name: [], right.name: []}
            last_outputs: dict[str, Any] = {}
            for sample in range(args.samples):
                order = (0, 1) if (group + sample) % 2 == 0 else (1, 0)
                for index in order:
                    operation = (left, right)[index]
                    observation, output = _measure_once(operation, *events[index])
                    samples[operation.name].append(observation)
                    last_outputs[operation.name] = output
            for operation in (left, right):
                operation.check(last_outputs[operation.name])
            check_inputs()
            _save_device_snapshot(args.profile_dir, f"{name}-group-{group}-after")
            groups.append(
                {
                    "group": group,
                    "order": "AB/BA alternating per sample; group parity reverses first pair",
                    "samples": samples,
                }
            )
        summary = {}
        for operation in (left, right):
            observations = [sample for group in groups for sample in group["samples"][operation.name]]
            summary[operation.name] = {
                metric: _summary([sample[metric] for sample in observations])
                for metric in ("event_us", "host_enqueue_us", "completed_wall_us")
            }
        left_median = summary[left.name]["event_us"]["median_us"]
        right_median = summary[right.name]["event_us"]["median_us"]
        result["pairs"][name] = {
            "left": left.name,
            "right": right.name,
            "groups": groups,
            "summary": summary,
            "left_over_right_event_median": left_median / right_median if right_median > 0 else None,
        }
        timing_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        logger.info("Paired timings %s: %s", name, summary)

    # Same synchronization frequency for both complete adapters. Kernel-only
    # reset/fill is deliberately excluded here rather than mislabelled as native.
    result["amortized_complete_interval"] = {}
    for name, left, right in pairs:
        if name.startswith("native_"):
            continue
        summaries = {left.name: [], right.name: []}
        for group in range(args.groups):
            _save_device_snapshot(args.profile_dir, f"{name}-amortized-{group}-before")
            order = (left, right) if group % 2 == 0 else (right, left)
            for operation in order:
                operation.check(operation.run())
                check_inputs()
                torch.npu.synchronize()
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                start.record()
                end.record()
                end.synchronize()
                start.record()
                begin = time.perf_counter_ns()
                for _ in range(args.amortized_calls):
                    output = operation.run()
                enqueued = time.perf_counter_ns()
                end.record()
                end.synchronize()
                completed = time.perf_counter_ns()
                operation.check(output)
                summaries[operation.name].append(
                    {
                        "event_us_per_call": start.elapsed_time(end) * 1000 / args.amortized_calls,
                        "host_enqueue_us_per_call": (enqueued - begin) / 1000 / args.amortized_calls,
                        "completed_wall_us_per_call": (completed - begin) / 1000 / args.amortized_calls,
                    }
                )
            check_inputs()
            _save_device_snapshot(args.profile_dir, f"{name}-amortized-{group}-after")
        result["amortized_complete_interval"][name] = {"calls_per_sync": args.amortized_calls, "groups": summaries}
    check_inputs()
    timing_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _main() -> None:
    args = _parse_args()
    args.profile_dir.mkdir(parents=True, exist_ok=False)
    torch.npu.set_device(args.device)
    inputs = _make_inputs(args)
    input_snapshots = [(tensor, tensor.cpu().clone()) for tensor in (inputs.draft, inputs.target, inputs.bonus)]
    expected = _semantic_oracle(inputs, args.mask)
    _check_outputs(_torch_reference(inputs, args.mask), expected)

    kernel_factory = tilelang.jit(pass_configs=GREEDY_PREFIX_VERIFY_PASS_CONFIGS)(build_greedy_prefix_verify_kernel)
    kernel = kernel_factory(
        task_count=args.task_count,
        draft_bits=args.draft_bits,
        target_bits=args.target_bits,
        bonus_bits=args.bonus_bits,
    )
    _check_outputs(_candidate(kernel, inputs, args.mask), expected)
    fixed_outputs = _allocate_outputs(inputs, args.mask)
    fixed_arguments = _kernel_args(inputs, *fixed_outputs)

    def check_full_masked(output: Any) -> None:
        _check_outputs(output, expected)

    def check_inputs() -> None:
        for tensor, original in input_snapshots:
            torch.testing.assert_close(tensor.cpu(), original, rtol=0, atol=0)

    def launch_fixed() -> tuple[torch.Tensor, torch.Tensor | None]:
        if args.batch_size != 0:
            kernel(*fixed_arguments)
        return fixed_outputs

    old_torch = _Operation(
        "torch_same_contract_complete", lambda: _torch_reference(inputs, args.mask), check_full_masked
    )
    candidate = _Operation(
        "tilelang_same_contract_complete", lambda: _candidate(kernel, inputs, args.mask), check_full_masked
    )
    fixed_candidate = _Operation("tilelang_strided_kernel", launch_fixed, check_full_masked)
    operations = [old_torch, candidate, fixed_candidate]
    pairs = [("complete_torch_same_contract", old_torch, candidate)]
    triton_metadata: dict[str, Any] = {"status": "NOT_REQUESTED"}
    native_tilelang_sources: dict[str, str] = {}
    if args.compare_triton:
        if not args.mask or args.batch_size == 0 or args.draft_length == 0:
            triton_metadata = {
                "status": "N/A",
                "reason": "Reference comparisons require mask=true, B>0, K>0; K=0 is not validated.",
            }
        else:
            reference = TritonGreedyReference(
                args.vllm_ascend_root, args.batch_size, args.draft_length, inputs.target.device, args.metadata_policy
            )
            triton_complete = _Operation(
                "triton_same_contract_complete",
                lambda: reference.verify(inputs.draft, inputs.target, inputs.bonus),
                check_full_masked,
            )
            pairs.append(("complete_triton_same_contract", triton_complete, candidate))
            operations.append(triton_complete)
            # Native rows exclude preparation and use the reference's draft/bonus
            # types. Compile a matching TileLang variant for each target dtype.
            native_draft = inputs.draft.to(torch.int32).contiguous()
            native_bonus = inputs.bonus.to(torch.int32).contiguous()
            flat_draft, flat_bonus = native_draft.view(-1), native_bonus.view(-1)
            input_snapshots.extend((tensor, tensor.cpu().clone()) for tensor in (native_draft, native_bonus))
            triton_metadata = {"status": "REQUESTED", **reference.metadata, "compiled": {}}
            target_dtypes = (torch.int32, torch.int64) if args.triton_native_dtype else (torch.int32,)
            for target_dtype in target_dtypes:
                native_target = inputs.target.to(target_dtype).contiguous()
                flat_target = native_target.view(-1)
                input_snapshots.append((flat_target, flat_target.cpu().clone()))
                dense = (native_draft, native_target, native_bonus)
                dense_inputs = _Inputs(*dense, *(_flat_storage_view(tensor) for tensor in dense))
                dense_outputs = _allocate_outputs(dense_inputs, True)
                dense_arguments = _kernel_args(dense_inputs, *dense_outputs)
                dense_kernel = kernel_factory(
                    task_count=args.task_count,
                    draft_bits=32,
                    target_bits=32 if target_dtype == torch.int32 else 64,
                    bonus_bits=32,
                )

                def launch_dense(
                    dense_kernel: Any = dense_kernel,
                    dense_arguments: tuple[Any, ...] = dense_arguments,
                    dense_outputs: tuple[torch.Tensor, torch.Tensor | None] = dense_outputs,
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
                    dense_kernel(*dense_arguments)
                    return dense_outputs

                dense_candidate = _Operation(
                    f"tilelang_dense_target_{str(target_dtype).split('.')[-1]}_two_outputs",
                    launch_dense,
                    check_full_masked,
                )
                operations.append(dense_candidate)
                output = torch.empty_like(dense_outputs[0])
                output.fill_(-1)
                compiled = reference.launch_kernel(output, flat_draft, flat_target, flat_bonus)
                dtype_name = str(target_dtype).split(".")[-1]
                native_tilelang_sources[dtype_name] = dense_kernel.get_kernel_source()
                triton_metadata["compiled"][dtype_name] = {
                    "compiled_hash": compiled.hash,
                    "draft_dtype": "int32",
                    "target_dtype": dtype_name,
                    "bonus_dtype": "int32",
                    "output_dtype": "int32",
                }

                def launch_triton_native(
                    flat_target: torch.Tensor = flat_target, output: torch.Tensor = output
                ) -> torch.Tensor:
                    reference.launch_kernel(output, flat_draft, flat_target, flat_bonus)
                    return output

                def check_native_masked(output: torch.Tensor) -> None:
                    if output.dtype != torch.int32:
                        raise AssertionError("The native Triton output must be INT32")
                    torch.testing.assert_close(output.cpu(), expected[1], rtol=0, atol=0)

                native = _Operation(
                    f"triton_native_target_{dtype_name}_masked_only",
                    launch_triton_native,
                    check_native_masked,
                    lambda output=output: output.fill_(-1),
                )
                pairs.append((f"native_target_{dtype_name}", native, dense_candidate))
                operations.append(native)

    task_comparison_source = None
    baseline_source = None
    if args.kernel_only:
        operations = [fixed_candidate]
        pairs = []
        if args.compare_task_count is not None:
            comparison_kernel = kernel_factory(
                task_count=args.compare_task_count,
                draft_bits=args.draft_bits,
                target_bits=args.target_bits,
                bonus_bits=args.bonus_bits,
            )
            comparison_outputs = _allocate_outputs(inputs, args.mask)
            comparison_arguments = _kernel_args(inputs, *comparison_outputs)

            def launch_comparison() -> tuple[torch.Tensor, torch.Tensor | None]:
                if args.batch_size != 0:
                    comparison_kernel(*comparison_arguments)
                return comparison_outputs

            comparison = _Operation(
                f"tilelang_tasks_{args.compare_task_count}_kernel", launch_comparison, check_full_masked
            )
            operations.append(comparison)
            pairs.append(("task_count_same_contract", fixed_candidate, comparison))
            task_comparison_source = comparison_kernel.get_kernel_source()
        if args.baseline_revision is not None:
            baseline_path = args.profile_dir / "baseline_builder.py"
            baseline_code = subprocess.check_output(
                [
                    "git",
                    "show",
                    f"{args.baseline_revision}:xllm/python/kernels_npu/tilelang/greedy_prefix_verify.py",
                ],
                cwd=Path(__file__).resolve().parents[1],
            )
            baseline_path.write_bytes(baseline_code)
            module_name = f"xllm.python.kernels_npu.tilelang._baseline_{args.baseline_revision}"
            spec = importlib.util.spec_from_file_location(module_name, baseline_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot load recorded baseline builder from {baseline_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            baseline_factory = tilelang.jit(pass_configs=module.GREEDY_PREFIX_VERIFY_PASS_CONFIGS)(
                module.build_greedy_prefix_verify_kernel
            )
            baseline_kernel = baseline_factory(
                task_count=args.task_count,
                draft_bits=args.draft_bits,
                target_bits=args.target_bits,
                bonus_bits=args.bonus_bits,
            )
            baseline_outputs = _allocate_outputs(inputs, args.mask)
            baseline_arguments = _kernel_args(inputs, *baseline_outputs)

            def launch_baseline() -> tuple[torch.Tensor, torch.Tensor | None]:
                if args.batch_size != 0:
                    baseline_kernel(*baseline_arguments)
                return baseline_outputs

            baseline = _Operation("tilelang_baseline_kernel", launch_baseline, check_full_masked)
            operations.append(baseline)
            pairs.append(("revision_same_contract", baseline, fixed_candidate))
            baseline_source = baseline_kernel.get_kernel_source()

    for operation in operations:
        if operation.reset is not None:
            operation.reset()
        operation.check(operation.run())
    for _ in range(args.warmup):
        for operation in operations:
            if operation.reset is not None:
                operation.reset()
            operation.run()
    torch.npu.synchronize()
    check_inputs()

    generated_source = kernel.get_kernel_source()
    source_root = Path(__file__).resolve().parents[1]
    builder_path = source_root / "xllm/python/kernels_npu/tilelang/greedy_prefix_verify.py"
    descriptor_path = source_root / "xllm/compiler/tilelang/targets/ascend/aot/greedy_prefix_verify.py"
    (args.profile_dir / "python-kernel.cpp").write_text(generated_source, encoding="utf-8")
    baseline_metadata = None
    if baseline_source is not None:
        baseline_kernel_path = args.profile_dir / "python-baseline-kernel.cpp"
        baseline_kernel_path.write_text(baseline_source, encoding="utf-8")
        baseline_metadata = {
            "revision": args.baseline_revision,
            "builder_sha256": hashlib.sha256(baseline_code).hexdigest(),
            "generated_path": str(baseline_kernel_path),
            "generated_sha256": hashlib.sha256(baseline_source.encode()).hexdigest(),
            "task_count": args.task_count,
        }
    task_comparison = None
    if task_comparison_source is not None:
        comparison_path = args.profile_dir / "python-task-comparison.cpp"
        comparison_path.write_text(task_comparison_source, encoding="utf-8")
        task_comparison = {
            "task_count": args.compare_task_count,
            "path": str(comparison_path),
            "sha256": hashlib.sha256(task_comparison_source.encode()).hexdigest(),
        }
    native_tilelang_variants = {}
    for dtype_name, source in native_tilelang_sources.items():
        source_path = args.profile_dir / f"python-native-target-{dtype_name}.cpp"
        source_path.write_text(source, encoding="utf-8")
        native_tilelang_variants[dtype_name] = {
            "path": str(source_path),
            "sha256": hashlib.sha256(source.encode()).hexdigest(),
            "task_count": args.task_count,
            "input_bits": {"draft": 32, "target": 32 if dtype_name == "int32" else 64, "bonus": 32},
        }
    metadata = {
        "measurement": "python_callable_paired_npu_device_kernel_durations",
        "primary_metric": "NPU profiler kernel_details.csv Duration(us), in microseconds",
        "device_duration_file": "device-kernel-durations.json",
        "profile_order": "Independent operation/group traces; AB then BA by group parity",
        "device_attribution": "Fixed-shape repeated Name/Type/Core/Device sequence; ambiguous schema/counts fail",
        "native_reset": "Separate reset-only identity trace; exclude matching reset records from actual combined trace",
        "scope": "Not C++ AOT/registry or MTP integration. Candidate input conversion runs in UB; outputs are INT32.",
        "collect_intervals": args.collect_intervals,
        "timing_file": "timings.json" if args.collect_intervals else None,
        "triton_reference": triton_metadata,
        "native_kernel_scope": "Unequal output work: Triton writes only valid masked positions; TileLang writes full+masked.",
        "event_scope": "Event elapsed intervals may include host submission gaps; inspect independent device trace for kernel duration.",
        "interference": "UNREVIEWED: npu-smi group snapshots require operator review; not stable-speedup evidence yet.",
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "tilelang_module": str(Path(tilelang.__file__).resolve()),
        "tilelang_version": tilelang.__version__,
        "device": args.device,
        "device_name": torch.npu.get_device_name(args.device),
        "ascend_home_path": os.environ.get("ASCEND_HOME_PATH"),
        "pass_configs": GREEDY_PREFIX_VERIFY_PASS_CONFIGS,
        "task_count": args.task_count,
        "kernel_only": args.kernel_only,
        "task_comparison": task_comparison,
        "baseline": baseline_metadata,
        "input_bits": {"draft": args.draft_bits, "target": args.target_bits, "bonus": args.bonus_bits},
        "index_dtype": "int32",
        "index_validation": "Nonnegative shapes/strides and relative physical spans fit signed INT32; no narrowing casts.",
        "output_dtype": "torch.int32",
        "target_bonus_shared_storage": args.layout == "mtp" and args.target_bits == args.bonus_bits,
        "complete_reference_cast_scope": "Torch/Triton full-output INT32 conversion is inside the timed interval.",
        "mask": args.mask,
        "layout": args.layout,
        "rejection": args.rejection,
        "id_base": args.id_base,
        "vocab_size": args.vocab_size,
        "input_generation": "Deterministic sequential vocabulary IDs; one mismatch followed by matching IDs.",
        "warmup": args.warmup,
        "groups": args.groups,
        "interval_samples_per_group": args.samples if args.collect_intervals else None,
        "device_calls_per_group": args.iterations,
        "metadata_policy": args.metadata_policy,
        "shape_policy": "One fixed B/K per invocation; no claim about varying-shape metadata updates.",
        "active_steps": args.iterations,
        "profiler_warmup_steps": 1,
        "empty_batch_skips_candidate_launch": args.batch_size == 0,
        "input": {name: _tensor_metadata(getattr(inputs, name)) for name in ("draft", "target", "bonus")},
        "python_kernel_sha256": hashlib.sha256(generated_source.encode()).hexdigest(),
        "native_tilelang_variants": native_tilelang_variants,
        "builder_source": {"path": str(builder_path), "sha256": hashlib.sha256(builder_path.read_bytes()).hexdigest()},
        "aot_descriptor": {
            "path": str(descriptor_path),
            "sha256": hashlib.sha256(descriptor_path.read_bytes()).hexdigest(),
        },
        "aot_source": None,
        "correctness": "PASS",
        "profile": "RUNNING",
        "regions": {
            f"f1/{operation.name}": "CPU labels aid inspection only; durations come exclusively from device CSV records."
            for operation in operations
        },
    }
    if args.aot_source is not None:
        metadata["aot_source"] = {
            "path": str(args.aot_source.resolve()),
            "sha256": hashlib.sha256(args.aot_source.read_bytes()).hexdigest(),
            "note": "A source hash is not evidence of equal execution or performance.",
        }
    metadata_path = args.profile_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    logger.info("Correctness matches the semantic oracle; collecting isolated NPU device-kernel profiles")
    device_profiles = _collect_device_profiles(args, pairs, operations, check_inputs)
    metadata["sampling_meets_plan_minimum"] = device_profiles["sampling_meets_plan_minimum"]
    metadata["profile"] = device_profiles["status"]
    metadata["post_profile_correctness"] = (
        "PASS" if args.batch_size != 0 else "N/A: empty-batch correctness checked before profiling"
    )
    metadata["timing"] = "NOT_REQUESTED"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if args.collect_intervals:
        logger.info("Collecting supplementary event/host/wall intervals; these are not device-kernel durations")
        _collect_timings(args, pairs, check_inputs)
        metadata["timing"] = "COLLECTED; supplementary only; interference unreviewed"
    check_inputs()
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    logger.info("Python F1 device profile status %s; evidence in %s", metadata["profile"], args.profile_dir)


if __name__ == "__main__":
    _main()
