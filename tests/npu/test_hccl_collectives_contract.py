# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""Real registered HCCL contracts, isolated from tests/python runtime stubs.

Set XLLM_TEST_NATIVE_LIBRARY and XLLM_TEST_NPU_DEVICE inside the NPU container.
The caller checks free HBM and sets both HCCL socket port ranges. Each failing
case has its own process; buffer/count checks use a warmed single-rank HCCL
communicator, never fake pointers or peers that could be stranded by abort.
Multi-rank numerical/capture behavior is covered by probe_hccl_collectives.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest

OPERATIONS = ("all_reduce", "all_gather", "reduce_scatter")
INPUT_ERRORS = {
    "noncontiguous_input": "HCCL requires a contiguous tensor.",
    "empty_input": "HCCL requires a nonempty tensor.",
    "nz_input": "HCCL requires ND storage format.",
    "null_comm": "HCCL communicator must not be null.",
}
OUTPUT_ERRORS = {
    "noncontiguous_output": "HCCL requires a contiguous tensor.",
    "empty_output": "HCCL requires a nonempty tensor.",
    "nz_output": "HCCL requires ND storage format.",
    "dtype_mismatch": "HCCL buffer dtype mismatch.",
    "overlap_alias": "AllGather/ReduceScatter buffers must not overlap.",
    "overlap_partial": "AllGather/ReduceScatter buffers must not overlap.",
}


@pytest.fixture(scope="module")
def native_environment() -> dict[str, str]:
    library = os.environ.get("XLLM_TEST_NATIVE_LIBRARY")
    device = os.environ.get("XLLM_TEST_NPU_DEVICE")
    if not library or device is None:
        pytest.skip("set XLLM_TEST_NATIVE_LIBRARY and XLLM_TEST_NPU_DEVICE for real HCCL tests")
    assert Path(library).is_absolute() and Path(library).is_file(), "native library must be an existing absolute path"
    assert device.isdecimal(), "XLLM_TEST_NPU_DEVICE must be a nonnegative logical device ID"
    for name in ("HCCL_HOST_SOCKET_PORT_RANGE", "HCCL_NPU_SOCKET_PORT_RANGE"):
        assert os.environ.get(name), f"set {name} explicitly for this test job"
    return {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}


def _run_case(
    tmp_path: Path,
    environment: dict[str, str],
    operation: str,
    case: str,
    expected_error: str | None = None,
) -> None:
    script = Path(__file__).resolve()
    command = [
        sys.executable,
        str(script),
        "--operation",
        operation,
        "--case",
        case,
        "--rendezvous",
        str(tmp_path / "rendezvous"),
    ]
    log_path = tmp_path / "worker.log"
    with log_path.open("xb") as log:
        completed = subprocess.run(
            command,
            cwd=script.parents[2],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=120,
            check=False,
        )
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "command": command,
                "returncode": completed.returncode,
                "worker_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
                "library": environment["XLLM_TEST_NATIVE_LIBRARY"],
                "device": environment["XLLM_TEST_NPU_DEVICE"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if expected_error is None:
        assert completed.returncode == 0, f"worker failed ({completed.returncode}); inspect {log_path}"
    else:
        assert completed.returncode == -signal.SIGABRT, f"expected CHECK abort, got {completed.returncode}; {log_path}"
        assert expected_error in log_path.read_text(encoding="utf-8", errors="replace"), (
            f"missing diagnostic; {log_path}"
        )


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("case,diagnostic", INPUT_ERRORS.items())
def test_rejects_invalid_input_or_null_comm(
    tmp_path: Path, native_environment: dict[str, str], operation: str, case: str, diagnostic: str
) -> None:
    _run_case(tmp_path, native_environment, operation, case, diagnostic)


@pytest.mark.parametrize("operation", ("all_gather", "reduce_scatter"))
@pytest.mark.parametrize("case,diagnostic", OUTPUT_ERRORS.items())
def test_rejects_invalid_output_buffers(
    tmp_path: Path, native_environment: dict[str, str], operation: str, case: str, diagnostic: str
) -> None:
    _run_case(tmp_path, native_environment, operation, case, diagnostic)


@pytest.mark.parametrize(
    "operation,diagnostic",
    (
        ("all_gather", "AllGather expects one input-sized output block per rank."),
        ("reduce_scatter", "ReduceScatter expects one output-sized input block per rank."),
    ),
)
def test_rejects_count_mismatch(
    tmp_path: Path, native_environment: dict[str, str], operation: str, diagnostic: str
) -> None:
    _run_case(tmp_path, native_environment, operation, "count_mismatch", diagnostic)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_fake_collectives_return_none_without_real_communicator(
    tmp_path: Path, native_environment: dict[str, str], operation: str
) -> None:
    _run_case(tmp_path, native_environment, operation, "fake")


def _worker(operation: str, case: str, rendezvous: Path) -> None:
    import torch
    import torch.distributed as dist
    import torch_npu

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.logger import logger

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    torch.set_num_threads(1)
    device = torch.device(f"npu:{os.environ['XLLM_TEST_NPU_DEVICE']}")
    torch.npu.set_device(device)
    torch.ops.load_library(os.environ["XLLM_TEST_NATIVE_LIBRARY"])
    op = getattr(torch.ops.xllm_ops, f"npu_{operation}")

    if case == "fake":
        from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

        from xllm.python.kernels_npu import _custom_op  # noqa: F401

        with FakeTensorMode():
            source = torch.empty((16, 16), dtype=torch.float32, device=device)
            output = torch.empty_like(source)
            assert isinstance(source, FakeTensor) and isinstance(output, FakeTensor)
            result = op(source, 0) if operation == "all_reduce" else op(source, output, 0)
            assert result is None
        logger.info("FakeTensor %s returned None without a real communicator", operation)
        return

    source = torch.arange(256, dtype=torch.float32).reshape(16, 16).to(device)
    output = torch.empty_like(source)
    comm = 0
    if case not in INPUT_ERRORS:
        dist.init_process_group(
            backend="hccl",
            init_method=rendezvous.as_uri(),
            rank=0,
            world_size=1,
            timeout=timedelta(seconds=30),
        )
        group = dist.new_group(ranks=[0], backend="hccl", timeout=timedelta(seconds=30))
        warm = torch.ones(1, dtype=torch.float32, device=device)
        dist.all_reduce(warm, group=group)
        torch.npu.synchronize()
        comm = group._get_backend(device).get_hccl_comm(device.index)
        assert comm, "single-rank HCCL communicator was not initialized"

    if case == "noncontiguous_input":
        source = source.t()
        assert not source.is_contiguous()
    elif case == "empty_input":
        source = torch.empty(0, dtype=source.dtype, device=device)
    elif case == "noncontiguous_output":
        output = output.t()
        assert not output.is_contiguous()
    elif case == "empty_output":
        output = torch.empty(0, dtype=source.dtype, device=device)
    elif case == "dtype_mismatch":
        output = output.to(torch.bfloat16)
    elif case == "count_mismatch":
        output = torch.empty(128, dtype=source.dtype, device=device)
    elif case == "overlap_alias":
        output = source
    elif case == "overlap_partial":
        storage = torch.empty(257, dtype=source.dtype, device=device)
        source, output = storage[:256].view(16, 16), storage[1:].view(16, 16)
    elif case in ("nz_input", "nz_output"):
        torch.npu.config.allow_internal_format = True
        private = torch_npu.npu_format_cast(source, 29)
        assert torch_npu.get_npu_format(private) == 29, "FRACTAL_NZ test precondition was not established"
        assert private.is_contiguous(), "NZ case must reach the storage-format guard, not the stride guard"
        if case == "nz_input":
            source = private
        else:
            output = private
    elif case != "null_comm":
        raise ValueError(f"unknown contract case {case}")
    torch.npu.synchronize()
    logger.info("Invoking %s with invalid %s", operation, case)
    if operation == "all_reduce":
        op(source, comm)
    else:
        op(source, output, comm)
    raise AssertionError(f"{operation} accepted invalid {case}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=OPERATIONS, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--rendezvous", type=Path, required=True)
    args = parser.parse_args()
    _worker(args.operation, args.case, args.rendezvous)
