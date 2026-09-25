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

"""NPU collectives issued on the caller's stream.

torch_npu's HCCL process group schedules every collective on the communication
stream it owns and makes the caller's stream wait for the outcome, so a captured
ACLGraph has to carry a cross-stream edge around each collective (the
``CAPTURE_WAIT`` / ``CAPTURE_RECORD`` pair in a trace).  HCCL's own C API takes the
stream as an argument

    HcclAllReduce(sendBuf, recvBuf, count, dataType, op, comm, stream)

so the same collective can be submitted on the stream the surrounding computation
already runs on, with no cross-stream edge at all.  The submission is the
registered ``xllm_ops::npu_all_reduce``, which takes the tensor and the
communicator and issues the collective on the caller's current stream.

Submitting a collective inline requires the communicator to expand on AIV
(``HCCL_OP_EXPANSION_MODE=AIV``): an AICPU-expanded collective cannot run on the
capture stream and fails the capture instead of degrading silently.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  (registers the npu device and its stream API)


def all_reduce_on_current_stream(x: torch.Tensor, group: dist.ProcessGroup) -> None:
    """In-place SUM all-reduce of ``x`` over ``group``, on the caller's stream.

    ``group`` must be an HCCL group on the same device as ``x``.  The payload keeps
    the caller's dtype: this entry point changes which stream the collective is
    submitted on, not how it reduces.
    """
    if x.device.type != "npu":
        raise RuntimeError(f"an NPU all-reduce needs an NPU tensor, got {x.device}")
    torch.ops.xllm_ops.npu_all_reduce(x, _hccl_comm(group, x))


def _hccl_comm(group: dist.ProcessGroup, x: torch.Tensor) -> int:
    """The HCCL communicator this process holds for ``group`` on ``x``'s device.

    The argument selects among the communicators the backend keeps.  The value that
    addresses this process's own communicator is the device index: measured on a
    2-rank group placed on devices 4-7, where the ranks are 0..1 but the device
    indices are 4..7, the device index returned a distinct valid handle and the
    reduction was bit-exact.  Passing a rank instead, as torch_npu's own callers do,
    is only equivalent while rank == device index, which is what one process per
    device gives and a device base or an uneven rank map does not.
    """
    comm = group._get_backend(x.device).get_hccl_comm(x.device.index)
    if not comm:
        raise RuntimeError(f"group {group.group_name} has no HCCL communicator on {x.device}")
    return comm
