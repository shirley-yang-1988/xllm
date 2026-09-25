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

"""The NPU side of ``xllm.python.distributed``.

The collectives in the parent package are hardware-neutral and reach this package
only where the NPU needs something the hardware-neutral path cannot express, so
only an NPU process imports it.
"""

from __future__ import annotations

from xllm.python.distributed.npu.hccl import all_reduce_on_current_stream

__all__ = [
    "all_reduce_on_current_stream",
]
