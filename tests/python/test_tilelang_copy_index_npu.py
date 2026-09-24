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

"""Lower-only Ascend copy regression for the INT32 indexing contract.

Run in the selected NPU container from the exact Git commit pushed for testing.
No F1/utils imports, torch_npu initialization, JIT compilation or device launch.
The earlier INT32/INT64 comparison is retained in the frozen probe evidence;
INT64 indexing is no longer part of the F1 contract. Successful lowering is not
device correctness or performance evidence. Count=8 does not cover dynamic
copy extents.
"""

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from scripts.logger import logger

_PASS_CONFIGS = {
    "tl.ascend_auto_sync": False,
    "tl.ascend_memory_planning": True,
    "tl.ascend_auto_cross_core_sync": False,
    "tl.ascend_auto_cv_combine": False,
}


@pytest.fixture(scope="module")
def tilelang_modules() -> tuple[ModuleType, ModuleType]:
    return importlib.import_module("tilelang"), importlib.import_module("tilelang.language")


def _build_copy_probe(language: ModuleType) -> Any:
    T = language

    @T.prim_func
    def copy_probe(
        source: T.Tensor((16,), "int32"),
        destination: T.Tensor((16,), "int32"),
        offset: T.int32,
    ) -> None:
        with T.Kernel(1, is_npu=True), T.Scope("V"):
            temporary = T.alloc_ub((8,), "int32")
            count = T.int32(8)
            T.copy(source[offset : offset + count], temporary[0:count])
            T.set_flag("mte2", "mte3", 0)
            T.wait_flag("mte2", "mte3", 0)
            T.copy(temporary[0:count], destination[offset : offset + count])

    return copy_probe


def test_copy_int32_metadata_lowering(tilelang_modules: tuple[ModuleType, ModuleType], tmp_path: Path) -> None:
    tilelang, language = tilelang_modules
    primitive = _build_copy_probe(language)
    assert str(primitive.params[2].dtype) == "int32"
    assert all(str(buffer.dtype) == "int32" for buffer in primitive.buffer_map.values())
    primitive_path = tmp_path / "copy_index32.tir.py"
    primitive_path.write_text(primitive.script(), encoding="utf-8")
    logger.info(
        f"Ascend copy lowering: metadata=int32, tokens=int32, "
        f"count=8, runtime offset; TileLang={tilelang.__file__}; TIR={primitive_path}"
    )
    with tilelang.tvm.transform.PassContext(opt_level=3, config=_PASS_CONFIGS):
        lowered = tilelang.engine.lower(primitive)
    assert lowered.kernel_source, "Copy probe lowering returned no kernel source"
    source_path = tmp_path / "copy_index32.cpp"
    source_path.write_text(lowered.kernel_source, encoding="utf-8")
    logger.info(f"Ascend copy lowering completed: metadata=int32; source={source_path}")
