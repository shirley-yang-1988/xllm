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

"""Real-device probes for the GLM absorbed-Q and latent-V layout contracts.

Run with pytest on an Ascend NPU. The standalone entrypoint measures one isolated
projection including its TND layout adaptation; --profile-dir collects a separate
CANN trace after timing, or the entrypoint can run under msprof --application.
The candidate calls the same TND projection helper used by GLM and DeepSeek.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from functools import cache
from pathlib import Path
from time import perf_counter
from types import ModuleType

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu", reason="MLA projection probes require torch_npu")

# Fixed before collecting results; FP32 references use the quantized inputs.
_TOLERANCES = {
    torch.bfloat16: (1e-2, 2e-2),
    torch.float16: (2e-3, 2e-3),
}
_TOKEN_COUNTS = (1, 2, 3, 4, 8, 16, 32, 48, 64)
_LAYOUTS = ("q_split", "q_offset", "q_contiguous", "v_contiguous", "v_narrow")


def _require_npu() -> None:
    if not torch.npu.is_available():
        raise RuntimeError("MLA projection probes require a real Ascend NPU")
    if not hasattr(torch.ops.npu, "npu_transpose_batchmatmul"):
        raise RuntimeError("npu_transpose_batchmatmul is unavailable in the selected runtime")
    torch.npu.set_device(0)


def _make_inputs(
    tokens: int,
    heads: int,
    layout: str,
    dtype: torch.dtype,
    q_nope_dim: int = 192,
    v_head_dim: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260922 + tokens + heads)
    input_dim, output_dim = (q_nope_dim, 512) if layout.startswith("q_") else (512, v_head_dim)
    if layout in ("q_split", "q_offset"):
        extra_row = int(layout == "q_offset")
        shape = (tokens + extra_row, heads, input_dim + 64)
        backing = torch.randn(shape, generator=generator, dtype=dtype).to("npu")
        x = backing[extra_row:].split((input_dim, 64), dim=-1)[0]
    elif layout == "v_narrow":
        shape = (tokens, heads * 2, input_dim)
        backing = torch.randn(shape, generator=generator, dtype=dtype).to("npu")
        x = backing.narrow(1, heads, heads)
    elif layout in ("q_contiguous", "v_contiguous"):
        backing = torch.randn((tokens, heads, input_dim), generator=generator, dtype=dtype).to("npu")
        x = backing
    else:
        raise ValueError(f"Unknown projection input layout: {layout}")
    weight = torch.randn((heads, input_dim, output_dim), generator=generator)
    weight = (weight / math.sqrt(input_dim)).to(dtype=dtype, device="npu")
    return backing, x, weight


def _legacy_projection(x: torch.Tensor, weight: torch.Tensor, layout: str) -> torch.Tensor:
    if layout.startswith("q_"):
        return torch.bmm(x.transpose(0, 1), weight).transpose(0, 1)
    return torch_npu.npu_transpose_batchmatmul(x.transpose(0, 1), weight, perm_y=(1, 0, 2))


@cache
def _attention_kernels() -> ModuleType:
    import xllm

    _ = xllm.xllm_export
    from xllm.python.kernels_npu import attention

    return attention


def _candidate_projection(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _attention_kernels().batch_matmul_transpose(x, weight)


def _sfa_consumer(tokens: int, heads: int) -> Callable[[torch.Tensor], torch.Tensor]:
    _attention_kernels()
    from xllm.python.kernels_npu.sparse_attention import sparse_flash_attention_out

    generator = torch.Generator().manual_seed(20260922 + tokens)
    dtype = torch.bfloat16
    cache = torch.randn((tokens, 16, 1, 512), generator=generator, dtype=dtype).to("npu")
    rope_cache = torch.randn((tokens, 16, 1, 64), generator=generator, dtype=dtype).to("npu")
    q_pe = torch.randn((tokens, heads, 64), generator=generator, dtype=dtype).to("npu")
    topk = torch.arange(4, dtype=torch.int32).view(1, 1, 4).expand(tokens, 1, 4).contiguous().to("npu")
    block_table = torch.arange(tokens, dtype=torch.int32, device="npu").view(tokens, 1)
    actual_q = torch.arange(1, tokens + 1, dtype=torch.int32, device="npu")
    actual_kv = torch.full((tokens,), 16, dtype=torch.int32, device="npu")
    output = torch.empty((tokens, heads, 512), dtype=dtype, device="npu")

    def attend(query: torch.Tensor) -> torch.Tensor:
        sparse_flash_attention_out(
            query,
            cache,
            cache,
            topk,
            block_table,
            actual_q,
            actual_kv,
            q_pe,
            rope_cache,
            1.0 / 16.0,
            1,
            "TND",
            "PA_BSND",
            3,
            output,
        )
        return output

    return attend


def _fp32_reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.einsum("thd,hdo->tho", x.cpu().float(), weight.cpu().float())


def _tensor_metadata(x: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(x.shape),
        "stride": list(x.stride()),
        "storage_offset": x.storage_offset(),
        "dtype": str(x.dtype),
        "format": torch_npu.get_npu_format(x),
        "contiguous": x.is_contiguous(),
    }


def _error(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    difference = (actual.cpu().float() - reference).abs()
    return {
        "max_absolute": difference.max().item(),
        "max_relative": (difference / reference.abs().clamp_min(1e-12)).max().item(),
    }


def _check_output(actual: torch.Tensor, reference: torch.Tensor, dtype: torch.dtype) -> None:
    assert actual.dtype == dtype
    assert actual.device.type in ("npu", "privateuseone")
    assert actual.is_contiguous(), f"Expected contiguous TND output, got {actual.stride()}"
    assert tuple(actual.shape) == tuple(reference.shape)
    rtol, atol = _TOLERANCES[dtype]
    torch.testing.assert_close(actual.cpu().float(), reference, rtol=rtol, atol=atol)


@pytest.mark.parametrize("tokens", _TOKEN_COUNTS)
@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_projection_eager(
    tokens: int,
    layout: str,
    dtype: torch.dtype,
    record_property: Callable[[str, object], None],
) -> None:
    _require_npu()
    backing, x, weight = _make_inputs(tokens, 4, layout, dtype)
    original_backing, original_weight = backing.cpu(), weight.cpu()
    reference = _fp32_reference(x, weight)
    legacy = _legacy_projection(x, weight, layout)
    actual = _candidate_projection(x, weight)
    torch.npu.synchronize()
    _check_output(actual, reference, dtype)
    rtol, atol = _TOLERANCES[dtype]
    torch.testing.assert_close(actual, legacy, rtol=rtol, atol=atol)
    torch.testing.assert_close(backing.cpu(), original_backing, rtol=0, atol=0)
    torch.testing.assert_close(weight.cpu(), original_weight, rtol=0, atol=0)
    record_property("operator_schema", str(torch.ops.npu.npu_transpose_batchmatmul.default._schema))
    record_property(
        "projection",
        json.dumps(
            {
                "input": _tensor_metadata(x),
                "weight": _tensor_metadata(weight),
                "output": _tensor_metadata(actual),
                "candidate_error": _error(actual, reference),
                "legacy_error": _error(legacy, reference),
            }
        ),
    )


@pytest.mark.parametrize("heads", (1, 2, 8, 16))
@pytest.mark.parametrize("layout", ("q_split", "v_narrow"))
def test_projection_other_head_counts(heads: int, layout: str) -> None:
    _require_npu()
    _, x, weight = _make_inputs(4, heads, layout, torch.bfloat16)
    _check_output(_candidate_projection(x, weight), _fp32_reference(x, weight), x.dtype)


def _capture(
    operation: Callable[[], torch.Tensor],
    stream: torch.npu.Stream | None = None,
) -> tuple[torch.npu.NPUGraph, torch.Tensor]:
    if stream is None:
        stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            operation()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        output = operation()
    return graph, output


@pytest.mark.parametrize("tokens", _TOKEN_COUNTS)
@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_projection_acl_graph_replays_updated_storage(tokens: int, layout: str, dtype: torch.dtype) -> None:
    _require_npu()
    backing, x, weight = _make_inputs(tokens, 4, layout, dtype)
    original_backing, original_weight = backing.cpu(), weight.cpu()
    legacy_graph, legacy = _capture(lambda: _legacy_projection(x, weight, layout))
    graph, actual = _capture(lambda: _candidate_projection(x, weight))
    addresses = (x.data_ptr(), weight.data_ptr(), actual.data_ptr())
    previous = None
    for scale in (1.0, -0.5, 1.5):
        backing.copy_(original_backing * scale)
        weight.copy_(original_weight * (2.0 - scale))
        torch.npu.synchronize()
        legacy_graph.replay()
        graph.replay()
        torch.npu.synchronize()
        _check_output(actual, _fp32_reference(x, weight), x.dtype)
        rtol, atol = _TOLERANCES[dtype]
        torch.testing.assert_close(actual, legacy, rtol=rtol, atol=atol)
        torch.testing.assert_close(backing.cpu(), original_backing * scale, rtol=0, atol=0)
        torch.testing.assert_close(weight.cpu(), original_weight * (2.0 - scale), rtol=0, atol=0)
        assert addresses == (x.data_ptr(), weight.data_ptr(), actual.data_ptr())
        current = actual.cpu()
        if previous is not None:
            assert not torch.equal(current, previous), "Graph replay reused stale input values"
        previous = current


@pytest.mark.parametrize("tokens", (1, 2, 4, 8))
@pytest.mark.parametrize("layout", ("q_split", "q_offset", "v_contiguous", "v_narrow"))
def test_projection_real_sfa_acl_graph(tokens: int, layout: str) -> None:
    _require_npu()
    backing, x, weight = _make_inputs(tokens, 4, layout, torch.bfloat16)
    original_backing, original_weight = backing.cpu(), weight.cpu()
    if layout.startswith("q_"):
        legacy_sfa, candidate_sfa = _sfa_consumer(tokens, 4), _sfa_consumer(tokens, 4)
        legacy_graph, legacy = _capture(lambda: legacy_sfa(_legacy_projection(x, weight, layout)))
        graph, actual = _capture(lambda: candidate_sfa(_candidate_projection(x, weight)))
    else:
        heads = 8 if layout == "v_narrow" else 4
        _, query, _ = _make_inputs(tokens, heads, "v_contiguous", torch.bfloat16)
        sfa = _sfa_consumer(tokens, heads)
        sfa_graph, sfa_output = _capture(lambda: sfa(query))
        x = sfa_output.narrow(1, 4, 4) if layout == "v_narrow" else sfa_output
        legacy_graph, legacy = _capture(lambda: _legacy_projection(x, weight, layout))
        graph, actual = _capture(lambda: _candidate_projection(x, weight))
    addresses = (x.data_ptr(), weight.data_ptr(), actual.data_ptr())
    previous = None
    for scale in (1.0, -0.5, 1.5):
        backing.copy_(original_backing * scale)
        weight.copy_(original_weight * (2.0 - scale))
        if layout.startswith("v_"):
            query.mul_(-0.5)
            torch.npu.synchronize()
            sfa_graph.replay()
        torch.npu.synchronize()
        original_x = x.cpu()
        legacy_graph.replay()
        graph.replay()
        torch.npu.synchronize()
        rtol, atol = _TOLERANCES[x.dtype]
        torch.testing.assert_close(actual, legacy, rtol=rtol, atol=atol)
        assert torch.isfinite(actual).all()
        assert actual.is_contiguous()
        if layout.startswith("v_"):
            _check_output(actual, _fp32_reference(x, weight), x.dtype)
        torch.testing.assert_close(x.cpu(), original_x, rtol=0, atol=0)
        torch.testing.assert_close(backing.cpu(), original_backing * scale, rtol=0, atol=0)
        torch.testing.assert_close(weight.cpu(), original_weight * (2.0 - scale), rtol=0, atol=0)
        assert addresses == (x.data_ptr(), weight.data_ptr(), actual.data_ptr())
        current = actual.cpu()
        if previous is not None:
            assert not torch.equal(current, previous), "SFA graph replay reused stale values"
        previous = current


@pytest.mark.parametrize("tokens", (1, 2, 4, 8))
@pytest.mark.parametrize("layout", ("q_offset", "v_contiguous", "v_narrow"))
def test_projection_deepseek_contract(tokens: int, layout: str) -> None:
    _require_npu()
    backing, x, weight = _make_inputs(tokens, 8, layout, torch.bfloat16, q_nope_dim=128, v_head_dim=128)
    original_backing, original_weight = backing.cpu(), weight.cpu()
    legacy_graph, legacy = _capture(lambda: _legacy_projection(x, weight, layout))
    graph, actual = _capture(lambda: _candidate_projection(x, weight))
    for scale in (1.0, -0.5, 1.5):
        backing.copy_(original_backing * scale)
        weight.copy_(original_weight * (2.0 - scale))
        torch.npu.synchronize()
        legacy_graph.replay()
        graph.replay()
        torch.npu.synchronize()
        _check_output(actual, _fp32_reference(x, weight), x.dtype)
        rtol, atol = _TOLERANCES[x.dtype]
        torch.testing.assert_close(actual, legacy, rtol=rtol, atol=atol)
        torch.testing.assert_close(backing.cpu(), original_backing * scale, rtol=0, atol=0)
        torch.testing.assert_close(weight.cpu(), original_weight * (2.0 - scale), rtol=0, atol=0)


@pytest.mark.parametrize("tokens", (128, 1024, 4096))
@pytest.mark.parametrize("layout", ("q_split", "q_offset", "v_contiguous", "v_narrow"))
def test_projection_prefill_and_parallel_layouts(tokens: int, layout: str) -> None:
    _require_npu()
    backing, x, weight = _make_inputs(tokens, 4, layout, torch.bfloat16)
    original_backing, original_weight = backing.cpu(), weight.cpu()
    legacy = _legacy_projection(x, weight, layout)
    actual = _candidate_projection(x, weight)
    _check_output(actual, _fp32_reference(x, weight), x.dtype)
    rtol, atol = _TOLERANCES[x.dtype]
    torch.testing.assert_close(actual, legacy, rtol=rtol, atol=atol)
    torch.testing.assert_close(backing.cpu(), original_backing, rtol=0, atol=0)
    torch.testing.assert_close(weight.cpu(), original_weight, rtol=0, atol=0)


def test_projection_acl_graph_bucket_switching() -> None:
    _require_npu()
    entries = []
    for tokens in (1, 2, 4, 8):
        backing, x, weight = _make_inputs(tokens, 4, "q_split", torch.bfloat16)
        graph, output = _capture(lambda x=x, weight=weight: _candidate_projection(x, weight))
        entries.append((backing, x, weight, graph, output, output.data_ptr()))
    for index in (0, 3, 1, 2, 0, 2, 3, 1):
        backing, x, weight, graph, output, address = entries[index]
        backing.mul_(-0.5)
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        _check_output(output, _fp32_reference(x, weight), x.dtype)
        assert output.data_ptr() == address


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=_LAYOUTS, required=True)
    parser.add_argument("--implementation", choices=("legacy", "candidate", "both"), required=True)
    parser.add_argument("--mode", choices=("eager", "graph"), default="graph")
    parser.add_argument("--boundary", choices=("tnd", "sfa"), default="tnd")
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--graph-repeat", type=int, default=1)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()
    if min(args.tokens, args.iterations, args.graph_repeat, args.samples) <= 0:
        parser.error("tokens, iterations, graph-repeat and samples must be positive")
    if args.mode != "graph" and args.graph_repeat != 1:
        parser.error("graph-repeat is only valid in graph mode")
    if args.output.exists():
        parser.error("output already exists; use a new evidence path")
    if args.profile_dir is not None and args.profile_dir.exists():
        parser.error("profile directory already exists; use a new evidence path")
    _require_npu()
    backing, x, weight = _make_inputs(args.tokens, 4, args.layout, torch.bfloat16)
    if args.boundary == "sfa" and args.layout.startswith("v_"):
        heads = 8 if args.layout == "v_narrow" else 4
        _, query, _ = _make_inputs(args.tokens, heads, "v_contiguous", torch.bfloat16)
        produce_values = _sfa_consumer(args.tokens, heads)
        backing = produce_values(query)
        x = backing.narrow(1, 4, 4) if args.layout == "v_narrow" else backing
    original_backing, original_weight = backing.cpu(), weight.cpu()
    reference = _fp32_reference(x, weight)
    if args.boundary == "sfa" and args.layout.startswith("q_"):
        reference_sfa = _sfa_consumer(args.tokens, 4)
        reference = reference_sfa(_legacy_projection(x, weight, args.layout)).cpu().float()
    implementations = ("legacy", "candidate") if args.implementation == "both" else (args.implementation,)
    operations = {}
    outputs = {}
    consumers = {}
    results = {}
    graphs = []
    capture_stream = torch.npu.Stream() if args.mode == "graph" else None
    measurement_stream = capture_stream if capture_stream is not None else torch.npu.current_stream()
    if "candidate" in implementations:
        candidate_output = _candidate_projection(x, weight)
        assert candidate_output.is_contiguous(), "Candidate must produce TND without an extra copy"
    for implementation in implementations:
        consume = _sfa_consumer(args.tokens, 4) if args.boundary == "sfa" and args.layout.startswith("q_") else None
        # Keep capture-external cache and metadata tensors alive for every graph.
        consumers[implementation] = consume

        def project(
            implementation: str = implementation,
            consume: Callable[[torch.Tensor], torch.Tensor] | None = consume,
        ) -> torch.Tensor:
            for _ in range(args.graph_repeat):
                if implementation == "legacy":
                    result = _legacy_projection(x, weight, args.layout)
                else:
                    result = _candidate_projection(x, weight)
                # The SFA boundary consumes Q directly; the isolated TND
                # boundary measures projection plus any required data copy.
                output = consume(result) if consume is not None else result.contiguous()
            return output

        if args.mode == "graph":
            graph, output = _capture(project, stream=capture_stream)
            graphs.append(graph)
            graph.replay()
            operation = graph.replay
        else:
            output = project()
            operation = project
        torch.npu.synchronize()
        _check_output(output, reference, x.dtype)
        for _ in range(20):
            operation()
        torch.npu.synchronize()
        operations[implementation] = operation
        outputs[implementation] = output
        results[implementation] = {
            "output": _tensor_metadata(output),
            "error": _error(output, reference),
            "synchronized_wall_us_per_projection": [],
            "device_event_us_per_projection": [],
        }

    sample_order = []
    projections_per_sample = args.iterations * args.graph_repeat
    for sample_index in range(args.samples):
        order = implementations if sample_index % 2 == 0 else implementations[::-1]
        sample_order.append(list(order))
        for implementation in order:
            operation = operations[implementation]
            start_event = torch.npu.Event(enable_timing=True)
            end_event = torch.npu.Event(enable_timing=True)
            torch.npu.synchronize()
            start = perf_counter()
            with torch.npu.stream(measurement_stream):
                start_event.record()
                for _ in range(args.iterations):
                    operation()
                end_event.record()
            torch.npu.synchronize()
            wall_us = (perf_counter() - start) * 1e6 / projections_per_sample
            device_us = start_event.elapsed_time(end_event) * 1e3 / projections_per_sample
            results[implementation]["synchronized_wall_us_per_projection"].append(wall_us)
            results[implementation]["device_event_us_per_projection"].append(device_us)

    for implementation in implementations:
        output = outputs[implementation] if args.mode == "graph" else operations[implementation]()
        _check_output(output, reference, x.dtype)
    if len(implementations) == 2:
        rtol, atol = _TOLERANCES[x.dtype]
        torch.testing.assert_close(outputs["candidate"], outputs["legacy"], rtol=rtol, atol=atol)
    torch.testing.assert_close(backing.cpu(), original_backing, rtol=0, atol=0)
    torch.testing.assert_close(weight.cpu(), original_weight, rtol=0, atol=0)
    evidence = {
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "schema": str(torch.ops.npu.npu_transpose_batchmatmul.default._schema),
        "layout": args.layout,
        "implementation": args.implementation,
        "mode": args.mode,
        "boundary": args.boundary,
        "includes_sfa_execution": args.boundary == "sfa" and args.layout.startswith("q_"),
        "iterations_per_sample": args.iterations,
        "projections_per_graph": args.graph_repeat,
        "sample_order": sample_order,
        "events_and_replay_on_capture_stream": args.mode == "graph",
        "input": _tensor_metadata(x),
        "weight": _tensor_metadata(weight),
        "results": results,
    }
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    if args.profile_dir is not None:
        args.profile_dir.mkdir(parents=True, exist_ok=False)
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=True,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.profile_dir)),
        ):
            for implementation in implementations:
                with torch.autograd.profiler.record_function(
                    f"glm_mla/{args.layout}/{implementation}/{args.mode}/T{args.tokens}"
                ):
                    with torch.npu.stream(measurement_stream):
                        for _ in range(args.iterations):
                            operations[implementation]()
                    torch.npu.synchronize()


if __name__ == "__main__":
    _main()
