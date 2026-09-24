# Greedy prefix verifier device performance

## Goal and measurement

For B={1,3}, K={1,3}, native INT64 draft/target/bonus, MTP strides and full+masked INT32 outputs: device kernel duration median <=1.5 us and P95 <=2.0 us. This is not yet achieved. C++ integration and model E2E are outside this optimization round.

Measurements use NPU profiler `kernel_details.csv` `Duration(us)`, 100 warmups, three alternating-order groups of 100 calls per implementation. Outputs are preallocated, inputs are not converted or copied by the launch adapter. Initialization, JIT, oracle checks and profiler export are outside active collection. Report device duration, not host/event/wall intervals.

Environment: Ascend910_9382 on 910c-27, `shirley-xllm-hc-9.0`, CANN9.0.0, torch_npu2.9.0, installed TileLang0.1.4. Check idle utilization and sufficient memory for profiling; correctness tests require memory but not idle utilization.

## Correct baseline

Kernel commit `973c400f233d4e24ed64c5e4315fb74ad0a8c93a`: 80 non-Triton precision cases passed; eight independent pinned Triton cases passed separately. Native INT64 input is narrowed only in UB. Metadata/indices remain INT32. The masked OR operates on UINT16 aliases of INT32 data and preserves both halves of every output tile.

Earlier tasks48 observations for the original INT64 MTP inputs:

| B/K | Median / P95 (us) |
| --- | --- |
| 3/3 | 4.42 / 5.60 |
| 3/1 | 4.32 / 5.30 |

The native Triton target64 verifier measures about1.34/1.44 us for K3/K1, but writes only masked output. The TileLang verifier writes full+masked. Complete-adapter gains do not establish native-kernel performance success.

## E1: reduce tasks for small batches

**Problem:** task_count48 may impose unnecessary scheduling cost for one or three rows.

**Controlled change:** kernel source remains byte-identical to973c. Compare tasks48 against tasks2 for B1, tasks4 for B3, using profiler commit `29b84408629fd4a4ebcb2360218b03263f2293ea`.

| B/K | tasks48 median / P95 (us) | Small tasks median / P95 (us) |
| --- | --- | --- |
| 1/1 | 4.04 / 4.34 | 2.36 / 2.48 (2) |
| 1/3 | 4.08 / 4.34 | 2.36 / 2.48 (2) |
| 3/1 | 4.08 / 4.70 | 2.54 / 2.68 (4) |
| 3/3 | 4.08 / 5.00 | 2.68 / 2.82 (4) |

**Result:** PASS/exit0, 300 samples per implementation, one device kernel per call, pre/post numerical checks pass. All six group-boundary snapshots per case record selected-device AICore utilization0. These snapshots are not continuous exclusivity guarantees.

**Decision:** retain the small-task direction. The target is still unmet; no claim that the default builder now selects small tasks automatically. This is a measured launch-configuration checkpoint, not a new algorithm.

Reproduce each pair with the existing profiler (prepare the TileLang import environment first):

```bash
python tools/profile_greedy_prefix_verify_npu.py \
  --batch-size 3 --draft-length 3 --layout mtp --mask \
  --kernel-only --task-count 48 --compare-task-count 4 \
  --warmup 100 --groups 3 --iterations 100 \
  --profile-dir <new-evidence-directory>
```

Use B1/K1 and B1/K3 with comparison tasks2, B3/K1 and B3/K3 with tasks4. Exact original generated sources, raw CSVs and snapshot review are in workspace-local `.claude/local/f1-e1-task-count-20260924T111115.708330Z/concise-summary.json` and accompanying artifacts; large local profiling artifacts are not committed.

## E2: vector offset construction (in progress)

Candidate `f0c3732020bd5105174e4def55da4c939fdd90c5` replaces each fixed64-iteration scalar offset loop with INT32 `createvecindex`, scalar min and multiplication, then a public UINT32 reinterpret view. Padding addresses the last valid ID, remains in bounds and is not consumed by prefix/output logic. All other scheduling and synchronization are unchanged.

Precision PASS/exit0: metadata/smoke23 cases, full non-Triton80 cases. Performance comparison against973c is pending, with identical task_count48 and then4 for B3/K3. Do not label E2 retained or faster before the paired result.

## Remaining work

- Reach the native-kernel target; report regressions and rejected experiments.
- Full existing precision regression for retained changes, then final B/K and representative dtype/stride/rejection matrix.
- Preserve arbitrary legal strides and optional-mask semantics; no input copies, host synchronization, fallback or compiler patch.
- INT64 values outside INT32 representability still lack a verified per-value fail-fast check; these precision results do not resolve that separate contract gap.
