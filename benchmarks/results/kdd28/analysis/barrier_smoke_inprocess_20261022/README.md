# Distributed GBT: time within and between barriers

First measurement with the manager-side barrier recorder (2026-10-22). Files here: one
`<tag>.csv` (one row per request round) and the matching driver `<tag>.log` per arm.
Summaries: `python benchmarks/src/distributed_spo_gbt/summarize_barriers.py <csv> [tree_training_s]`.
Reproduce: `benchmarks/src/distributed_spo_gbt/run_barrier_cell.sh SPARSE_OBLIQUE` (see its header).

## Are communication barriers the only barriers? No

Every manager barrier is a "wait for all W replies" loop (`NextAsynchronousProtoAnswer` in
`distributed_gradient_boosted_trees.cc`), but five different things fill the wait:

1. **Manager round-trips** (the communication barriers). Per depth: FindSplits, EvaluateSplits,
   ShareSplits round 1, ShareSplits round 2. Per tree: StartNewIter, EndIter. Depth-6 tree = 22 rounds.
2. **Straggler skew.** Barrier length = slowest worker, so `idle = sum_w (max - runtime_w)` worker-seconds
   are lost even with zero transport cost.
3. **Manager serial sections** between barriers: `SampleInputFeatures`, `MergeBestSplits`,
   `ApplySplitToTree`, `MakeSplitSharingPlan`, per-worker tree serialization in EndIter. All workers idle.
4. **Intra-worker joins** (not visible to the manager): two `BlockingCounter::Wait()` in the worker's
   FindSplits (feature-parallel axis-aligned pool, then per-node oblique pool), the first-touch column
   materialization in `DenseNumericalColumns::Prepare`, and the peer-to-peer `GetSplitValue` waits inside
   ShareSplits (each worker waits on ceil(sqrt(W)) peers).
5. **I/O barriers**: dataset-cache creation, periodic synchronous `CreateCheckpoint`, the synchronous
   validation at the last tree, training-log export.

The recorder measures 1, 2, 3 and the transport part of 1 directly; 4 is inside `worker_max_s`
(a worker-side split of `runtime_seconds` is the next step); 5 appears as its own stage rows.

## What the recorder writes (`DGBT_BARRIER_CSV=<path>`)

`internal::Monitoring` (`learner/distributed_gradient_boosted_trees/`) gets `SendBegin()`,
`DispatchDone(n, bytes)`, `WorkerReply(worker, runtime_s, bytes)`; every Emit* reply loop calls them
and `EndStage`/the next `SendBegin` flushes one row. Manager clock for all timestamps; worker clocks
contribute **durations only** (`WorkerResult.runtime_seconds`, already self-reported by every worker
but previously discarded outside FindSplits). Off by default: no env var, no file; the hooks that stay on cost a few clock reads and
proto size computations per round (below timing noise in the on/off comparison).

| column | meaning |
|---|---|
| `barrier_s` | stage begin to last reply received |
| `manager_serial_s` | stage begin to first request sent |
| `fanout_s` | serialize + enqueue all requests |
| `worker_max/med/min_s` | worker self-reported runtimes |
| `idle_s` | sum_w (max - runtime_w), worker-seconds waiting |
| `comm_residual_s` | barrier - manager_serial - fanout - worker_max |
| `tail_s` | last reply to stage end (merge on the manager) |
| `bytes_out/in` | request / reply proto sizes |

Identity (checked on both runs): sum(barrier + tail) + gaps between stages = tree-training time
(SPO 37.44 + 0.08 = 37.53 s vs harness 37.52 s; AA 31.01 + 0.09 = 31.10 s vs 31.10 s).

## First measurement: 100k x 256 trunk, 4 in-process workers, 50 trees, depth 6

Box: 8 vCPU (4 cores x 2 SMT) Xeon 8259CL, 30 GB; MULTI_THREAD distribute (in-process, no network).
Trunk: `gen_trunk.py --rows 100000 --cols 256 --shards 1 --test_shards 1` (seed 1 train, seed 2 test,
100k test rows). Driver `examples/train_distributed_gbt` defaults except `--num_threads 2`
(default 6; 4 workers x 6 would oversubscribe the 8 vCPUs and fake skew), `--num_trees 50`,
`--split_axis`, oblique density 1.5 / exponent 0.5 (binary weights, driver default). Each cell ran
alone; dataset-cache creation (5.2 s) is outside the tree-training time. Files: `*_20261022T2031*`
(SPO) and `*_20261022T2032*` (AA). An earlier pair of runs (same binary minus the CSV fixes, same
model: identical accuracies) was 11 % faster (33.7 / 27.9 s) with idle 6.1 / 4.2 %, so the idle
share is a 4-7 % band at W = 4 on one box and does not separate the two arms.

| arm | tree training | per tree | test acc | FindSplits share | worker idle at barriers | comm residual | manager serial + fan-out |
|---|---|---|---|---|---|---|---|
| SPO-GBT | 37.5 s | 751 ms | 0.9673 | 96.8 % | 5.4 % of worker time (8.1 worker-s) | 0.3 % | 0.4 % |
| AA-GBT | 31.1 s | 622 ms | 0.9535 | 95.7 % | 6.8 % of worker time (8.3 worker-s) | 0.5 % | 0.5 % |

Per stage, summed over 50 trees (seconds; SPO / AA):

| stage | rounds | barrier | worker_max | idle (worker-s) | comm residual | MB out / in |
|---|---|---|---|---|---|---|
| FindSplits | 250 | 36.25 / 29.77 | 36.16 / 29.68 | 6.61 / 6.84 | 0.027 / 0.044 | 0.64+1.02 / 0.73+0.87 |
| EvaluateSplits | 250 | 0.47 / 0.41 | 0.39 / 0.35 | 0.82 / 0.67 | 0.029 / 0.031 | 1.00+0.02 / 0.86+0.02 |
| ShareSplits r1 | 250 | 0.13 / 0.14 | 0.07 / 0.09 | 0.08 / 0.11 | 0.032 / 0.033 | 0.03+0.02 |
| ShareSplits r2 | 250 | 0.43 / 0.49 | 0.37 / 0.43 | 0.40 / 0.51 | 0.024 / 0.029 | 0.02+0.02 |
| StartNewIter | 50 | 0.10 / 0.13 | 0.09 / 0.11 | 0.07 / 0.10 | 0.005 / 0.011 | 0.01+0.01 |
| EndIter | 50 | 0.07 / 0.07 | 0.05 / 0.05 | 0.09 / 0.10 | 0.014 / 0.008 | ~0 |

FindSplits by layer, SPO-GBT (mean per round, ms; skew = (max - median) / max):

| layer | barrier | worker max | median | min | idle (worker-ms) | skew |
|---|---|---|---|---|---|---|
| 0 | 200.6 | 200.4 | 194.1 | 184.8 | 32.0 | 3.1 % |
| 1 | 127.0 | 126.8 | 123.4 | 115.3 | 22.7 | 2.7 % |
| 2 | 135.8 | 135.4 | 131.0 | 123.2 | 25.3 | 3.3 % |
| 3 | 134.1 | 133.8 | 129.8 | 118.9 | 28.4 | 3.0 % |
| 4 | 127.4 | 126.9 | 123.5 | 115.5 | 23.7 | 2.7 % |

AA-GBT: layer 0 98 ms (root cheaper than deeper layers, the opposite of SPO whose root pays
P = 16 projections over all 100k rows), layers 1-4 123-126 ms, skew 2.5-4.8 %. The slowest worker
rotates (SPO counts by worker 53 / 54 / 72 / 71), so the skew is per-round noise plus column-set
differences, not one slow worker.

Reading: in-process, transport is about 0.1 ms per round (comm residual 0.3-0.5 % of wall) and the
manager's serial sections plus fan-out are 0.4-0.5 %; the barrier cost that remains is **straggler
idle**, 5-7 % of worker time, almost all of it in FindSplits (the only stage with real work).
Everything else (EvaluateSplits, ShareSplits, StartNewIter, EndIter) is under 4 % of the tree time
at W = 4 on one box. Network cost needs the GRPC/Vertex setting; the recorder writes the same CSV
there (manager clock only). Failure path: on an iteration restart the aborted stage's partial row is
flushed and RestoreCheckpoint gets its own row.

## Vertex AI runs (GRPC, one VM per manager and per worker; 2026-10-24/25)

Run list, machine-type rules and what is still missing from the repo:
`benchmarks/src/distributed_spo_gbt/REPLICABILITY.md` section A. `vertex/p_F/` holds the P = F runs
(the paper's); the 2026-10-24 P = ceil(sqrt F) grids (`vertex/p_sqrtF/`) were deleted 2026-10-26, recoverable from commit 54e79671. Each cell
`<RUN_ID>_<dataset>/<arm>_w<W>/` holds `barriers.csv` (same columns as above, manager clock),
`eval.json`, `manager.log` and `summary.txt` (`summarize_barriers.py` output; the 9 `pfw1{base,parstd,vqsort}*`
summaries were regenerated locally on 2026-10-26, the box had no pandas).

What GRPC changed vs. the in-process measurement above (`p_sqrtF/` n2-highmem-4 grids, 50 trees, deleted 2026-10-26, recoverable from commit 54e79671,
P = ceil(sqrt F), read from each cell's `summary.txt`; wall = first stage begin to last stage end):

| cell | wall | comm residual (% of wall) | idle (% of worker time) |
|---|---|---|---|
| SUSY spo_w8 | 144 s | 0.8 | 12.9 |
| SUSY aa_w8 | 206 s | 0.5 | 24.8 |
| HIGGS spo_w4 | 344 s | 0.3 | 17.9 |
| Epsilon spo_w8 | 48 s | 2.5 | 13.3 |
| YouTube-8M spo_w16 | 222 s | 0.5 | 10.4 |
| YouTube-8M aa_w8 | 8276 s | 0.0 | 7.3 |

Transport (comm residual) stays below 1 % of wall except on the shortest cell (Epsilon W=8, 48 s);
manager serial time and fan-out stay below 0.5 %. Straggler idle is the barrier cost that grew:
7-25 % of worker time on the natural datasets vs 5-7 % in-process on the trunk, worst for AA on
tall-narrow data (SUSY W=8). The identity sum(barrier + tail) + gaps = wall holds on every cell.

## Next

1. Split `runtime_seconds` on the worker into Prepare I/O, AA pool, oblique pool, peer wait, apply.
2. Per-job durations in the worker's oblique pool to attribute the skew to node imbalance.
