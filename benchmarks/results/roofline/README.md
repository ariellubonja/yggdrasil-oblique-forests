# Roofline analysis — Intel Advisor 2025.2 (`--collect=roofline`)

Machine: AWS m7i.metal-24xl, Xeon Platinum 8488C, 48 cores / 48 threads, 377 GB, kernel 7.0.0-1010-aws
(Advisor's Pin instrumentation works; perf does not, `perf_event_paranoid=4`).
Harness: `examples/train_oblique_forest.cc`, **48 trees on 48 threads** (one tree per thread), `--seed=1`,
full-depth RF, bagging.

Arms (icx 2025.2.1, `-O3 -march=native`, plus `-g` so loops resolve to source; same codegen as the benchmark builds):

| key | branch / commit | flags |
|---|---|---|
| `dyn` | `rebased-main` @ 661a903c | `--numerical_split_type 'Dynamic Random Histogram' --histogram_num_bins=64 --dynamic_split_threshold=250` |
| `hwy_exact` | same | `--numerical_split_type Exact` (Highway VQSort exact finder, every node) |
| `may2025_exact` | `ydf-may-2025` @ f673e43c (upstream 2d1454cc + harness port), bazel 6.5.0, **plus an uncommitted `__attribute__((noinline))` on `ProjectionEvaluator::Evaluate`** (oblique.h) so ApplyProjection is a separate symbol, as in the fork; ~1 % cost (training block 16.6 s vs 16.8 s on trunk 100k) | `--numerical_split_type Exact` (std::sort exact finder) |

Shapes: `trunk100k` = trunk 100k×4096; `higgs` = HIGGS_with_header.csv 11M×28; `epsilon` = 400k×2000;
`trunk1500k` = trunk 1.5M×4096.

## Collection

`benchmarks/profiling/roofline/run_roofline.sh <shape> [arms]` = `advisor --collect=roofline --interval=<ms>` and no
cache simulation (`CACHESIM=0`, file suffix `_nocs`; L1-level intensity, the classic CARM roofline), then the CSV
exports the analysis needs (`survey`, `survey --show-functions`, `top-down`, `roofs`) and Advisor's own HTML rooflines.
Instrumented slowdown of the training block: 7–12× on the fork binaries, 4.5× (trunk) to 6× (HIGGS) on May-2025.

- **No `--stacks`.** Not needed for a roofline; it only adds inclusive FLOP columns. The results from the first pass
  (trunk100k, HIGGS, Epsilon, fork trunk1500k) were collected with it — harmless for the numbers.
- **Survey sampling interval.** 20 ms everywhere except `may2025_exact trunk1500k` (100 ms). Advisor's survey always
  samples call stacks, and the May-2025 tree grower is recursive (`NodeTrain` → children), so its stack file grows
  to 1.3–1.6 GB at HIGGS / trunk 1.5M with 20 ms sampling and the single-threaded finalization then takes 1.5–2.4 h
  (plus ~30 min per report export). 100 ms cuts that ~5× (392 MB, ~10 min). Cost: per-loop *elapsed* time (the
  slowest thread's time in the loop, which Advisor uses as the roofline denominator) is a max-over-threads estimate
  and is biased up by sampling noise: on trunk100k the same loops read 20–60 % longer at 100 ms than at 20 ms, so
  the May-2025 trunk1500k GFLOP/s are ≈10–20 % low; summed CPU time (the shares) is unaffected (±3 %).
- **Cache simulation** (`CACHESIM=1`, no suffix): 75–140× slowdown; the only way to get an L2/L3/DRAM-level
  intensity. Done once, on trunk100k, all three arms (the May-2025 run predates the noinline patch).
- Never run `advisor --report` on a project whose collection is still running: it destroys the FLOP-data merge
  (that is how `may2025_exact epsilon` lost its FLOP data; not re-run).

## Analysis (`benchmarks/profiling/roofline/roofline_analysis.py`)

Advisor books each sample on the innermost node, and that node is often an *inlined callee* rather than the loop
(`std::isnan`, `vector::operator[]`, `AttributeValue`). In the May-2025 binary 87 % of `Evaluate`'s time sits on the
inlined `std::isnan` (the gather's latency lands on the first use of the loaded value), so a loop-only view shows
ApplyProjection at 1 %. The script therefore rebuilds Advisor's call tree from the top-down report and charges every
node's self time to the nearest enclosing kernel function — the chrono-scope definition:

| category | kernel function |
|---|---|
| ApplyProjection | `ProjectionEvaluator::Evaluate` (old May-2025 binary: the inlined row loop at `oblique.cc`, no line) |
| Split search (histogram) | `FindSplitLabelClassificationFeatureNumericalHistogram` |
| Split search (sort/scan) | `FindSplitLabelClassificationFeatureNumericalCart` (VQSort / std::sort / bucket fill / scan) |
| Projection sampling | `SampleProjection` (+ the RNG leaf functions Advisor sometimes detaches from their caller) |
| Other | everything else under the tree-training thread loop: per-node setup, label stats, partition, tree bookkeeping |

Shares are over the `ThreadPool::ThreadLoop` subtree = the training block (dataset load / trunk generation excluded).

Roofline dots: loops with Advisor's **self** metrics (FLOP or FLOP+INTOP per byte of L1 traffic; ops per self-elapsed
second), size ∝ self time, colour = category from the tree. The **star** is the ApplyProjection *kernel*: the whole
`Evaluate` subtree — Σ self FLOP and bytes of its loops and inlined callees (cross-checked against Advisor's inclusive
`Total GFLOP` where `--stacks` results have it: equal), over the function's elapsed time. Roofs are Advisor's own
measurements from the `dyn trunk100k` run (they vary a few % between runs).

Per shape directory: `adv_<arm>_<shape>[_nocs].{survey,functions,topdown,roofs}.csv`, `.roofline_{mixed,float}.html`
(Advisor's interactive roofline), `roofline_<shape>_<level>_<ops>.png`, `*.top.csv` (the plotted loops),
`instrumented_timings.txt`. `summary.csv`: per (shape, arm) training CPU time, category shares, and the AP kernel's
FLOP / bytes / elapsed / GFLOP/s / intensity (`_float` = FLOP only, `_mixed` = FLOP+INTOP; `AP_DRAM_*` only for the
cache-simulation rows, shape `trunk100k_cachesim`).

Regenerate a figure: `roofline_analysis.py --roofs <dyn trunk100k roofs.csv> --level L1|DRAM --ops float|mixed
--shape S --summary summary.csv --out X.png --title T "label=<dir>/adv_<arm>_<shape>[_nocs]" ...`.
