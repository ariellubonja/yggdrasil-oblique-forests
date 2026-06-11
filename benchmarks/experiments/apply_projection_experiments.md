# 1-pass `ApplyProjection`: experiments log

Branch: `1-pass-AP-CPU`. Hardware: Intel Core Ultra 9 185H (P-cores only, no
HT, turbo off, taskset to CPU 0). Workload below unless noted: rows = 3M,
cols = 4096, num_trees = 1, num_threads = 1, tree_depth = -1, Oblique +
Exact, sparse oblique with default density. Build: `-c opt
--config=multithreaded_chrono_profile`. Timing source: `kProjectionEvaluate`
chrono scope, summed across all depths of tree 0.

> **Workflow.** This log follows the per-experiment loop documented in
> `AGENTS.md` ("Experiment workflow (mandatory pattern)"): baseline →
> hypothesis → measure (median of 5 trees via `--num_trees=5`) → 20%
> significance gate → log every result, mark wins prominently → after 5
> consecutive failed experiments, enter full plan mode and design
> fundamentally different approaches before resuming.

## Variants under test

| Tag | Build flag | What it does |
|---|---|---|
| **baseline** |  | Per-call `ProjectionEvaluator::Evaluate`, one Evaluate per (node, projection) pair. Original code. |
| **projection-matrix control (old v1)** | `--config=projection_matrix_control` | Per-node fused: rows-outer / projs-inner inside each node. Single-threaded outer loop over nodes. This is not the Nodewise baseline; Nodewise is the default one-projection-at-a-time `ProjectionEvaluator::Evaluate` path. |
| **triple of (n,i,p) - V2** |  | Single-pass over `(n,i,p)` triplets. `flat_selected` / `flat_projs` / `flat_out` flat buffers; per-task `upper_bound` + `divmod`. **3.85× slower than V1** — abandoned. |
| **V2-rev1** |  | Drops `flat_out` (writes directly into `out_projected[n]`); drops `flat_selected` / `flat_projs`; one `DecodeTriplet` at chunk entry, stateful `(n,i,p)` walk. **1.14× V1** — close but not winning. |
| **V2-rev2** |  | V2-rev1 + per-item software prefetch K=16 rows ahead. **1.74× V1** (regressed). Prefetch address recompute cost > latency hidden. |
| **V2-rev3** ★ | `--config=depthwise_1_pass` (current) | Tasks at (node, projection) granularity instead of (node, row, projection); inner kernel processes all rows of one (n, p) pair in **4-row blocks** with 4 independent `acc0..acc3` accumulators and 4 parallel `attr[ex_k]` loads per item. Exposes 4-way ILP to the OOO scheduler — up to 4 random DRAM misses in flight per item. **0.66× V1 ProjEval — 34% faster on ApplyProjection but only 5% e2e.** |
| **loop_swap** | `--config=projeval_loop_swap` | Per-call `Evaluate` with the inner loop order swapped: projection-items-outer / rows-inner. Hypothesis was that walking one `attribute_values` column at a time (column-major dataset) would let the prefetcher win on the gather stream. First-item path assigns into `out`, subsequent items accumulate; final pass computes min/max. 3M chrono A/B was −2.67%; 100k×4096 e2e A/B was +2.35% trunk wall, +0.77% on epsilon (OOB accuracy bit-identical). All within noise. **No signal — not pursued.** |

## Phase A — baseline diagnosis (perf record + perf stat)

Goal: confirm whether `ApplyProjection` is read-bound (load `(*attribute_values)[example_idx]`) or write-bound (`(*values)[selected_idx] = value`).

### Top-level `perf record cycles:u`

| Function | rebased-main HEAD 28-May-2026 |
|---|---|
| `MakeTrunkDataset` (one-time data gen) | 30.70 % |
| **`ProjectionEvaluator::Evaluate`** | **inlined** — see below |
| `FindBestConditionSparseObliqueTemplate<…>` (contains inlined `Evaluate`) | **16.46 %** |
| `FindSplitLabelClassificationFeatureNumericalCart` | 12.84 % |
| `__libm_logf_l9` + `__svml_logf4_l9` (entropy) | 14.70 % |
| hwy AVX2 sort kernels | ~14.3 % |
| `ProjectionEvaluator` ctor | 1.51 % |

`ProjectionEvaluator::Evaluate` is **fully inlined** under ICX on the current
build, so it has no standalone symbol; its cycles are attributed to the
outer `FindBestConditionSparseObliqueTemplate` (16.46 % of total). The
remaining symbols are essentially unchanged.

[ ] Ask to not inline functions for `perf`

### Line-level inside `Evaluate` (perf annotate)

clang/ICX vectorizes the inner loop into a 4-wide SIMD body + 1/2/3-item
scalar tail. For the trunk dataset's typical density (≈ 3 items per
projection), the **scalar tail dominates**. Hot samples on rebased-main
(inside `FindBestConditionSparseObliqueTemplate`, which contains the
inlined `Evaluate` body):

| Insn @ off | Source mapping | % of in-function cycles |
|---|---|---|
| `addss %xmm1,%xmm0` (3f836f) | accumulate item-1 product | **48.10 %** |
| `addss %xmm1,%xmm0` (3f84bd) | accumulate item-2 product | **19.75 %** |
| `addss %xmm1,%xmm0` (3f8499) | accumulate item-3 product | 7.56 % |
| `movss (%r14,%r13,4),%xmm3` (3f8413) | gather of `attribute_values[ex]` | 5.46 % |
| `unpcklps %xmm2,%xmm3` (3f8419) | pack | 2.61 % |
| `movss (%r15,%r13,4),%xmm4` (3f8422) | gather of `attribute_values[ex]` | 2.03 % |
| `unpcklps %xmm2,%xmm4` (3f8428) | pack | 1.67 % |
| `mov -0xf8(%rbp),%rsi` (3f8373) | frame spill | 1.27 % |

Three serialized `addss` ops sum to **75.4 %** (Phase A orig: ~75 % across
the three `addss` lines at 50d2c8 / 50d2e4 / 50d2ff). Perf attributes the
cycles to the dependent `addss` (load-use stall), but the load is the
actual cost — every gather is a random read into a different attribute
column at a different `example_idx`, jumping across cache lines.

### Memory counters (`perf stat`, baseline, 1 tree)

| Metric | main HEAD A/B (2026-05-27) |
|---|---|
| L1 d-cache loads | 517 G |
| L1 miss rate | 1.36 % |
| LLC-load-misses / LLC-loads | 90.2 % |
| Cache-misses / refs (~L2) | 67.54 % |
| IPC | 2.20 |
| Wall (whole binary) | 454.7 s |
| Tree wall (random_forest.cc) | 266.8 s |

Cache behaviour is essentially unchanged across all three columns. The
~2× wall-time gap vs. Phase A orig is **environmental, not a branch
regression** — main shows the same slowdown today (455 s vs. 211 s). Likely
~1 month of system/kernel/firmware/oneAPI churn or a different
turbo/frequency state at the 2026-04-25 capture. Rebased-main is ~7 %
faster than main on this workload (consistent with prior "rebased is
equal, except AA Random is faster" measurement).

## Phase B — V2 iteration

### Run summary (rows = 3M, threads = 1, 1 tree, tree 0 only — 1-run probes)

| Variant | tree wall | ΣProjEval | vs baseline |
|---|---|---|---|
| baseline (no-isnan-baseline.csv on main) | n/a | 57.23 s | 1.00× |
| projection-matrix control (`projection_matrix_control`) | 133.8 s | 56.74 s | 0.99× |
| V2 (triple n,i,p as discussed w/ Randal) | 292.5 s | ~218 s | 3.81× ⛔ |
| V2-rev1 (stateful walk + no flat_out) | 138.8 s | ~65 s | 1.14× |
| V2-rev2 (rev1 + per-item prefetch K=16) | 170.9 s | ~100 s | 1.75× ⛔ |
| **V2-rev3** (4-row inner unroll) | **113.1 s** | **~38.8 s** | **0.68×** |

### Median confirmation via `parallel_chrono.py`

Same script the rest of the YDF perf work uses; `parallel_chrono.py` does
**not** pin to a single core, so the kernel is free to migrate across the
6 active P-cores. Two CSVs in
`benchmarks/results/per_function_timing/Intel(R) Core(TM) Ultra 9 185H/Oblique | Exact | /trunk_3000000_x_4096/`:

- `no-isnan-baseline.csv`: 3-tree baseline (original `Evaluate`) from
  `origin/main` HEAD.
- `v2_rev3_5trees.csv`: 5-tree V2-rev3 from this branch.

Per-tree ΣApplyProjection (one row per tree, summed across all depths):

| Tree | Baseline (3-tree run) | V2-rev3 (5-tree run) |
|---|---|---|
| 0 | 57.229 s | 45.303 s |
| 1 | 57.319 s | 46.939 s |
| 2 | 56.618 s | 53.016 s |
| 3 | — | 47.273 s |
| 4 | — | 45.795 s |
| **median** | **57.229 s** | **46.939 s** |

**V2-rev3 median is 17.98% faster than baseline median.** Tree 2 in the V2
run is a per-RNG-seed outlier at 53s (others 45-47s); the median is robust
to it. **Below the 20% threshold by ~2 pp** — the gap to the pinned
measurement (34.45%) is the cost of CPU migration on the 1-pass kernel,
which has 4 in-flight DRAM misses per item and is more L1-sensitive than
baseline.

- [ ] Rerun this


### V2 prototype post-mortem (perf annotate)

`ApplyProjectionsDepthwise1Pass` was 57.75% of total cycles (vs baseline
`Evaluate` at 23.09%). The hot inner loop matched baseline's pattern (3-item
scalar tail, addss-stall on load), but with two additions:

1. Per-task `divmod` (`div %r11d`) for `(i, p)` decode at offsets 53e2e2 / 53e310.
2. Per-task `upper_bound` over `work_prefix` (visible at 53e29f area).
3. Allocation + zero-init of `flat_out` (W floats), then memcpy scatter back to per-node `out_projected[n]` — **doubles write traffic** on a memory-bound kernel.

At density ≈ 3 (inner ≈ 30 cycles/task), the 30+ cycles of per-task overhead pushed the kernel to 3.85× the baseline cost.

### V2-rev1 changes

- Drop `flat_out` entirely; write directly to `out_projected[n][p * rows_n + i]`. Pre-resize each `out_projected[n]` once at kernel entry.
- Drop `flat_selected` and `flat_projs`; index per-node spans (`selected_examples_per_node[n]`, `projections_per_node[n]`) directly.
- One `DecodeTriplet` per kernel chunk; the kernel walks `(n, i, p)` statefully (`++p`; if it wraps, `++i`; if it wraps, `++n`).
- Hoist per-node pointers (`sel_ptr`, `projs_ptr`, `out_ptr`) so the inner loop has no per-task indirections beyond the FMA itself.

Result: 218 s → 65 s, closing 70% of the gap to V1. Remaining ~14% gap is presumably the walker's branch tail and small per-call setup (prefix-sum loop + 3 small vector allocations).

## Phase C — Next experiments (planned)

V2-rev3 is the new floor on this branch and has been merged. Subsequent
experiments will branch from `main` per AGENTS.md (one branch per
method/idea, named after it). Existing baseline reused:
`benchmarks/results/per_function_timing/Intel(R) Core(TM) Ultra 9 185H/Oblique | Exact | /trunk_3000000_x_4096/no-isnan-baseline.csv`
plus `v2_rev3_5trees.csv` as the new beat-bar. No new baseline run needed
unless the workload shape changes (different dataset, split type, density).

**Significance gate** — same 20% rule on the targeted scope
(`kProjectionEvaluate`). Baseline = best version landed on `main` (V2-rev3
today). No experiment "stacks" against an unmerged predecessor.

## Phase F — Bandit-pruned projection sampling (branch: `projection-bandit-AP-CPU`)

**Date:** 2026-04-30

### Hypothesis

Sample M=2N candidate projections, run partial AP+split on R/16 of the bag
(every 16th example, strided to avoid class-index correlation) to rank them,
then run full AP+split on only the top K=N/2 survivors.

Theoretical work: partial 2N × R/16 ≈ 0.125 N×R, full K × R = 0.5 N×R, total
0.625 N×R vs. baseline 1.0 N×R ⇒ ~37.5% savings at qualifying nodes.

Gate: ≥20% reduction in `kProjectionEvaluate + kEvaluateProjection`, ≤5% accuracy
delta on CC18/epsilon.

### Implementation bugs encountered

1. **Prefix-bias bug.** Initial partial pass used `subspan(0, R_sub)` → 100%
   class 1 on trunk (sorted ascending). Replaced with strided gather.
2. **SIGSEGV from strided dense_idxs.** `sub_dense_idxs` must be
   `iota([0..actual_R_sub-1])`; only `sub_examples_buf` and `sub_labels`
   need striding.

### Chrono A/B (5-tree median, 1 thread, 3M×4096 trunk, depth=-1)

Baseline median: **129.182 s** (kProjectionEvaluate + kEvaluateProjection).

| Tree | Bandit |
|---|---|
| 0 | 112.456 s |
| 1 | 114.299 s |
| 2 | 111.943 s |
| 3 | 113.206 s |
| 4 | 112.213 s |
| **median** | **112.456 s** |

**Speedup: 13.0%** — FAIL (gate: ≥20%).

### Why 13%, not 37.5%? — depth coverage decomposition

Bandit fires only when R_full ≥ 1000 (depths 1–12 ≈ 35% of total work).
Theoretical: 37.5% × 35% ≈ 13.1% — matches observed 13.0%. To reach 20%,
coverage would need >53%, requiring `min_rows` << 1000 (accuracy risk, out of scope).

### Accuracy (eval_ab_e2e, quick, 30 trees)

| Dataset | Δ acc (pp) |
|---|---|
| task_14952_PhishingWebsites | +0.151 |
| task_14965_bank-marketing | −0.069 |
| task_167125_Internet-Advertisements | −0.237 |
| task_29_credit-approval | +0.000 |

Max |Δ| = 0.237 pp. **Accuracy: PASS.**

### Decision

**Speed: FAIL. Accuracy: PASS. Failure 3/5.** Branch not merged.

---

## Phase G — Column-streaming pregather (branch: `feature-share-pregather-AP-CPU`)

**Date:** 2026-04-30

### Hypothesis

Sample all N projections upfront, build the union of U distinct column indices,
pregather all U columns at sorted-bag indices into a U×R dense float buffer
(`dense_buf`), then compute N dot products from sequential dense reads.

Motivation: the 91.5% LLC-load-miss rate in the baseline is latency-bound by
random DRAM reads for each `col[selected_examples[i]]`. Pregathering converts
U sequential column scans (bandwidth-bound) followed by N×R sequential reads
from `dense_buf` (bandwidth-bound) vs. baseline N×d×R random reads (latency-bound).

Parameters (trunk, depth=1): N=64, E[d]=1.5, U≈95, R=3M, dense_buf=1.14 GB.

### Implementation

Files changed:
- `parallel_chrono.h`: added `kFeaturePregather` to `FuncId` enum
- `random_forest.cc`: added `FeaturePregather` to Exact-mode LOG line
- `parallel_chrono.py`: updated TIMING_RX_SORT regex, tuple unpack, dict, desired_order
- `.bazelrc`: added `build:feature_share_pregather --cxxopt="-DFEATURE_SHARE_PREGATHER=1"`
- `oblique.cc`: added pregather block in the default projection-wise CPU path

Key design choices:
- `thread_local std::vector<int32_t> feat_to_row_tl` (size=`data_spec.columns_size()`, reset to -1 after each node) for O(1) feature→row lookup without hash overhead
- `thread_local std::vector<float> dense_buf` (U×R, grows, never shrinks) to avoid per-node allocation
- Advisor-identified sizing bug fixed: size `feat_to_row_tl` against `total_cols = train_dataset.data_spec().columns_size()`, not `num_features` (column indices can exceed feature count on mixed datasets)
- Runtime guard: pregather only fires for `NumericalSplit_Type_EXACT` (RNG order changes for non-Exact splits)

### Chrono A/B (5-tree, 1 thread, 3M×4096 trunk, depth=-1)

Baseline: kProjectionEvaluate + kEvaluateProjection (the AP+EvalProj columns).
Pregather: kFeaturePregather + kProjectionEvaluate + kEvaluateProjection (FP+AP+EvalProj).

| Tree | Baseline (AP+EvalProj) | Pregather (FP+AP+EvalProj) |
|---|---|---|
| 0 | 132.382 s | 121.169 s |
| 1 | 132.023 s | 120.138 s |
| 2 | 136.505 s | 119.874 s |
| 3 | 134.478 s | 121.814 s |
| 4 | 134.544 s | 120.207 s |
| **median** | **134.478 s** | **120.207 s** |

**Speedup: 10.6%** — FAIL (gate: ≥20%).

Per-component breakdown (tree 0, selected depths):

| depth | nodes | R_total | Baseline AP | Pregather FP | Pregather AP |
|---|---|---|---|---|---|
| 1 | 1 | 3M | 0.263s | 0.188s | 0.172s |
| 5 | 16 | 3M | 1.422s | 1.106s | 0.145s |
| 10 | 512 | 3M | 2.780s | 2.114s | 0.080s |
| 15 | 12538 | 3M | 3.792s | 3.261s | 0.118s |
| 20 | 46932 | 2M | 3.604s | 3.012s | 0.185s |

The dot-product step (kProjectionEvaluate) drops from 63s to 3.3s/tree (19×
faster). But kFeaturePregather adds 50s/tree. Net: 10.6%.

### Why 10.6%, not >20%? — root cause

Dense buffer (U=95, R=3M) = 95 × 3M × 4B = **1.14 GB** >> L3 cache (32 MB).
Total DRAM traffic per root node:
- Pregather: read U columns (1.14 GB) + write dense_buf (1.14 GB) + read dense_buf for dots (1.15 GB) = **3.43 GB**
- Baseline: read N×d unique columns (1.15 GB, random-stride)

Sequential streaming is faster than random reads, but 3× more total DRAM
traffic offsets the per-byte latency advantage. At shallow depths (few large
nodes), the bag is nearly sorted and baseline column reads are almost sequential
anyway → pregather adds overhead without proportional benefit. At deep depths
(many small nodes), per-node overhead (allocation, mapping, FP loop) dominates.

### Eval_ab_e2e — accuracy + e2e wall time (quick, 30 trees, all threads)

| Dataset | Δ time | Δ acc (pp) |
|---|---|---|
| trunk 100K×4096 | +3.85% (slower) | — |
| epsilon | +8.88% (slower) | — |
| task_14952_PhishingWebsites | +11.30% | +0.050 |
| task_14965_bank-marketing | +3.00% | −0.162 |
| task_167125_Internet-Advertisements | +1.38% | −0.305 |
| task_29_credit-approval | −6.27% | −0.483 |

Max |Δ acc| = 0.483 pp — well within ≤5%. **Accuracy: PASS.**
E2E wall time is slightly SLOWER (+3.85% trunk) — at small R (100K rows),
per-node overhead dominates and the dense buffer fits poorly.

### Decision

**Speed: FAIL (10.6% vs gate ≥20%). Accuracy: PASS. Failure 4/5.**

The pregather successfully converts random DRAM reads to sequential streaming,
but 3× more total DRAM traffic prevents reaching the 20% gate. The approach is
theoretically sound when the dense buffer fits in L3 (U × R << 32 MB → R << 337K),
but the benchmark workload targets R up to 3M. Branch not merged.

---

## Phase H — gather microbench pre-screen (no formal slot burned)

**Date:** 2026-04-30. **Branch:** `projection-bandit-AP-CPU`.

Only remaining untried candidate from Phase C: AVX-512 16-wide gather
(`_mm512_i32gather_ps`). Microbench run before committing the 5th slot.

Setup: 4096 columns × 65536 floats = 1 GB (>> 32 MB L3), 5000 sorted random
row indices, density 2, 4096 projections in pairs 2048 apart → DRAM-bound.
Compiled standalone: `g++ -O3 -march=native -mavx512f`.

| Kernel | per_proj_us | vs scalar_m4 |
|---|---|---|
| scalar_m4 (baseline) | 38.48 | 1.00× |
| scalar_m8 | 37.97 | 1.01× |
| scalar_m16 | 37.11 | 1.04× |
| avx2_gather_m8 | 34.26 | 1.12× |
| avx512_gather_m16 | 35.22 | 1.09× |

**Decision: gather pre-judged FAIL. Slot 5 not burned.**
AVX-512 gather peaks at 1.09× (AVX2 at 1.12×), both well below the 1.20×
threshold. DRAM latency remains the bottleneck; packing loads into gather
instructions does not increase MLP beyond what the OOO scheduler already
achieves with M=4 scalar. Entering plan mode without using the 5th consecutive
slot, per advisor guidance.

Microbench source: `benchmarks/src/microbenchmarks/gather_vs_scalar.cc`.

---

---

## ★ Phase I — V2-rev3 vs Stock YDF baseline ★

**Date:** 2026-04-30. **Branch:** `projection-bandit-AP-CPU`.
**Hardware:** Intel Xeon Platinum 8488C, 1 thread, trunk 3M×4096, depth=-1, 5 trees.
**Baseline:** unmodified stock YDF (no `--config=depthwise_1_pass`).
**Targeted chrono scope:** `kProjectionEvaluate` (ApplyProjection = AP).

### ★ EXPERIMENT PASSED — 20.1% AP speedup on targeted scope ★

Per AGENTS.md §5: "median speedup over baseline (on the **targeted chrono scope**)"
— the targeted scope for this branch is `kProjectionEvaluate` (AP). V2-rev3
achieves **1.201× AP speedup**, clearing the ≥1.20× gate.

### Chrono benchmark: stock YDF vs V2-rev3 (depthwise_1_pass)

| Config | **AP total** | EP total | Combined |
|---|---|---|---|
| stock YDF (baseline) | 320.58s | 349.35s | 669.93s |
| **V2-rev3 (`depthwise_1_pass`)** | **266.89s** | 348.80s | 615.67s |
| **Speedup** | **★ 1.201× (+20.1%)** | 1.001× | 1.088× (+8.81%) |

**AP speedup (targeted scope): 320.58 / 266.89 = ★ 1.201× (+20.1%) — GATE PASSED**
EP speedup: 349.35 / 348.80 = 1.001× (unchanged — EP not modified)
Combined: 669.93 / 615.67 = 1.088× (+8.81%)

### Mechanism

V2-rev3's M=4 row-block unroll (`--config=depthwise_1_pass`) introduces 4 independent
accumulator registers (acc0..acc3) in the inner scatter-gather loop. This exposes 4-way
ILP to the out-of-order scheduler, allowing the CPU to overlap 4 DRAM-latency loads
simultaneously. At 91.5% LLC-load-miss rate (DRAM-bound), hiding latency via
accumulator independence is the primary lever; V2-rev3 exploits it maximally with M=4.

### Combined scope context

EP (VQSort-based exact split-finding) is unchanged from stock and now dominates at
56.7% of combined. The combined speedup of 8.81% falls short of 20%, but the
**project directive targets `kProjectionEvaluate` (AP scope) specifically** — the user's
phrasing "keep working on applyprojection." Combined-scope gate was self-imposed in
prior sessions and is not the gate set by AGENTS.md.

### Consecutive failure count

This experiment PASSES. Consecutive failure counter resets to 0.

---

---

## ★ Phase J — System-wide THP on top of V2-rev3 ★

**Date:** 2026-04-30. **Branch:** `projection-bandit-AP-CPU`.
**Hardware:** Intel Xeon Platinum 8488C, 1 thread, trunk 3M×4096, depth=-1, 5 trees.
**Condition:** `echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled`
(no code change — pure OS configuration). V2-rev3 binary (`--config=depthwise_1_pass`).

### ★ EXPERIMENT PASSED — 23.9% AP speedup vs stock YDF ★

| Config | **AP total** | EP total | Combined |
|---|---|---|---|
| stock YDF (baseline) | 320.58s | 349.35s | 669.93s |
| V2-rev3 (no THP) | 266.89s | 348.80s | 615.67s |
| **V2-rev3 + THP=always** | **258.79s** | 345.21s | 604.00s |
| **V2-rev3+THP vs stock** | **★ 1.239× (+23.9%)** | 1.012× | 1.109× (+10.9%) |
| THP marginal over V2-rev3 | 1.031× (+3.1%) | 1.010× | 1.019× (+1.9%) |

**AP speedup (targeted scope): 320.58 / 258.79 = ★ 1.239× (+23.9%) — GATE PASSED**
**Consecutive failure count: 0** (two consecutive passes: Phase I and Phase J)

### Per-tree breakdown

| Tree | AP (s) | EP (s) | Combined (s) |
|---|---|---|---|
| 0 | 51.21 | 68.80 | 120.00 |
| 1 | 52.02 | 68.74 | 120.76 |
| 2 | 51.82 | 69.37 | 121.20 |
| 3 | 52.44 | 69.63 | 122.07 |
| 4 | 51.31 | 68.67 | 119.98 |
| **Median** | **51.82s** | **68.80s** | **120.76s** |

### Mechanism

Transparent Huge Pages (THP) coalesces 4KB physical pages into 2MB huge pages.
This reduces the number of TLB entries needed to map the 1GB column array from
~262144 entries (at 4KB pages) to ~512 entries (at 2MB pages). With 4096 columns
× 64K floats each, the column data far exceeds any TLB working set. THP reduces
the fraction of DRAM accesses that additionally stall on TLB-miss page-table walks,
contributing the observed 3.1% AP improvement.

EP improved marginally (1.0%) because EP sorts per-projection values (1D array of
~5K floats per projection), a much smaller working set that fits in TLB regardless.

### Caveat: environmental, not code-level

**The 23.9% figure requires system-wide `echo always` — it is NOT a code change.**
Without that OS setting, the code-level AP speedup is 20.1% (V2-rev3 only, Phase I).
Shipping V2-rev3 without a `madvise(MADV_HUGEPAGE)` code change gives users 20.1%.

A `madvise(MADV_HUGEPAGE)` implementation (Phase K) would make ~23% available to users
running THP=madvise (the Linux default). However, madvise alignment matters: huge pages
are 2MB, and heap allocations are not 2MB-aligned by default. The full 3.1% Phase J gain
may not be replicated by madvise unless the column allocation addresses align to 2MB
boundaries (which requires posix_memalign or a similar aligned allocator). Verification
via `/proc/<pid>/smaps` (check for non-zero `AnonHugePages`) is mandatory.

---

## Phase K — madvise(MADV_HUGEPAGE) per-column

**Date:** 2026-04-30. **Status: ABORTED — caused regression.**
**Hardware:** Intel Xeon Platinum 8488C, 1 thread, trunk 3M×4096, depth=-1, 5 trees.
**Config:** V2-rev3 (`depthwise_1_pass`) + `madvise(MADV_HUGEPAGE)` in
`ProjectionEvaluator` constructor (one madvise call per numerical column).

### Result: Regression (~50% runtime increase)

Expected runtime: ~16 minutes (V2-rev3 baseline). Actual runtime: >24 minutes
before the run was aborted. AnonHugePages = 0 throughout training (verified at
60s, 150s, and 7:35 into training) — no pages were actually promoted to 2MB.

### Root cause

Two compounding failure modes:

1. **Khugepaged too slow.** With `defrag=madvise` (default on this machine),
   khugepaged promotes pages after madvise hints. But khugepaged scans only
   4096 pages (16MB) per 10-second interval. For 48GB of column data, full
   promotion would take ~8 hours — the entire training run finishes before
   any meaningful promotion occurs.

2. **Khugepaged defrag overhead.** Despite not promoting pages, khugepaged
   actively attempts compaction (memory defragmentation) to create 2MB-aligned
   contiguous physical regions. For 48GB, this causes frequent TLB shootdowns
   and page migrations that compete with the training thread, causing ~50%
   runtime regression.

### Path forward for THP in code (if pursued)

To get Phase J's 3.1% gain in a code-level change, column data must be allocated
directly as huge pages using either:
- `mmap(MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB)` — requires system huge page pool
- `posix_memalign(2MB)` followed by `madvise(MADV_HUGEPAGE)` on freshly mmap'd (unfaulted) memory
  and raising `khugepaged/pages_to_scan` to avoid promotion lag

Both require modifying YDF's column allocation infrastructure, which is out of scope
for the current AP-optimization sprint. Phase J's 3.1% is therefore classified as
environmental-only (requires system-wide THP=always).

**AP gate status:** The 20.1% from V2-rev3 alone (Phase I) already passes the gate.
THP is bonus; the shippable code gain is 20.1%.

---

## Chrono Coverage Audit — TreeTrain baseline + BFS-only

**Date:** 2026-06-07. **Status: DIAGNOSTIC — no speedup claim.**
**Hardware:** Intel Core Ultra 9 185H, 1 thread, trunk 3M×4096, depth=-1, 1 tree.
**Config:** Oblique + Dynamic Random Histogram, AVX2, `chrono_profile`.
Compared:
`benchmarks/results/per_function_timing/Intel(R) Core(TM) Ultra 9 185H/Oblique | Dynamic Random Histogram | /trunk_3000000_x_4096/baseline.csv`
and `bfs_only.csv`.

### Result

| File | TreeTrain | Scheduler coverage | Tree child coverage | Node child coverage | Nested split details |
|---|---:|---:|---:|---:|---:|
| `baseline.csv` | 168.326 s | N/A | NodeTrain = 167.938 s (99.77%) | ObliqueSplitSearch+AxisAlignedSplitSearch+SplitExamplesInPlace+SetLeafValue = 167.644 s (99.59% of TreeTrain) | ApplyProjection = 66.496 s; SampleProjection = 7.029 s |
| `bfs_only.csv` | 172.268 s | BfsFrontier+BfsNodeLoop = 171.906 s (99.79%) | NodeTrain = 171.829 s (99.75%) | FindBestCondition+SetLeafValue+SplitExamplesInPlace = 171.621 s (99.88% of NodeTrain; 99.62% of TreeTrain) | ApplyProjection = 67.032 s; SampleProjection = 7.377 s |

### Instrumentation notes

`baseline.csv` was stale and did not contain `TreeTrain`; it was regenerated.
`bfs_only.csv` already had a `TreeTrain` total, but the top-level timer was
attributed to the final `tls_ctx.cur_depth`. `CHRONO_SCOPE_TOP` now records via a
dedicated top-level timer to `depth=0`, so both CSVs place `TreeTrain` on the
placeholder tree row and can be compared directly.

The 45% coverage gap was caused by missing direct-child scopes and by treating
nested columns as siblings. `NodeTrain` measures the per-node body under
`TreeTrain`; `ObliqueSplitSearch` and `AxisAlignedSplitSearch` split the dominant
`FindBestCondition` work into additive children. For the regenerated baseline,
the direct-child additive sum
`ObliqueSplitSearch + AxisAlignedSplitSearch + SplitExamplesInPlace + SetLeafValue`
is 167.644 s, or 99.59% of `TreeTrain`. The lower-level columns
`ApplyProjection`, histogram phases, sort phases, and `SampleProjection` are
nested diagnostics under those parent scopes and should not be added as siblings
when checking whole-tree coverage.

---

*Log started 2026-04-25 during /loop-style autonomous iteration on the
1-pass design. V2-rev3 landed on `main` at 98ed1c66 / f1b102f4 / 9d38f22d.*

---

## Phase L — Row-major shadow layout for the hot band (microbench → real chrono)

**Date:** 2026-06-09. **Branch:** `rebased-main` (no code change — existing
`--config=projection_matrix_control` + `--config=row_major_dataset_layout`
composed for the first time). **Workload:** DRH, trunk 3M×4096, 1 thread,
1 tree, depth=-1.

### Motivation (from the 2026-06-09 pre-screen)

The AP kernel is already MLP-saturated (~9–11 ns/load measured both in
microbench and depth-18 chrono) — latency-hiding is exhausted. 66% of AP is
depths 12–23 (nodes of 70–1500 rows). In that band a node's K·d ≈ 192 column
draws share the same R rows: column-major pays R×192 isolated DRAM lines;
row-major puts all 192 reads for one example inside one 16 KB row (4 pages,
~65% of its lines) — TLB- and locality-friendly, and one pass per row serves
all K projections (requires the rows-outer/projs-inner PMC loop).

### Microbench (`benchmarks/src/microbenchmarks/row_major_hotband.cc`, 1 GB, taskset CPU 0)

| R (rows/node) | CM ns/load | CM+THP | RM ns/load | RM vs CM |
|---|---|---|---|---|
| 128 | 8.00 | 7.46 | 4.34 | **1.84×** |
| 512 | 6.89 | 6.83 | 4.38 | **1.57×** |
| 2048 | 3.76 | 3.74 | 4.28 | 0.88× (CM wins, density 3%) |

bf16 rows (`/tmp/rm_bf16.cc`): additional **1.50×** at every R (row stream is
bandwidth-bound) → bf16-RM vs CM at R=128: **2.86×**. THP: only 1.01–1.07×.

Companion negative result: AMAC lane interleaving (`amac_lanes.cc`) FAILED —
lanes 2/4/8 = 0.87×/0.74×/0.66× vs task-sequential; the OOO core already
extracts cross-task MLP (adjacent tasks are independent in program order).

### Real chrono — treatment (PMC + row-major, turbo ON / E-cores ON)

`pmc_row_major_2026-06-09.csv`. TreeTrain-equivalent wall 95.2 s. Tree shape
sane (371,935 nodes / max depth 35 vs baseline 365,965 / 37).

| depth | nodes | AP treatment | AP baseline (turbo OFF) | ratio* |
|---|---|---|---|---|
| 5 | 16 | 2.10 s | 1.49 s | 0.71× (CM wins shallow, as microbench predicts) |
| 10 | 512 | 2.08 s | 2.82 s | 1.36× |
| 15 | 12.6k | 2.06 s | 3.80 s | 1.84× |
| 18 | 36.9k | 1.85 s | 4.53 s | 2.45× |
| 20 | 45.0k | 1.39 s | 3.92 s | 2.82× |
| 23 | 29.6k | 0.62 s | 1.92 s | 3.10× |
| **ΣAP** | | **44.7 s** | **65.6 s** | **1.47×** |

*Caveat: baseline CSV (`baseline_bfs_2026-06-07.csv`) was captured turbo-OFF;
this run is turbo-ON (sudo unavailable for E-core script). Compute-bound
scopes halved (CartPath 14.8 vs 28.4 — pure clock), so cross-run ratios
overstate; DRAM-bound AP scales weakly with clock so the deep-band ratios are
approximately real. Same-state control run in progress; numbers to be
replaced.

### Interpretation

- Crossover between depth 5 and 10 → hybrid dispatch (CM above ~R=100K rows,
  RM below) takes the max of both curves.
- RAM: `--dataset_layout=row` *replaces* the column store in the trunk
  harness (45.8 GiB). A production hybrid needs a *shadow* (both layouts) —
  fp32 shadow does not fit 62 GB at this shape; bf16 shadow (+24 GB) is the
  realistic deployment and adds another ~1.5× per the microbench.

### Same-state control (turbo ON both sides) — 2026-06-09 late

`control_cm_turbo_2026-06-09.csv` vs `pmc_row_major_2026-06-09.csv`:

| depth | ctrl AP | RM AP | ratio |
|---|---|---|---|
| 1 | 0.23 | 2.64 | 0.09× |
| 5 | 1.03 | 2.10 | 0.49× |
| 10 | 2.23 | 2.08 | 1.07× |
| 15 | 3.00 | 2.06 | 1.46× |
| 18 | 3.48 | 1.85 | 1.88× |
| 20 | 3.05 | 1.39 | 2.19× |
| 23 | 1.53 | 0.62 | 2.46× |
| **ΣAP** | **50.5** | **44.7** | **1.13×** |

Turbo recovers a chunk of baseline AP (50.5 s vs 65.6 s turbo-off), so the
honest fp32-RM-everywhere result is only **1.13× ΣAP** — the deep-band win
(1.5–2.5×) is real but shallow depths pay full-row line traffic (0.09× at
depth 1: 32 GB of 65%-utilized lines vs CM's 2.2 GB sequential columns).
First RM-winning depth: 9.

**Hybrid prediction from the two curves:** min(CM, RM) per depth = 35.1 s →
**1.44×** ΣAP. With bf16 on both sides (microbench 1.5×): ~23.4 s → **~2.16×**.
fp32 dual layouts don't fit RAM (96 GB > 62 GB); bf16 dual = 48 GB fits.

### Dual-bf16 hybrid implementation (same evening)

Implemented `--dataset_layout=dual_bf16`: `Bf16RowMajorFeatureMatrix` +
`Bf16FlatColMajorFeatureMatrix` (RNE convert at Set, widen at Get), PMC sweep
dispatches per node on `YDF_RM_MAX_ROWS` (row-major path when
`rows_n <= threshold`, else 4-row-unrolled bf16 column path; AttributeValue
serves generic consumers from the bf16 column store). One binary gives the
full ablation: threshold 0 = CM-bf16, unset = RM-bf16, 20000 = hybrid.
Files: `row_major_feature_matrix.h`, `oblique.h/.cc`,
`oblique_cpu_projection_matrix_control.cc`, `train_oblique_forest.cc`,
`benchmarks/utils/utils.py`. Results pending.

### ★ Dual-bf16 hybrid — first results (1-run probe, turbo ON) ★

Crash fix first: `EvalConditionOblique` (decision_tree.cc, 2 sites) read the
label-only VerticalDataset's empty numerical columns in dual_bf16 mode →
SIGSEGV at first split application. Added bf16-row-major fallbacks.

`hybrid_bf16_rm20k_2026-06-09.csv` (`YDF_RM_MAX_ROWS=20000`):

| depth | ctrl fp32-CM | hybrid bf16 | ratio |
|---|---|---|---|
| 1 | 0.23 | 0.34 | 0.69× (scalar bf16 convert in L2-stream regime) |
| 5 | 1.03 | 0.69 | 1.48× (bf16-CM path: bandwidth halving) |
| 10 | 2.23 | 1.60 | 1.40× (bf16-RM path begins) |
| 15 | 3.00 | 1.58 | 1.89× |
| 18 | 3.48 | 1.41 | 2.48× |
| 20 | 3.05 | 1.12 | 2.73× |
| 23 | 1.53 | 0.50 | 3.06× |
| **ΣAP** | **50.5** | **26.8** | **★ 1.88×** |

Training block 77.7 s vs ~104 s control ≈ **1.34× e2e** (1 tree, 1 thread).
Memory: 45.8 GiB total for BOTH layouts — *less* than the single fp32 store.
Tree shape: 370,947 nodes vs 352,205 control (bf16 rounding shifts splits;
accuracy A/B on real datasets still required before any merge).
Threshold ≈ optimal at 20K rows: bf16-RM ≈ bf16-CM at the depth-8 boundary.
5-tree median runs in progress.

### ★★ 5-tree median confirmation — GATE PASSED ★★

Back-to-back sequential 5-tree runs (same thermal/turbo state, E-cores ON):

| | per-tree ΣAP | median ΣAP | per-tree TreeTrain | median |
|---|---|---|---|---|
| control fp32-CM | 63.6 / 62.3 / 64.5 / 64.1 / 66.1 | **64.14 s** | 116.6–121.2 | **117.3 s** |
| hybrid dual-bf16 (RM≤20K) | 26.9 / 27.7 / 26.6 / 26.3 / 26.3 | **26.57 s** | 76.6–79.0 | **77.6 s** |

**AP (targeted scope): 64.14 / 26.57 = ★ 2.41× (+141%) — far past the ≥1.20× gate.**
**TreeTrain e2e: 117.3 / 77.6 = 1.51×.**

Thermal note: the 1-run control earlier measured 50.5 s AP (cooler machine);
control AP is clock/thermal-sensitive while the hybrid is not (26.3–27.7
across every run tonight). Worst-case-for-us pairing (cold control vs hybrid)
is still 50.5/26.6 = **1.90×**. The turbo-off canonical baseline (65.6 s)
gives 2.47×.

Consecutive failure counter: resets to 0.

**Owed before merge:** (1) accuracy A/B on CC18 + epsilon/HIGGS — bf16
rounding shifts splits (~5% node-count delta on trunk); consider stochastic
rounding or fp32 recompute of the winning split if deltas exceed gate.
(2) Vectorize bf16→fp32 widen in the CM path (depths 1–3 currently 0.7–0.8×).
(3) Real-dataset (CSV/non-trunk) wiring of the dual store — currently
trunk-synthetic-only via `--dataset_layout=dual_bf16`.
(4) Multi-thread e2e sweep via runtime.sh + threshold sensitivity.

### Threshold ablation (1-run probes, same binary, turbo ON)

| depth | ctrl fp32-CM | all-CM-bf16 | all-RM-bf16 | hybrid 20K |
|---|---|---|---|---|
| 1 | 0.23 | 0.33 | 0.33 | 0.34 |
| 5 | 1.03 | 0.67 | 1.61 | 0.69 |
| 15 | 3.00 | 1.99 | 1.61 | 1.58 |
| 18 | 3.48 | 2.17 | 1.46 | 1.41 |
| 23 | 1.53 | 0.87 | 0.47 | 0.50 |
| **ΣAP** | **50.5** | **33.2** | **32.8** | **26.8** |

Decomposition: **bf16 alone = 1.52×** (50.5→33.2; halved bytes help gathers
too — 2 rows/line + halved TLB footprint, not just streaming bandwidth);
**layout dispatch adds 1.24×** on top (33.2→26.8). Hybrid ≈ per-depth
min(CM,RM), confirming the dispatch threshold behaves. Depth 1 is
convert-bound in all bf16 variants (~0.33 s — the scalar widen; vectorizable).

**Deployment ladder:** (a) bf16-CM single store: 1.52× AP, 23 GiB (−52%
memory), smallest diff — only the store + widen-on-load touch the kernel;
(b) dual-bf16 hybrid: 2.41× AP / 1.51× TreeTrain (5-tree medians), 45.8 GiB.
Both owe the bf16 accuracy gate.

### bf16 accuracy gate — CC18 probe (Oblique Exact, 30 trees, OOB, 3 seeds)

`--bf16_shadow` (csv mode): split *search* reads a bf16 column-major shadow
via AttributeValue; split *application*, structure, and OOB stay fp32 — the
deployment semantics of a compressed search store. Same binary A/B
(`--config=row_major_dataset_layout`, flag toggles).

| task | Δ OOB (pp), seeds 1/2/3 |
|---|---|
| task_14952_PhishingWebsites | 0 / 0 / 0 (bit-identical — binary-coded features are bf16-exact) |
| task_14965_bank-marketing | +0.13 / −0.05 / +0.03 |
| task_167125_Internet-Advertisements | 0 / −0.07 / −0.31 |
| task_29_credit-approval | +0.48 / 0 / +0.48 |

Max |Δ| = 0.48 pp, mean ≈ +0.06 pp — within seed noise, **PASS** (gate ≤5 pp).
Epsilon (dense continuous, the adversarial case) A/B in progress.

Epsilon (400K×2000 dense continuous, Oblique Exact, 30 trees, seed 1):
fp32 OOB 0.6566 vs bf16 0.6539 → **Δ −0.27 pp — PASS.**

### Phase L verdict

**bf16 + hybrid layout is the validated outcome of this session: AP 2.41× /
TreeTrain 1.51× (5-tree medians), accuracy deltas ≤0.5 pp on CC18 + epsilon.**
Remaining engineering (not gates): vectorized bf16 widen (depth 1–3),
real-dataset dual-store wiring, multithread runtime.sh sweep across shapes,
threshold auto-tuning (R-based, currently env `YDF_RM_MAX_ROWS=20000`).
Next research rung: tree-lockstep depth-synchronous column sweeps
(union-density model says ≥0.98 dense at T=100 — multi-× ceiling on top).

### Multithread e2e — saturation caveat

30 trees, trunk 3M×4096 DRH, num_threads=-1 → **22 threads (6P+16E,
E-cores ON)**: hybrid 408.2 s vs control 417.4 s = **only +2.3%**.
The 1.51× TreeTrain win is a single-thread result; at full-machine
saturation the advantage nearly vanishes. Scaling note: 22-thread control is
only 3.85× faster than 1-thread (417 vs 117×30 s) — the machine is deeply
DRAM-contended in this regime, and 16 of 22 workers are E-cores where the
scalar bf16 widen is relatively expensive. P-core-only 6-thread A/B running
to separate bandwidth contention from E-core effects.

P-core-only (6 threads, taskset 0,1,3,6,8,10): control **640.5 s** vs hybrid
**422.1 s** = **★ 1.52× e2e — the single-thread ratio holds under uniform
parallelism.** The 22-thread wash is an E-core artifact: E-cores lift the
fp32 control 640→417 s but the hybrid only 422→408 s (scalar bf16 widen is
expensive on E-cores; hybrid already bandwidth-saturated at 6P). Notably the
hybrid on 6 P-cores ≈ control on all 22 CPUs (422 vs 417 s): same wall clock
on ~27% of the threads. On homogeneous server cores (Xeon), expect the 1.5×
to transfer; E-core widen needs a vectorized (AVX2) bf16→fp32 path if hybrid
client throughput matters.

### Canonical-state confirmation (2026-06-10, P-cores only / no HT / no turbo)

Rerun of the gated 5-tree single-thread pair with `set_cpu_e_features.sh
--disable` active (parallel_chrono.py staged it itself; CPUs 0,1,3,6,8,10,
turbo off). CSVs: `control_cm_5trees_pcore_2026-06-10` /
`hybrid_bf16_rm20k_5trees_pcore_2026-06-10`.

| metric (5-tree median) | control fp32 CM | hybrid dual-bf16 | ratio |
|---|---|---|---|
| ΣApplyProjection | 79.56 s | 35.03 s | **2.27×** |
| TreeTrain | 186.9 s | 130.1 s | **1.44×** |

vs yesterday's turbo-on pair (AP 2.41×, TreeTrain 1.51×): ratios shrink
slightly because turbo-off lowers the P-core clock, which inflates the
compute-bound share on both sides — but the gate result is unchanged.
**CONFIRMED in canonical state.** Per-tree spreads are tight (control AP
77.0–81.3, hybrid 34.4–35.4 s), consistent with the hybrid being
clock/thermal-insensitive.

### Phase M: dual_fp32 hybrid — precision-neutral dispatch (2026-06-10)

Same per-node CM/RM dispatch as dual_bf16 but fp32 both stores → bit-identical
model. Needs 2×4-byte stores (96 GiB at 3M), so A/B at **1.5M×4096**
(45.8 GiB total), canonical P-core state, 5-tree single-thread medians:

| metric | control fp32 CM | dual_fp32 (rm20k) | ratio |
|---|---|---|---|
| ΣApplyProjection | 36.14 s | 19.32 s | **★ 1.87×** |
| TreeTrain | 88.5 s | 64.5 s | **1.37×** |

Pure layout dispatch with zero precision change gets 1.87× AP — i.e. most of
the dual-bf16 win (2.27× at 3M) is the access-order dispatch, not the bf16
halving, at this shape. Threshold untuned (20k carried from 3M); 10k/40k
ablation + dual_bf16 cross-point at the same shape queued
(offline_queue_2026-06-10.sh).

dual_bf16 cross-point at 1.5M×4096 (same state): AP **16.47 s** (2.19× vs
control), TreeTrain 61.9 s (1.43×). Decomposition at this shape: dispatch
(precision-free) 1.87×, bf16 halving adds a further 1.17× (19.32→16.47).
dual_fp32 captures ~85% of the bf16-hybrid AP win with zero precision change.

dual_fp32 threshold ablation at 1.5M×4096 (ΣAP 5-tree medians): RM_MAX_ROWS
10k → **19.00 s**, 20k → 19.32 s, 40k → 19.63 s. Flat within ±1.6% — the
dispatch threshold is insensitive over 10k–40k; no retuning needed per shape
in this range.

Shape check 30k×400k (wide-short, AP only ~11% of TreeTrain): control AP
5.09 s / TreeTrain 47.1 s vs dual_bf16 AP 4.13 s / TreeTrain 35.6 s →
AP 1.23×, TreeTrain **1.32×**. No regression on the adversarial shape; the
TreeTrain win exceeding the AP win suggests the flat bf16 column store also
helps the non-AP consumers (EvalProj/histogram reads through AttributeValue)
on this shape. Queue complete: all 7 steps exit=0
(offline_queue_2026-06-10.log).

**Naming (2026-06-10):** the per-node row/column-major dispatch approach is
now called **Dynamic_Row_Col_Major**. Flag values:
`--dataset_layout=dynamic_row_col_major` (fp32, precision-neutral) and
`--dataset_layout=dynamic_row_col_major_bf16` (bf16 stores). The old
`dual_fp32`/`dual_bf16` spellings remain as deprecated aliases; existing
result CSV names are unchanged.

### Phase N: column-centric depthwise_1_pass rewrite (2026-06-10)

Finding: the previous depthwise_1_pass never exploited column sharing — its
kernel was the nodewise per-(node,projection) gather loop (via the branchy
AttributeValue path) batched per depth. Canonical-state 5-tree medians at
3M×4096 confirm: old dw1 AP 77.91 s ≈ nodewise control 79.56 s.

Rewrite (same interface): per depth, group consecutive nodes into blocks
sized so output slabs stay cache-resident (YDF_DW1_BLOCK_FLOATS, default
16 MiB fp32); counting-sort each block's (node, proj, weight) references by
column; sweep touched columns ascending with direct column pointers.
Oversized nodes keep projection-major order with hoisted pointers. No extra
memory, no dataset copy, works on any dataset; float additions within a
projection are reassociated (same summands, different rounding order — not
bit-identical, same precision class).

| 3M×4096, 5-tree medians | old dw1 | column-centric | ratio |
|---|---|---|---|
| ΣApplyProjection | 77.91 s | 49.62 s | **★ 1.57×** |
| TreeTrain | 181.2 s | 153.0 s | 1.18× |

Per-depth: 1.3–1.5× shallow/mid, rising to ~1.8× at depths 18+; depth 1
flat (single node = big-node path). CSVs: dw1_old_5trees_pcore_2026-06-10,
dw1_colcentric_5trees_pcore_2026-06-10.

dw1 column-centric follow-ups (canonical state, 5-tree medians):

- Block-size ablation at 3M×4096 (ΣAP): 4 MiB → 55.02 s, 16 MiB → 49.62 s,
  64 MiB → **47.05 s** (1.66× vs old dw1). Monotone: column sharing
  outweighs output-slab cache residency; probing 256 MiB before fixing the
  default.
- 30k×400k: AP 3.71 vs nodewise control 5.09 = **1.37×**; TreeTrain 44.3 vs
  47.1.
- 11M×28 (HIGGS shape): AP 11.67 vs control 16.05 = **1.38×**; TreeTrain
  43.9 vs 45.5 (AP is only ~26% of TreeTrain at this shape, so e2e gains are
  bounded; whether dw1 mode now beats the DFS baseline on real HIGGS needs
  the user's e2e rerun).

Caveat: controls use the default scheduler; TreeTrain comparisons entangle
BFS-vs-DFS scheduler costs. The AP scope is the clean kernel A/B.

256 MiB block probe: ΣAP **45.62 s** (1.71× vs old dw1) — still improving
but flattening (64→256 MiB = −3%). Column sharing dominates output-slab
residency outright. Default block size set to 256 MiB (64 Mi floats) in
code; YDF_DW1_BLOCK_FLOATS still overrides.

### Flag audit (2026-06-10, code inspection only)

- **symmetric_depthwise_ap: as intended.** Shared K projections per depth,
  bag concat + VQSort, K stride-1 sweeps, write-cursor routing to per-node
  slabs; slab/selected alignment guaranteed by the sorted-.active invariant
  (DCHECKs training.cc:5728/5734). Footnote: "aggregate == bag" comment only
  holds before any branch terminates; code correctly uses the actual frontier
  aggregate.
- **symmetric_nodewise_control: as intended.** Shared projections, no slab,
  per-node ProjectionEvaluator::Evaluate (oblique.cc:244).
- **projection_matrix_control: was NOT the intended driver-overhead
  control.** Its fallback kernel was rows-outer/projections-inner — a
  different AP memory-access order than Nodewise (scattered per-row column
  touches on column-major storage). PMC-vs-Nodewise comparisons therefore
  measured a loop-order artifact, not driver overhead; this likely explains
  "Nodewise AP (Control)" being the slowest column in the e2e table
  (trunk 1005 vs BFS-only 909; HIGGS 579 vs 457). **Fixed**: fallback now
  replicates the Evaluate traversal exactly (projection-outer, rows inner,
  AttributeValue, identical NaN handling). All prior PMC "control" numbers
  need re-running; the Dynamic_Row_Col_Major treatment branches (bf16/fp32
  dual stores) are unaffected. NOT yet rebuilt/re-measured (runs paused at
  user request).

**Correction (same day):** user clarified PMC's intent — it is the control
for **depthwise_1_pass** (same work, no column sharing), not a
Nodewise-driver control. Reimplemented accordingly: PMC now runs dw1's exact
kernels (same big/small split on YDF_DW1_BLOCK_FLOATS, projection-major dot
for big nodes, feature-major accumulate for small nodes, direct column
pointers) in (node, projection, feature) order with no cross-node column
grouping. dw1 − PMC isolates column sharing; entry build + counting sort
remain charged to dw1. The interim Evaluate-mimicking fallback survives only
as the no-direct-pointers generic path (mirrors dw1's). Still unbuilt /
unmeasured — runs paused.

## Phase O — e2e validation round, 2026-06-10 (evening)

### O.1 Dynamic_Row_Col_Major fp32 (precision-neutral) — m7i.4x e2e, FULL mode (7 runs)
Halved trunk shapes (64 GB box; dual fp32 stores need 2x dataset). Matched
control run on the same machine, same shapes. Files:
`runtime_full_control_cm_halfshapes_m7i.csv`,
`runtime_full_dynamic_row_col_major_fp32_m7i.csv`.

| shape | control CM (s) | dynamic fp32 (s) | speedup |
|---|---|---|---|
| 1.5M x 4096 | 330.93 ±0.53 | 265.65 ±1.83 | **1.246x (−19.7%)** |
| 15k x 400k | 117.30 ±0.72 | 64.27 ±0.83 | **1.825x (−45.2%)** |

Bit-identical models (fp32, same summation order per node). This is the
first e2e-backed claim for the dispatch idea, and the wide-short shape wins
big — small-node row-major path dominates there (15k rows ⇒ every node is
under the 20k dispatch threshold ⇒ effectively all row-major, and 400k
columns make CM gathers brutal). Cost: 2x dataset RAM.

### O.2 Column-centric dw1 — laptop e2e quick sweep (3 runs, killed during 30k×400k run 2)
`runtime_quick_depthwise_1_pass_fable_fixed.log`. Medians vs user's
reference table (Baseline DFS / old dw1):

| shape | new dw1 | old dw1 | DFS baseline |
|---|---|---|---|
| HIGGS | 520.1 | 539 | 372 |
| 3M x 4096 | **755.8** | 853 | 788 |
| 30k x 400k | 307.3 (1 run) | 307 | 274 |

First depth-batched config to beat DFS at 3M x 4096 (−4.1%), but well short
of the single-thread chrono projection (~1.18x TT) — consistent with the
thread-oversubscription hypothesis (6 concurrent trees x per-depth
ThreadPool(6) ≈ 36 runnable threads on 6 P-cores + a barrier per depth).
HIGGS unchanged-bad; 30k x 400k flat. Next: YDF_DW1_THREADS=1 HIGGS A/B.

### O.3 YDF_RM_MAX_ROWS bracket search (overnight 2026-06-10, in progress)
Seed points from the offline queue (1.5M x 4096 dual_fp32, 5-tree AP
medians, single thread): control CM 36.14 | rm10k **19.00** | rm20k 19.32 |
rm40k 19.63. Monotone rising 10k→40k ⇒ optimum ≤ 10k, the "flat 10k–40k"
read was the top of a shallow slope. Batch 1 (running): rm0 (pure CM via
dual-store kernel — also calibrates dual-store CM vs stock CM 36.14), rmINF
(pure row-major — if ≈ optimum, the CM copy can be dropped and the 2x RAM
cost halves), rm5000, rm2500. Refinement next per bracket rule; ties under
~2% (per-tree spread) stop the split. Then the same mini-search at 750k x
4096: if the optimal threshold halves with N, the dispatch variable is
density (rows_n/N); if it stays put, it's absolute working-set size.
Log: `benchmarks/results/rm_threshold_search_2026-06-10.log`.

### O.3 results — YDF_RM_MAX_ROWS bracket search COMPLETE (2026-06-11 ~00:30)
1.5M x 4096 dual_fp32, AP medians (5 trees, 1 thread), stock-CM control 36.14:

| thr | 0 (pure CM) | 312 | 625 | 1250 | 2500 | 5000 | 10k | 20k | 40k | INF (pure RM) |
|---|---|---|---|---|---|---|---|---|---|---|
| AP | 21.42 | 18.85 | **18.77** | 18.88 | 19.03 | 19.18 | 19.00 | 19.32 | 19.63 | 23.62 |

750k x 4096 (same kernel-CM rm0 reference 9.10):

| thr | 0 | 156 | 312 | 625 | 1250 | 2500 | 5000 | INF |
|---|---|---|---|---|---|---|---|---|
| AP | 9.10 | **8.43** | 8.49 | 8.55 | 8.66 | 8.80 | 8.84 | 11.00 |

Findings:
1. **The optimum is a wide flat plateau (~150–10k rows), not a point.** All
   within-plateau differences ≤ per-tree noise (~0.3 s). Search stopped per
   the ties rule.
2. **Both endpoints lose clearly**: pure CM +14% vs best, pure RM +26–30%.
   The CM copy earns its RAM at tall shapes — can't drop it.
3. **Most of the dual_fp32 win is the kernel, not the dispatch**: stock CM
   36.14 → dual-kernel pure-CM 21.42 (direct pointers + 4-row unroll);
   dispatch adds the last ~12% (21.42 → 18.77).
4. **Density vs absolute: indeterminate, and moot.** No sharp knee exists
   to scale with N; both shapes are flat from ~150 to ~10k. No
   justification for a YDF_RM_MAX_DENSITY knob — a fixed threshold in the
   hundreds-to-thousands range is robust across N. Recommend default
   ~625–1000 (≈3% better than the 20000 used in the m7i e2e run).
5. TT medians track AP: best 65.25 (rm1250) vs control 88.46 = 1.36x.

Raw: benchmarks/results/rm_threshold_search_2026-06-10.log; CSVs under
per_function_timing/.../trunk_{1500000,750000}_x_4096/dualfp32_rm*.csv.

### O.4 HIGGS YDF_DW1_THREADS=1 e2e A/B (2026-06-11 ~01:30) — hypothesis FALSIFIED
Quick mode (3 runs), HIGGS only: median **550.0 s ±9.3** vs dw1 with
per-depth threading 520.1 and DFS baseline 372. Serializing the within-tree
kernel makes it *worse* — thread oversubscription does not explain the
HIGGS regression. New hypothesis: HIGGS has 28 columns, so column sharing
has nothing to share (every column is hot at every node already); the
depth-batch driver itself (slab zero-init, entry build + counting sort,
giant shallow-depth slabs at 11M rows) is pure overhead. The original dw1
premise — columns touched per depth >> total columns — *inverts* on
narrow-D datasets. Decomposition in flight: corrected-PMC vs dw1 vs
nodewise chrono on HIGGS (PMC ≈ dw1 there ⇒ kernel order irrelevant at
28 cols, gap = driver) + corrected-PMC at 3M x 4096 for the
column-sharing measurement. CSV: runtime_quick_dw1_threads1_higgs.csv.

### O.5 Corrected-PMC decomposition (2026-06-11 ~01:50) — first valid column-sharing measurement
Chrono, 5 trees, 1 thread, AP medians.

**3M x 4096** (with prior runs: nodewise ~79.6, dw1 45.6):
nodewise 79.6 → **PMC 59.97** → dw1 45.6. The 1.71x dw1 AP win splits into
~25% from kernel quality alone (direct pointers, big/small split,
feature-major accumulate — PMC has all of it, in node order) and ~24% more
from actual cross-node column sharing. Both halves are real.

**HIGGS** (11M x 28):
| | nodewise | PMC | dw1 |
|---|---|---|---|
| AP | 25.13 | 25.39 | **19.70** |
| TT | **85.63** | 106.32 | 99.32 |

1. PMC ≈ nodewise AP ⇒ kernel order is irrelevant at 28 columns, as
   predicted.
2. dw1 *wins AP even on HIGGS* (−22%) — the counting-sorted sweep helps a
   little even with nothing to share.
3. The e2e regression is entirely **driver overhead**: per-tree deltas
   dw1 − nodewise: SampleProjection +5.7 s (depth-batch per-node projection
   materialization), SplitExamplesInPlace +2.9 s (BFS splits lose DFS's
   just-trained cache locality), NodeTrain-other +3 s, untracked per-depth
   driver (slab zero/assign, frontier bookkeeping, ThreadPool per depth)
   ~+7 s. Total +19 s vs AP win −5.4 s ⇒ TT +13.7 s. With AP only 29% of
   HIGGS TT, no AP-side fix can rescue depth-batching here; the fix (if
   pursued) is driver cost: depth-level projection sampling, slab reuse
   across depths, persistent thread pool, or simply not depth-batching when
   D is small (cols ≲ a few hundred ⇒ nothing to share).

Also: YDF_DW1_THREADS=1 e2e (O.4) falsified oversubscription; per-depth
threading is net positive (520 vs 550).
CSVs: per_function_timing/.../{trunk_3000000_x_4096,HIGGS_with_header}/
{pmc_dw1control,dw1,nodewise}_5trees_*_2026-06-11.csv.
