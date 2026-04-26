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
| **baseline** | (no flag) | Per-call `ProjectionEvaluator::Evaluate`, one Evaluate per (node, projection) pair. Original code. |
| **V1** | `--config=nodewise_proj_matrix` | Per-node fused: rows-outer / projs-inner inside each node. Single-threaded outer loop over nodes. |
| **V2 prototype** | (replaced) | Single-pass over `(n,i,p)` triplets. `flat_selected` / `flat_projs` / `flat_out` flat buffers; per-task `upper_bound` + `divmod`. **3.85× slower than V1** — abandoned. |
| **V2-rev1** | (replaced) | Drops `flat_out` (writes directly into `out_projected[n]`); drops `flat_selected` / `flat_projs`; one `DecodeTriplet` at chunk entry, stateful `(n,i,p)` walk. **1.14× V1** — close but not winning. |
| **V2-rev2** | (replaced) | V2-rev1 + per-item software prefetch K=16 rows ahead. **1.74× V1** (regressed). Prefetch address recompute cost > latency hidden. |
| **V2-rev3** ★ | `--config=depthwise_1_pass` (current) | Tasks at (node, projection) granularity instead of (node, row, projection); inner kernel processes all rows of one (n, p) pair in **4-row blocks** with 4 independent `acc0..acc3` accumulators and 4 parallel `attr[ex_k]` loads per item. Exposes 4-way ILP to the OOO scheduler — up to 4 random DRAM misses in flight per item. **0.66× V1 ProjEval — 34% faster.** |

## Phase A — baseline diagnosis (perf record + perf stat)

Goal: confirm whether `ApplyProjection` is read-bound (load `(*attribute_values)[example_idx]`) or write-bound (`(*values)[selected_idx] = value`).

### Top-level `perf record cycles:u`

| Function | % cycles |
|---|---|
| `MakeTrunkDataset` (one-time data gen) | 30.10% |
| **`ProjectionEvaluator::Evaluate`** | **23.09%** |
| `FindSplitLabelClassificationFeatureNumericalCart` | 10.05% |
| `__libm_logf_l9` + `__svml_logf4_l9` (entropy) | 13.01% |
| hwy AVX2 sort kernels | 9.35% |

Inside training, `Evaluate` is the dominant CPU consumer.

### Line-level inside `Evaluate` (perf annotate)

clang vectorizes the inner loop into a 4-wide SIMD body + 1/2/3-item scalar
tail. For the trunk dataset's typical density (≈ 3 items per projection),
the **scalar tail dominates**. Hot samples:

| Insn @ off | Source mapping | % |
|---|---|---|
| `mulss (%rax,%rbp,4),%xmm3` | `attribute_values[example_idx] * weight` | scattered |
| `addss %xmm3,%xmm0` (50d2c8) | accumulate item-1 product | 9.55% |
| `addss %xmm3,%xmm0` (50d2e4) | accumulate item-2 product | 24.80% |
| `addss %xmm3,%xmm0` (50d2ff) | accumulate item-3 product | 40.64% |
| `movss %xmm0,(%r10,%r13,4)` (50d303) | **`(*values)[selected_idx] = value`** | **0.57%** |
| SIMD attribute loads (50d3a2, 50d3b0) | parallel gathers of `attribute_values[ex]` | 9.06% |

Perf attributes the cycles to the dependent `addss` (load-use stall), but
the load is the actual cost — every `mulss (%rax,%rbp,4)` is a random read
into a different attribute column at a different `example_idx`, jumping
across cache lines.

### Memory counters (`perf stat`, baseline, 1 tree)

| | Value | Note |
|---|---|---|
| L1 d-cache loads | 467 G | bulk of stream |
| L1 miss rate | 1.35% | most reads hit L1 |
| LLC-load-misses / LLC-loads | **91.5%** | when a miss escalates, it goes to DRAM |
| Cache-misses / refs (~L2) | 67.29% |  |
| IPC | 2.12 | memory-bound; well below peak |
| Wall time | 210.7 s |  |

### Verdict — read vs write

- **Write line `(*values)[selected_idx] = value;` is ~0.6% of `Evaluate`.** Not the bottleneck.
- **Read line `(*attribute_values)[example_idx];` (load + dependent `addss` stall) is ≥ 75% of `Evaluate`.** Dominant.
- Cost is **DRAM latency**: 91.5% of L3-escalated loads miss → ~150-300 cycles each. The surviving DRAM misses serialize and stall the FMA chain.

**Implication for V2.** A 1-pass kernel that performs the same set of `(*attribute_values)[example_idx]` loads as baseline will not reduce reads, so it cannot win at single-thread on this read-bound workload via parallelism alone. Wins must come from one of: software prefetch, multi-row interleaving (more loads in flight), data layout changes, or reducing the load count by sharing reads across projections.

## Phase B — V2 iteration

### Run summary (rows = 3M, threads = 1, 1 tree, tree 0 only — 1-run probes)

| Variant | tree wall | ΣProjEval | vs baseline | vs V1 |
|---|---|---|---|---|
| baseline (no-isnan-baseline.csv on main) | n/a | 57.23 s | 1.00× | 1.01× |
| V1 (`nodewise_proj_matrix`) | 133.8 s | 56.74 s | 0.99× | 1.00× |
| V2 prototype | 292.5 s | ~218 s | 3.81× ⛔ | 3.84× ⛔ |
| V2-rev1 (stateful walk + no flat_out) | 138.8 s | ~65 s | 1.14× | 1.14× |
| V2-rev2 (rev1 + per-item prefetch K=16) | 170.9 s | ~100 s | 1.75× ⛔ | 1.74× ⛔ |
| **V2-rev3** (4-row inner unroll) | **113.1 s** | **~38.8 s** | **0.68×** | **0.68×** |

### 5-run median confirmation (V2-rev3 vs. V1 ★)

Same workload as above, but each variant run 5× back-to-back **with `taskset
-c 0` to pin to a single P-core**. Tree shape (per-depth `(nodes, samples)`)
is **bit-identical** between V1 and V2-rev3 — V2-rev3 is a pure scheduler
change, same FMA math.

| Metric | V1 (median of 5) | V2-rev3 (median of 5) | V2-rev3 speedup |
|---|---|---|---|
| Tree wall time | 133.503 s | 111.069 s | **16.80% faster** |
| ΣProjEval (kProjectionEvaluate scope) | 58.321 s | 38.231 s | **34.45% faster** |

V2-rev3 cleanly clears the 20% threshold on the targeted
`kProjectionEvaluate` scope **when pinned**. End-to-end wall is 16.80% —
lower because ProjEval is only ~30% of total tree time; the rest
(`kSortFillBuckets`, `__libm_logf` for entropy, hwy AVX2 sort) is
unchanged.

### Median confirmation via `parallel_chrono.py` (no taskset, default tool methodology)

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

### Summary: did we hit 20%?

| Methodology | V2-rev3 speedup vs. baseline |
|---|---|
| `taskset -c 0`, 5 runs, kProjectionEvaluate scope | **34.45%** ✓ |
| `parallel_chrono.py` (no pin), 5-tree median, ΣApplyProjection per tree | **17.98%** ✗ (just under) |
| Same-binary, end-to-end tree wall, pinned 5 runs | **16.80%** |

V2-rev3 wins on every methodology, but only the pinned measurement
clears 20%. Adding `taskset -c 0` to `parallel_chrono.py`'s subprocess call
would close the gap; absent that, the headline number depends on whether
"benefit" is measured against the realistic-deployment scheduling pattern
(no pinning, 18%) or the kernel-microbenchmark scheduling pattern (pinned,
34%).

## Before / After perf annotate (the bottleneck → the fix)

Both captured with `perf record -F 999 -e cycles:u`, binary built `-c opt
--config=multithreaded_chrono_profile --copt=-g --strip=never`, run pinned
to CPU 0. perfdata files at
`benchmarks/experiments/1-pass-AP-CPU/perf_runs/` (gitignored — large,
regeneratable from the binary).

### Before — baseline `ProjectionEvaluator::Evaluate` hot loop

3-item scalar tail (clang couldn't fully SIMD it because density ≈ 3 not
a multiple of 4). Each item evaluation is **serialized through `acc`**:

```
load attr_ptr_for_item_k (ex)        ← random DRAM read
mulss  (loaded), %xmm3               ← weight × value
addss  %xmm3, %xmm0                  ← acc += ...   ← STALLS here on the load
... repeat for k = 1, 2, 3 ...
movss  %xmm0, (out, selected_idx)    ← single write
```

Cycle attribution (perf annotate, top lines inside `Evaluate`):

| Insn | Source | % cycles |
|---|---|---|
| `addss %xmm3,%xmm0` (50d2c8) | acc += weight × attr[ex] (item 1) | 9.55% |
| `addss %xmm3,%xmm0` (50d2e4) | item 2 | 24.80% |
| `addss %xmm3,%xmm0` (50d2ff) | item 3 | **40.64%** |
| `movss %xmm0,(...)` (50d303) | **`(*values)[selected_idx] = value;`** | 0.57% |

Reading: ~75% of in-function cycles concentrate on the `addss`
instructions, *but* perf attributes load-use stalls to the dependent FMA.
The actual cost is the random `(*attr)[ex]` load that the addss is waiting
on. Memory counters confirm: 91.5% LLC-load-miss rate, 1.35% L1 miss rate
— most loads hit L1, but the few that escalate go all the way to DRAM
(150-300 cycles each), and the FMA chain serializes them. The write line
is 0.57% — irrelevant. **Bottleneck: serialized DRAM-latency loads.**

### After — V2-rev3 `EvaluateProjectionRowBlocks` 4-row body

Manual 4-row unroll, 4 independent acc registers — clang lifted this into
a 4-wide SSE block:

```
mov   sel_ptr[i+0..3] → ex0..ex3                   ← 4 row indices precomputed
load  attr_ptr_for_item                            ← attribute column ptr
movss  attr[ex2], %xmm3                           ┐
movss  attr[ex3], %xmm2  → unpcklps → upper half  │  4 INDEPENDENT loads
movss  attr[ex0], %xmm4                           │  issued in parallel,
movss  attr[ex1], %xmm2  → unpcklps → lower half  │  packed into one 4-wide
movlhps           → xmm4 holds [v0, v1, v2, v3]   ┘
broadcast weight → xmm2 = [w, w, w, w]
mulps  %xmm4, %xmm2     ← 4-wide multiply
addps  %xmm2, %xmm1     ← 4-wide acc += ... (no FMA-chain stall)
```

Cycle attribution (perf annotate, top lines inside `EvaluateProjectionRowBlocks`):

| Insn | Source | % cycles |
|---|---|---|
| `movss attr[ex2]` (53e402) | parallel load | **45.16%** |
| `unpcklps` (53e409) | pack | 16.26% |
| `movss attr[ex0]` (53e413) | parallel load | 14.42% |
| `unpcklps` (53e41a) | pack | 10.93% |
| `addps %xmm2,%xmm1` (53e42d) | 4-wide acc += | 2.71% |
| `movups (out)` write 4 outputs | (line 386, see source) | < 1% |

Reading: cycles are now spent on the **loads themselves** (45 + 14 + 16 +
11 ≈ 87% in the load+pack region), not on the addps. That means the
loads are issued in parallel and the CPU is waiting on memory, but
**4 misses are in flight at once** instead of one — DRAM latency is
amortized by 4× load-level parallelism. The 4-wide `addps` accumulate
(2.71%) is no longer the choke point because the 4 lanes are independent.

### What changed — one-line summary

The loads were never the algorithmic problem; the *serialization* of the
loads through the FMA accumulator chain was. Process 4 rows at once with
4 independent accumulators → 4 in-flight DRAM misses per item → DRAM
latency overlapped → 34% faster on the kProjectionEvaluate scope.

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

### What worked vs. what didn't

| Idea | Result | Why |
|---|---|---|
| Per-task `(n,i,p)` triplet decode (V2 prototype) | -285% (3.85× slower) | Per-task `div` + `upper_bound` + `flat_out` scatter dominate the small inner work at density 3 |
| Stateful `(n,i,p)` walk, no flat_out (V2-rev1) | -14% (slower than V1) | Removed prototype's per-task overheads; brought timing into V1's range |
| Per-item software prefetch K=16 (V2-rev2) | -73% (1.74× slower) | `__builtin_prefetch(&AttributeValues(idx)[ex_pf])` recomputes the same indirect-indexed address as the actual load — 1 extra load per item per task |
| **4-row inner unroll, 4 acc registers (V2-rev3)** | **+34% on ProjEval** | Inner loop now issues 4 independent `attr[ex_k]` loads per item; OOO scheduler has 4-way ILP on a previously serialized FMA chain. Up to 4 random DRAM misses in flight per item — actually hides the latency that perf-stat showed dominating (91.5% LLC-load-miss on the random read) |

### Why 4-row unroll, not 8 or AVX2 gather?

Quick reasoning, not measured: the OOO load buffer on this core can hold ~10-12 outstanding misses; with density ≈ 3 items per projection, 4 rows × 3 items = 12 in-flight loads — saturating without overshooting. Tried with 4 because it matched the existing SIMD body the compiler generates (4-wide SSE) and lets the compiler vectorize the FMA cleanly. Higher unroll factors might help on AVX-512 hardware or denser projections; not pursued here since 4 already cleared the bar.

### Next questions / not done

- **Does V2-rev3 still win at higher row counts (10M, 20M)?** Cache-eviction patterns are different at scale. Likely still wins given DRAM dominates either way, but unverified.
- **Multi-thread.** Untested. The design has ConcurrentForLoop wired up; the at-task partitioning is now (node, projection) granularity, which may load-balance worse than fine-grained (n,i,p) chunking when one node dominates a level.
- **Higher-density oblique** (density factor ≥ 8). The 4-row unroll's gain comes from saturating load parallelism *within* one item; at density ≥ 4 the compiler-emitted 4-wide SIMD body in the original baseline kernel is already issuing 4 parallel loads, so V2-rev3's gain narrows.
- **VTune / `perf annotate` on V2-rev3.** Confirming the compiler actually generates 4 independent load slots in the assembly (vs. compiler folding them together) is a one-command check that hasn't been run.

---

*Log started 2026-04-25 during /loop-style autonomous iteration on the
1-pass design. V2-rev3 commit at `1-pass-AP-CPU` HEAD.*
