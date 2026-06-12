# Memory-safe Dynamic Row-Col replacement: experiments log

Branch: `auto-fable`. Hardware: AWS m7i (Xeon Platinum 8488C, 8 vCPU,
61 GiB RAM). Stable machine → **1 run per config during exploration, 3×
only to confirm significant wins.**

## Problem

`Dynamic_Row_Col_Major` (dual fp32 stores + per-node row/col dispatch) is
the best variant wherever it fits (HIGGS 266 s vs 282 stock; trunk
1.5m×4096: 241 vs 331) but **OOMs on every full-size dataset**: it keeps
TWO full fp32 copies (row-major + flat col-major). 3m×4096 / 300k×40k /
30k×400k are each ~46-48 GiB per copy → 2 copies > 61 GiB RAM. The
algorithm is fine; the memory model is what's broken.

## Diagnosis notes (what the dual store actually buys)

Per-depth chrono (laptop CSVs, 1.5m×4096, `control_cm` vs `dualfp32_rm10k`):

- ApplyProjection improves at **every** depth, not just deep/small nodes
  (depth 1-8 nodes are ≥600k rows): flat slab + kernel effects, not only
  the row store.
- The BFS dual run is PMC-driven (`PmcSweep`): projections pre-sampled,
  fused sweep. `FindObliqueSetup` 34.4 s → 9.0 s and `SampleProjection`
  17.5 → 0 (moved/amortized): per-node setup waste is a real chunk.
- Sparse-oblique sampling math, trunk 4096: P = ⌈4096^0.5⌉ = 64
  projections × density 1.5 ≈ 96 feature-uses per node over ~92 distinct
  features — **uses ≈ distinct**, so (a) cross-projection row reuse ≈ 1,
  (b) a node shares only ~2 % of its features with any descendant.
  Consequence: lazy gathering/caching of feature columns can never
  amortize on wide-feature data; the dual row store wins because its
  transpose is prepaid once for the whole forest.

## Experiments

| # | Variant | Build | trunk 1.5m×4096 probe (8 trees, 8 thr, DynRandHist t=1350) | RSS | Verdict |
|---|---------|-------|------------------------------------------------------------|-----|---------|
| 0 | stock (plain VD col-major, DFS) | (none) | **84.0 s** | 24.8 GB | baseline |
| 1 | Subtree gather cache: per-subtree compact gathered feature columns for nodes ≤ YDF_RM_MAX_ROWS, epoch-tagged slot map, budget-capped | `--config=subtree_gather` | 120.2 s (**+43 % — failed**) | 25.7 GB | ⛔ Gather overdraws ~4×: block gathers `distinct_features × block_rows` but the subtree only consumes ~¼ of it (2 % feature overlap between nodes). Kept in-tree for reference; do not pursue on wide-feature data. |
| 2 | 4-row-block Evaluate kernel (V2-rev3 inner loop per (node, projection)): 4 independent accumulators → 4 concurrent DRAM gathers per item | `--config=evaluate_4row` | 77.1 s (**−8.2 %**) | 24.8 GB | Below 15 % gate alone, bit-identical, zero memory. Composes. |
| 3 | 8-row unroll variant of #2 | `--config=evaluate_4row --cxxopt=-DEVALUATE_ROW_BLOCK_8` | 77.7 s | — | No gain over 4-row (MLP saturated). Keep 4. |
| 4 | #2 + per-thread cached ProjectionEvaluator (stock rebuilds the 4096-entry pointer tables per node) | `--config=evaluate_4row --config=cache_projection_evaluator` | **72.5 s (−13.7 %)** | 24.8 GB | Bit-identical (OOB acc/logloss match stock to all printed digits, incl. YDF_RM_MAX_ROWS ∈ {50, 200000} and Exact splits). Verdict run below. |

All variants verified bit-identical to stock on trunk 50k×256 OOB
(accuracy 0.853706, logloss 1.80006, seed 7).

## Verdict (runtime.sh, AVX2 vectorized section, 40 trees, t=250, 1 run)

Build: `EXTRA_BAZEL_CONFIGS="--config=evaluate_4row --config=cache_projection_evaluator"`

PENDING — run `runtime_1runs_4row_cache_m7i.csv` in flight.

Reference points (user's table, same harness):

| Dataset | stock DFS col | dyn DFS T=5000 (dual, OOM-prone) |
|---|---|---|
| HIGGS 11m×29 | 282 | 266 |
| trunk 3m×4096 | 688 | oom |
| trunk 1.5m×4096 | 331 | 241 |
| trunk 150k×40k | — | 110 |
| trunk 10k×400k | 78 | 47 |

## Next ideas

- Full-size run (3m×4096, 300k×40k, 30k×400k) — these are free wins vs
  the dyn column (oom) if the build holds its probe margin.
- The remaining dyn gap at 1.5m is deep-node row-locality; memory-safe
  candidates: bf16 dual (23+23 GiB fits 61 GiB, accuracy-changing —
  needs the accuracy gate), or hierarchical re-blocking (complex, gather
  economics still unproven).
- `dense_example_idxs` iota + `selected_labels` extraction per node are
  smaller per-node setup items if more setup shaving is needed.
