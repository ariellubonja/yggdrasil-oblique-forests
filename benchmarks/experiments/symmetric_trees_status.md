# Symmetric Trees (CatBoost-style) in Sparse Oblique RF — Status

**Branch:** `symmetric-trees` (off `rebased-main`).
**Session:** 2026-05-30. Author: Ariel + Claude.

## Goal

Implement CatBoost's oblivious-tree idea inside Sparse Oblique Random
Forests: **K candidate projections are drawn once per depth and shared
across all 2^d nodes** at depth d. The aggregate of all nodes' selected
examples at depth d == the tree's bag, so the per-projection column reads
become stride-1 sequential (vs. per-node scattered gathers that waste
cache lines as 2^d grows).

Expected win: per-cache-line useful-float ratio rises from ~16/2^d on
baseline (very wasteful at deep depths) to bag_density on the symmetric
path. Effect scales with tree depth.

## Design (frozen)

- **Scheduler.** New BFS depth-cohort scheduler `GrowTreeLocalBFS` next to
  the existing `GrowTreeLocal` (DFS). At each depth, collect all queue-front
  nodes whose `depth == current_depth` into `depth_batch`, process the
  cohort, then advance.
- **Projection sampling.** ONCE per depth, just before the per-cohort work,
  draw K projections (`shared_projections`). All nodes in `depth_batch`
  evaluate the same K. (User confirmed in interview Q1 = "b".)
- **AP kernel.** New
  `oblique_cpu_depthwise_symmetric_bagwide.{h,cc}` :
  `ApplyProjectionsDepthwiseSymmetricBagwide`. Algorithm:
  1. Concat per-node `selected_examples` into a flat `bag` + parallel
     `node_of_bag` array (per-node lists are individually sorted ascending).
  2. Stable-sort `bag` by `example_idx`, carrying `node_of_bag`. Within
     each (sorted) node group the relative order is preserved → cursor
     approach below gives the correct `pos_in_node`.
  3. For each projection k:
     - Hoist `(col_ptr, weight, na_value)` per item out of the i-loop.
     - Walk sorted `bag` stride-1. Per example: compute `value`, route
       via `write_cursor[node]++` into `out_projected[node][k*rows_n+pos]`.
- **Output layout** (decided by user): single bag-sized buffer per
  projection, logically partitioned into per-node contiguous segments
  (varying sizes; total = bag). Per-node slab packs K projections
  contiguously: `slab[k * rows_n + i]`.
- **Consumer plumbing.** `FindBestConditionSparseObliqueTemplate` in
  `oblique.cc` gets a branch (gated by
  `OBLIQUE_CPU_PRECOMPUTED_PROJECTIONS`) that skips `SampleProjection +
  Evaluate` when `precomputed_projected_values + depthwise_projection_defs
  + depthwise_monotonic` are all set, and reads from the slab instead.
- **Flags.**
  - `--config=symmetric_trees` → BFS scheduler only (`-DSYMMETRIC_TREES=1`).
    A/B-able by itself.
  - `--config=symmetric_bagwide` → BFS + kernel + consumer
    (`-DSYMMETRIC_TREES=1 -DDEPTHWISE_SYMMETRIC_BAGWIDE=1
    -DOBLIQUE_CPU_PRECOMPUTED_PROJECTIONS=1`). The full POC.

## What's been written (this session)

| File | Change |
|---|---|
| `.bazelrc` | Added `build:symmetric_trees` and `build:symmetric_bagwide`. |
| `yggdrasil_decision_forests/learner/decision_tree/training.h` | Added `GrowTreeLocalBFS` prototype. Added `precomputed_projected_values` field to `InternalTrainConfig`. |
| `yggdrasil_decision_forests/learner/decision_tree/training.cc` | `#include <deque>`. Changed `std::vector<NodeAndExamples>& node_stack` → `std::deque<...>&` in `NodeTrain`. Changed `GrowTreeLocal`'s local stack to deque (dropped reserve). Added `GrowTreeLocalBFS` function with depth_batch collection. Added `DEPTHWISE_SYMMETRIC_BAGWIDE` branch inside the BFS cohort loop that calls `ApplyProjectionsDepthwiseSymmetricBagwide` + plumbs per-node `node_config`. Gated the entry point in `DecisionTreeCoreTrain` on `SYMMETRIC_TREES`. |
| `yggdrasil_decision_forests/learner/decision_tree/oblique.cc` | Added a `has_precomputed_projected` branch in `FindBestConditionSparseObliqueTemplate` (gated by `OBLIQUE_CPU_PRECOMPUTED_PROJECTIONS`) that consumes the per-node slab. |
| `yggdrasil_decision_forests/learner/decision_tree/oblique_cpu_depthwise_symmetric_bagwide.h` | NEW: kernel declaration. |
| `yggdrasil_decision_forests/learner/decision_tree/oblique_cpu_depthwise_symmetric_bagwide.cc` | NEW: kernel implementation (gated by `DEPTHWISE_SYMMETRIC_BAGWIDE`). |
| `yggdrasil_decision_forests/learner/decision_tree/BUILD` | Added the new kernel `.cc/.h` to `training` cc_library. |

## Build status

- `bazel build -c opt --config=symmetric_trees //...:training` → **PASS** (BFS scaffold alone compiles, smoke-tested).
- `bazel build -c opt --config=symmetric_bagwide //...:training` → **IN PROGRESS** (background task `bmjts1css` at session checkpoint).

## Next steps (in order)

1. **Wait for symmetric_bagwide build to finish.** If errors, fix compile issues. Most likely failure modes:
   - `std::clamp` missing — already in `<algorithm>` but the file uses it directly; if header isn't transitively included, add `#include <algorithm>` to training.cc (probably already there).
   - Pointer/type mismatch on `depthwise_projection_defs` — should compile since `Projection` is `std::vector<AttributeAndWeight>`.
   - Anything else: inspect background task output at `/tmp/claude-1000/-home-ariel-prog-ydf-yggdrasil-oblique-forests/1957bec8-0343-4482-876f-2bcf99ad9287/tasks/bmjts1css.output`.

2. **Smoke-test:** train a small tree to ensure the symmetric path doesn't segfault. Need to confirm the trees actually grow (no infinite loop, no premature stop). One quick sanity run with `--num_trees=1` on a small dataset.

3. **Run `runtime.sh` quick** with `--config=symmetric_bagwide` and the suffix `corei0_symmetric_bagwide`. Comparison target: `/home/ariel/prog/ydf/yggdrasil-oblique-forests/benchmarks/results/runtime_full_baseline_corei0.csv`. Pre-bench:
   - `sudo benchmarks/utils/set_cpu_e_features.sh --disable`
   - `source /opt/intel/oneapi/setvars.sh`
   - Run runtime.sh and accuracy.sh sequentially (`&&`), never concurrently.
   - The benchmark output CSV will land in `benchmarks/results/runtime_quick_corei0_symmetric_bagwide.csv` (or similar — check runtime.sh's output convention).

4. **Compare** Sparse-Oblique columns vs baseline. Significance gate: ≥20% on `kProjectionEvaluate` chrono scope (per AGENTS.md). If accuracy of trees diverges (likely, since trees grow differently in BFS vs DFS), that's fine for POC; document the delta but don't gate on it.

5. **If the gate passes**: write up the result in `apply_projection_experiments.md` under a new "Phase L — symmetric bagwide" section, mirroring Phase G/I/J structure (Hypothesis / Implementation / Chrono A/B / Decision).

6. **If the gate fails** (≤20%): diagnose. The advisor flagged the failure mode — confirm column-access pattern with `perf stat -e LLC-load-misses` on baseline vs. variant (Phase G used the same metric). Open task #2 ("Thinking: Reconcile cache locality") is the place to capture the diagnostic.

## Open design questions (deferred)

- **Task #2 — Cache locality reconciliation.** Bag-sized buffer at N=1M is 4 MB (fits L3 but not L2). Per-node consume buffer fits L2. v3 column-streamed failed in part because column reuse alone wasn't enough; the symmetric path's bet is that bag-wide stride-1 access pattern fixes what v3 couldn't. Validation comes from runtime.sh + perf stat.
- **Symmetric-tree accuracy.** "Assume training to purity" means depth isn't capped, but BFS vs DFS changes RNG draw order → trees are NOT bit-identical to baseline. Time comparison is still meaningful (work is comparable); accuracy comparison needs separate eval_ab_e2e run.
- **Tile-and-route variant** (option d from interview): only relevant if scatter-write cache thrashing at deep depths hurts. Mitigation if runtime.sh shows weak speedup at deep depths only.

## Memory entries written

- `project_symmetric_depth_merged_projection.md` — the core idea.
- `project_bfs_frontier_already_landed.md` — note that `origin/1-node-ap-column-reuse` had the BFS scaffolding; ported manually to rebased-main on this branch.

## Prior art (do not re-litigate)

- **v3 `oblique_cpu_depthwise_column_streamed`** (`origin/1-node-ap-column-reuse @ cfb405b8`, "no faster"): column-deduplication across (n,p) consumers at BFS frontier. Inner loop still gathers `col_data[sel[i]]` per consumer. **Lesson**: column reuse alone doesn't beat baseline; need access-pattern change too.
- **Phase G "pregather"** (in `apply_projection_experiments.md`, +10.6%, failed 20% gate): per-node, U×R dense buffer = 1.14 GB at 3M×4096 >> 32 MB L3. **Lesson**: don't materialize a per-node dense buffer that exceeds L3.
- **V2-rev3** (landed on main, +20.1% AP): M=4 row-block unroll with 4 independent accumulators. Different mechanism (ILP for DRAM latency). The symmetric work is orthogonal and could compose.
