# HANDOFF — Oblique DW1 gather cache-locality investigation

This file is a self-contained context dump so a fresh Claude Code session
(running **on this EC2 box**) can continue without the prior laptop session.
Repo: `/home/ubuntu/yggdrasil-oblique-forests`, branch `rebased-main`.

## The overarching goal
Characterize and (eventually) fix the dominant cache cost in the
`DEPTHWISE_1_PASS` oblique projection kernel. The hot loop
(`yggdrasil_decision_forests/learner/decision_tree/oblique_cpu_depthwise_1pass.cc`,
the "Critical section" inner loop, ~line 291):

```cpp
for (size_t i = 0; i < rows_n; ++i) {        // Critical section
  float v = col[sel_ptr[i]];                 // line 291: the GATHER (scatter read)
  o[i] += w * v;                             // line 292: sequential RMW
}
```
- `col` = `evaluator.AttributeData(c)` — one feature column over ALL examples (column-major / vertical dataset), indexed by global example id.
- `sel_ptr` = `selected_examples_per_node[node].data()` — the node's example ids. **Always sorted ascending** (enforced by `DCHECK(std::is_sorted(...))` in `SplitExamplesInPlace`, training.cc:5758). `UnsignedExampleIdx` is uint32 in this build.
- `o` = per-node output projection slab `out_projected[node].data() + proj*rows_n`.

## Established findings (do not re-derive)
1. **The gather `col[sel_ptr[i]]` is THE bottleneck.** Earlier program-wide cachegrind: line 291 = 55% of ALL L1 read misses in the whole program (87% of the kernel's). `o[i]` writes hit L1 (0 write misses, RMW reuses the just-read line); `sel_ptr` is sequential and cheap.
2. **Sorting `sel_ptr` is futile.** It's already sorted (DCHECK). The misses come from **sparsity**, not disorder: a depth-d node owns a scattered ~n/2^d subset of rows, so even read in order, consecutive ids jump across 64B lines (16 floats/line). You can't fix sparsity by sorting.
3. **Correctness coupling** (if you ever permute `sel`): everything is positionally keyed to `selected_examples[i]` — `selected_labels`, `selected_weights`, `dense_example_idxs`, and the slab `o[i]` — and the consumer in `oblique.cc` (`has_precomputed_projected` path, ~line 248) reads `slab[proj*rows_n + i]` assuming position i ↔ example i. You'd have to permute the canonical array consistently AND you'd break the is_sorted invariant. Not worth it.
4. **Per-depth useful-floats-per-64B-line curve (cross-validated, trunk 100k×128, 1 tree, single-thread, max_num_projections=200):**

   | kernel level | useful/line |
   |---:|---:|
   | 1 | 8.19 |
   | 2 | 4.90 |
   | 3 | 3.37 |
   | 4 | 2.55 |
   | 5 | 2.13 |
   | 6 | 1.95 |
   | 7 | 1.91 |

   Steep decay levels 1→4, then **plateaus ~1.9 (NOT 1.0)**. Real splits cluster a node's rows so you always salvage ~1.9 of 16 floats; the uniform-random model (→1.0) is a worst-case bound. `DLmr=0` here because 51MB fits in the 109MB LLC — at production scale (e.g. HIGGS ~49GB) these same misses become DRAM misses (~10× cost/miss). Metric formula: `useful/line = (Dr_line291/2) / (D1mr_line291 − iters/16)`.

## Three measurement methods
- **A — exact distinct-cache-line counter (NOT yet built; recommended for deep trees).** Native run (full scale, multi-threaded, ~1× overhead). At the kernel call site accumulate per depth: `rows += rows_n; lines += count_distinct(sel[i] >> 4)`; `useful/line = rows/lines`. This *is* the exact metric. **Best choice for HIGGS depth-60 curve** — minutes, no Valgrind.
- **B — cachegrind, tree_depth diff (no recompile).** Run binary at `--tree_depth=2..N`, diff line-291 `Dr`/`D1mr` between consecutive runs. Kernel fires only when `depth_batch.size()>1`, so gathers start at the first 2-node level. **O(N) runs — does NOT scale to deep trees** (depth-60 ⇒ ~60 near-full trainings under cachegrind = infeasible). Easiest to set up though (no code change).
- **C — callgrind per-depth dumps (recompile).** `CALLGRIND_ZERO_STATS`/`CALLGRIND_DUMP_STATS_AT` bracket the kernel call, gated `#ifdef CALLGRIND_DEPTH`. **O(1) runs** — one callgrind run dumps every depth. Scales to deep trees, but it's one full single-thread training under callgrind (~50–100× native single-thread). Fiddlier: needs source edit, rebuild, and `callgrind_annotate` parsing (callgrind stores relative line positions).

B and C agreed to the last digit on the toy → both validated.

## Current state of THIS box (left by prior session)
- `training.cc` has guarded callgrind hooks: `#ifdef CALLGRIND_DEPTH` include block (~line 42) + ZERO/DUMP around the kernel call (~lines 5464/5471). No-op without the define. **Backup: `training.cc.bak_callgrind`.**
- **`bazel-bin/.../train_oblique_forest` is currently the INSTRUMENTED C build** (define applied to training.cc only via `--per_file_copt`). For normal benchmarking, rebuild without that flag.
- Artifacts: `/tmp/cgB.m{2..8}.out` (+ `.anno`), `/tmp/cgC/callgrind.out.<pid>.{1..7}`, parsers `/tmp/parseB.py` `/tmp/parseC.py`.

## Build recipe (icx + profiler + AVX2, Valgrind-compatible)
oneAPI is NOT auto-sourced; pin icx explicitly. `-march=native` emits AVX-512 which Valgrind 3.22 can't decode → cap at skylake (AVX2; access pattern is width-independent, so representative for cache misses).
```bash
export PATH="/opt/intel/oneapi/compiler/latest/bin:$PATH"
bazel build -c opt --config=profiler --config=depthwise_1_pass \
  --copt=-march=skylake \
  --repo_env=CC=/opt/intel/oneapi/compiler/latest/bin/icx \
  --repo_env=CXX=/opt/intel/oneapi/compiler/latest/bin/icpx \
  //examples:train_oblique_forest
# add  "--per_file_copt=decision_tree/training\.cc@-DCALLGRIND_DEPTH"  ONLY for the callgrind (C) build
```
Benchmark run command (single-thread for clean Valgrind sim):
```bash
./bazel-bin/examples/train_oblique_forest --input_mode=trunk --rows=100000 --cols=128 \
  --num_trees=1 --tree_depth=8 --num_threads=1 --feature_split_type=Oblique --max_num_projections=200
```
Notes: `--input_mode=trunk` (synthetic; `uniform` is commented out). `tree_depth` → `max_depth` (node at `depth>=max_depth` becomes leaf). Real benchmark harness: `benchmarks/profiling/parallel_chrono.py`; build helper `benchmarks/utils/utils.py` (does NOT add profiler/skylake — those were added manually).

## Next steps (in priority order)
1. **HIGGS depth curve via method A.** HIGGS grows to ~depth 60, majority of work at depths 25–35. B is infeasible at that depth; C is one long run; **A is the right tool** (native, fast, exact). Implement the distinct-line counter, run on HIGGS, plot useful/line vs depth. Question to answer: does the ~1.9 plateau hold, or does HIGGS sink toward 1.0?
2. Optionally one **C (callgrind) run at a single deep level (~depth 30)** if real L1/L2/L3/DRAM miss *costs* are wanted (at HIGGS scale the gather misses go to DRAM, not LLC).
3. Decide on the fix (layer-3 access-order): `row_major_dataset_layout`, gather-once-reuse-K-times, or the symmetric bagwide path — measure per-column reuse factor first (under sparse-oblique density it may be ~1, which would kill gather-once payoff).
4. **Cleanup:** revert `training.cc` from `training.cc.bak_callgrind` and rebuild a clean (non-instrumented) binary before any timing benchmarks.

## Persistent memory (carried over)
See `~/claude-memory/` (copied from the laptop). Most relevant:
- `dw1-gather-useful-per-line-by-depth.md` — the curve above + method.
- `dynamic-row-col-major-dispatch.md` — BFS/DFS paths, row-major layout, RM_MAX_ROWS.
- `ablation-ideas-tested-individually.md` — never stack optimization ideas; one --config per idea.
