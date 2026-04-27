# AGENTS.md — Oblique Random Forest Performance Research

> **Scope**: oblique split methods only. Ignore other learners
> (gradient boosted trees, CART, etc.) unless explicitly asked.

A research fork of [Yggdrasil Decision Forests](https://github.com/google/yggdrasil-decision-forests)
focused on **speeding up oblique (sparse) random forests**. Two concurrent workstreams:

1. **Reduce `ApplyProjection` time on CPU.** `ProjectionEvaluator::Evaluate`
   (`oblique.cc`) is the top CPU hotspot in oblique split finding.
   In flight: `--config=depthwise_1_pass` (V2-rev3, 4-row inner unroll
   exposing load-level parallelism — see
   `benchmarks/results/1pass_apply_projection_experiments.md`),
   `--config=nodewise_proj_matrix` (V1, per-node fused matrix fill,
   kept as A/B baseline for narrow datasets where projection sharing
   may matter), Highway-SIMD `upper_bound`
   (`--config=enable_std_upper_bound_avx2|avx512`), gated `std::isnan`
   (`YDF_BENCH_SKIP_ISNAN` default-on; `--config=with_isnan` to flip back).

2. **Offload tree tops to GPU.** Shallow levels have large rows / many
   projections — maps cleanly onto CUDA. `oblique_gpu.cc` +
   `oblique_gpu_kernels.cu.cc` expose `ApplyProjectionsNodewise`,
   `FindBestSplitNodewise`, `FindBestSplitDepthwise` (plus `*Exact`).
   Build: `--config=oblique_gpu`.

## Experiment workflow (mandatory pattern)

Empiricism is foundational here — if this loop isn't followed, the work
is wasted. Run sessions can be many hours; do not worry about token
budget. Run the loop **as long as possible, ideally indefinitely**.

### Branch convention
- Each experiment lives on its own branch named after the method/idea
  being optimized (e.g. `1-pass-AP-CPU`, `gpu-bfs-trunk-offload`,
  `simd-isnan-gate`).
- Never modify other branches from inside an experiment branch.
- `git rebase` to land on `main` only when results are confirmed.

### Per-experiment loop
1. **Baseline first — but reuse existing baselines when possible.** Check
   `benchmarks/results/per_function_timing/<CPU>/<projection_mode> | <split_type> | <numerical_split> | /<dataset>/`
   for an existing baseline CSV (e.g. `no-isnan-baseline.csv` for the
   default Oblique+Exact path on trunk 3M × 4096). If one matches the
   config you're varying against, **use it** — don't re-run. If you have
   to create a new baseline (different split type, different dataset
   shape, etc.), name it descriptively
   (e.g. `oblique_random_histogram_baseline.csv`) and **commit it**, so
   the next experiment can reuse it. Coverage grows monotonically; over
   time most experiments should not need to produce a fresh baseline.
   Methodology when you do produce one: `parallel_chrono.py` +
   `n_trees=5`, `rows=3000000`, `num_threads=1`, E-cores disabled.
2. **Diagnosis.** Profile the code using perf (don't use Intel advisor or vtune), or parallel_chrono.py to gain insights. If necessary, split big functions into smaller ones to diagnose bottlenecks. These can be computational (i.e. heavy mathematics in a certain function), or memory access pattern-related. If the user asks you to optimize a certain function, and it takes only a small amount of the overall runtime, push back, and continue trying to optimize the big chunk of the runtime.
3. **Hypothesis + change.** Implement one change. Keep the diff small.
4. **Measure.** Same methodology as baseline. **Median of 3 trees** via
   `--num_trees=3`, never 3 separate process invocations (cold-cache cost
   inflates each run).
5. **Significance gate.** If median speedup over baseline (on the
   targeted chrono scope) is **< 20%**, this experiment is a **failed
   experiment** for the purpose of step 6.
6. **Log result.** Append to the branch's `*_experiments.md` (see
   `benchmarks/results/1pass_apply_projection_experiments.md` for the
   shape). Every experiment — failed or successful — gets a row. **Mark
   successful experiments (≥ 20% speedup) prominently** (★, bold table,
   etc.) so they're easy to spot.
7. **5 consecutive failures → enter full plan mode.** Stop the loop, list
   what was tried and what was learned (cache pattern, FMA serialization,
   load-buffer occupancy, …), and design fundamentally different
   approaches before resuming. Do not just keep trying small variations.
8. **Iterate.** Loop back to step 2.

### Hardware + measurement rules
- Intel Core Ultra 9 185H is hybrid (P-cores + E-cores). E-cores must be
  off for stable timing — `parallel_chrono.py` and `e2e_runtime.sh`
  toggle this automatically; for standalone `perf` runs, manually:
  `sudo benchmarks/src/utils/set_cpu_e_features.sh --disable`. Restore
  with `--enable` when done. With E-cores on, `perf` splits counters
  across `cpu_atom/*` and `cpu_core/*` PMUs and many events become
  `<not supported>` — A/B comparisons become meaningless. However, leave E-cores on for bazel builds to speed them up 3x. Note that parallel_chrono.py automatically manages this for per-function timings.
- Sudo is available on request; just ask.
- Default workload: `rows=3000000`, `num_threads=1`. Never run
  simultaneous experiments (timing noise).
- **Never use sep5 / VTune / Advisor** — freezes this system. `perf` is fine.

### `perf` discipline
- `perf record -F 999 -e cycles:u` for top-down attribution. Build the
  binary with `--copt=-g --strip=never` so source-line annotation works.
- `perf annotate` to find the actual hot instructions inside the
  function. Watch for load-use stalls attributed to the dependent FMA
  (the load is the cost, not the addss).
- `perf stat -e cycles:u,instructions:u,cache-misses,LLC-load-misses,...`
  for memory-bandwidth vs. compute attribution.
- `taskset -c 0` for stable single-thread measurement when running the
  binary outside `parallel_chrono.py`. The script doesn't pin; numbers
  produced via the script will trail pinned numbers by a few percentage
  points and that's fine — the baseline-comparison methodology is what
  matters, not the absolute number.

### Artifacts
- **Tracked**: the `.md` log, structured CSVs from `parallel_chrono.py`,
  small text excerpts of `perf annotate` inside the `.md`.
- **Not tracked**: `*.perfdata` (large binary), raw `.log` files
  (verbose, redundant). These live under
  `benchmarks/experiments/<branch-name>/perf_runs/` and
  `benchmarks/experiments/<branch-name>/iteration_logs/` — both
  gitignored. Regeneratable from the binary on demand.

### After completing a successful experiment
- Remind the user to describe how they want times-per-depth plotted
  from `parallel_chrono.py` outputs.

## Build commands

```bash
bazel build -c opt //yggdrasil_decision_forests/learner/decision_tree:training        # Default opt build
bazel build -c opt --config=multithreaded_chrono_profile //...                        # CHRONO timing per tree/depth
bazel build -c opt --config=intel_profiler //...                                       # -O2 + DWARF, perf-annotate-friendly
bazel build --config=intel_debug //...                                                  # icx debug build
```

### Key configs (`.bazelrc`)
| Config | Purpose |
|--------|---------|
| `multithreaded_chrono_profile` | `CHRONO_ENABLED` — per-function timing by tree/depth |
| `nodewise_proj_matrix` | V1 fused Apply (per-node rows-outer/projs-inner matrix fill) |
| `depthwise_1_pass` | V2-rev3 fused Apply (4-row inner unroll, load-level parallelism) |
| `with_isnan` | Re-enables `std::isnan` check (off by default; `YDF_BENCH_SKIP_ISNAN` is on) |
| `enable_std_upper_bound_avx2` / `avx512` | Vectorized `upper_bound` |
| `print_projection_matrices` | Prints projection matrix at each node (debug) |

## Critical files

| File | Role |
|------|------|
| `learner/decision_tree/training.{cc,h}` | Tree growth (`GrowTreeLocalBFS`), node processing (`NodeTrain`), split dispatch |
| `learner/decision_tree/oblique.{cc,h}` | Oblique split finding — projection sampling, evaluation, scoring; `ProjectionEvaluator::Evaluate` is the baseline hot loop |
| `learner/decision_tree/oblique_cpu_nodewise_proj_matrix.{cc,h}` | V1 — gated by `NODEWISE_PROJ_MATRIX` |
| `learner/decision_tree/oblique_cpu_depthwise_1pass.{cc,h}` | V2-rev3 — gated by `DEPTHWISE_1_PASS` |
| `learner/decision_tree/label.h` | `InternalTrainConfig`, umbrella macro `OBLIQUE_CPU_PRECOMPUTED_PROJECTIONS` |
| `learner/random_forest/random_forest.cc` | RF training loop, bagging, OOB |
| `utils/parallel_chrono.h` | `CHRONO_SCOPE`, per-function timing enums |
| `model/decision_tree/decision_tree.h` | `NodeWithChildren`, `SelectedExamplesRollingBuffer` |
| `learner/decision_tree/decision_tree.proto` | Config proto |

Tree growth is BFS via a `std::deque<NodeAndExamples>` queue: `NodeTrain`
checks exit conditions, calls `FindBestCondition` → oblique splitter,
calls `SplitExamplesInPlace`, enqueues both children.

## Tests

```bash
bazel test -c opt //yggdrasil_decision_forests/learner/random_forest:random_forest_test --test_filter="*SparseOblique*"
bazel test -c opt //yggdrasil_decision_forests/learner/decision_tree:training_test --test_filter="*Oblique*"
```

## Benchmarks layout

```
benchmarks/
├── src/
│   ├── parallel_chrono.py          # Main driver: CHRONO build + per-tree-depth CSV
│   ├── bash_scripts/e2e_runtime.sh # End-to-end runtime suite
│   └── utils/
│       ├── utils.py                # Shared CLI parser, E-core toggle, build helpers
│       ├── make_trunk_dataset.py   # Synthetic trunk datasets
│       └── set_cpu_e_features.sh   # E-core/turbo toggle
├── data/                           # HIGGS, SUSY, epsilon, CC18 tasks, synthetic trunk
├── results/                        # Tracked: per-method CSVs + experiment .md logs
│   └── per_function_timing/<CPU>/<exp>/<dataset>/<depth>Depth-<threads>Threads.csv
└── experiments/<branch-name>/      # Gitignored: perf_runs/ + iteration_logs/
```

Common invocations:

```bash
python benchmarks/src/parallel_chrono.py \
  --input_mode=trunk --rows=3000000 --tree_depth=-1 --num_trees=5 \
  --num_threads=1 --feature_split_type=Oblique --numerical_split_type=Exact \
  --depthwise_1_pass

bash benchmarks/src/bash_scripts/e2e_runtime.sh
```

`parallel_chrono.py` outputs CSV columns: `thread, tree, depth, nodes,
samples, SampleProj, ApplyProjection, EvalProj, sort phases, histogram
phases`. Sweeps in `e2e_runtime.sh`: split types (Oblique / Axis
Aligned), numerical methods (Exact / Random / Dynamic Random Histogram),
vectorization (None / AVX2 / AVX512), datasets.

## Conventions
- Data is **column-major** (`VerticalDataset`).
- Example indices: `UnsignedExampleIdx` (uint32).
- `SelectedExamplesRollingBuffer` is a non-owning span pair for in-place partitioning.
- Protobuf configs are the source of truth for hyperparameters.
- `// Ariel:` comments are the fork author's notes.
