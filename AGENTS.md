# AGENTS.md — Oblique Random Forest Performance Research

> **Scope**: This file covers only the oblique split methods in this YDF fork.
> Ignore all other learners (gradient boosted trees, CART, etc.) unless explicitly asked.

## What This Fork Is

A research fork of [Yggdrasil Decision Forests](https://github.com/google/yggdrasil-decision-forests) focused on **speeding up oblique (sparse) random forests**. The upstream repo supports many learners; this fork modifies only the oblique RF training path.

## Current Research Goals

Two concurrent workstreams:

1. **Reduce `ApplyProjection` time on CPU.** `ProjectionEvaluator::Evaluate`
   (`oblique.cc`) takes ~12% of multicore runtime on 4096-column synthetic data
   and is the top CPU hotspot in oblique split finding. Current experiments:
   a one-pass "fused" variant `ApplyProjectionsFusedLevel`
   (`oblique_cpu_depthwise.cc`) that sweeps each node's row range once and
   accumulates all P projections in the same pass, materializing a
   `P × rows_n` slab per node. Built via `--config=depthwise_cpu`; the
   baseline projection-wise path (single reused scratch buffer) is the default.
   Also in flight: Highway-SIMD sort for exact-split threshold scanning
   (`--config=use_std_sort` toggles back to `std::sort` for A/B timing), and
   a branchless/gated `std::isnan` path (`YDF_BENCH_SKIP_ISNAN`, default on
   via `.bazelrc`; flip back with `--config=with_isnan`).

2. **Offload entire tops of trees to GPU.** For the shallow levels of each
   tree, node row ranges are large, projection counts are high, and the work
   maps cleanly onto CUDA. `oblique_gpu.cc` + `oblique_gpu_kernels.cu.cc`
   expose `ApplyProjectionsNodewise`, `FindBestSplitNodewise`, and
   `FindBestSplitDepthwise` (plus `*Exact` variants) — each takes a whole
   level's (or node's) worth of work, does it on the GPU, returns the best
   split back to CPU. Builds with `--config=oblique_gpu` (requires CUDA).
   The CPU/GPU crossover depth is workload-dependent; once node row counts
   drop below the GPU's effective batch size, the CPU path takes over.

## Build System

Bazel. All commands run from the repo root.

```bash
# Optimized build (default via .bazelrc)
bazel build -c opt //yggdrasil_decision_forests/learner/decision_tree:training

# Debug build (Intel icx compiler)
bazel build --config=intel_debug //...

# Profiling build (VTune-friendly, -O2 + symbols)
bazel build -c opt --config=intel_profiler //...

# With custom CHRONO per-function timing
bazel build -c opt --config=multithreaded_chrono_profile //...

# Print oblique projection matrices (debug)
bazel build -c opt --config=print_projection_matrices //...
```

## Key Bazel Configs (.bazelrc)

| Config | Purpose |
|--------|---------|
| `multithreaded_chrono_profile` | Enables `CHRONO_ENABLED` — per-function timing by tree/depth |
| `print_projection_matrices` | Enables `PRINT_PROJECTION_MATRICES` — prints splits at each node |
| `enable_std_upper_bound_avx2` | AVX2 vectorized upper_bound |
| `enable_std_upper_bound_avx512` | AVX512 vectorized upper_bound |
| `disable_empty_projections` | Disallows zero-weight projections |

## Critical Files (Oblique Training Path)

### Decision Tree Core
| File | Role |
|------|------|
| `yggdrasil_decision_forests/learner/decision_tree/training.cc` | Tree growth loop (`GrowTreeLocalBFS`), node processing (`NodeTrain`), split finding dispatch |
| `yggdrasil_decision_forests/learner/decision_tree/training.h` | Public API, `NodeAndExamples` struct, `PerThreadCache`, `InternalTrainConfig` |
| `yggdrasil_decision_forests/learner/decision_tree/oblique.cc` | Oblique split finding — projection sampling, evaluation, scoring |
| `yggdrasil_decision_forests/learner/decision_tree/oblique.h` | Oblique split API |

### Random Forest Learner
| File | Role |
|------|------|
| `yggdrasil_decision_forests/learner/random_forest/random_forest.cc` | RF training loop — bagging, tree dispatch, OOB evaluation |

### Profiling Infrastructure
| File | Role |
|------|------|
| `yggdrasil_decision_forests/utils/parallel_chrono.h` | `CHRONO_SCOPE`, `TreeScope`, `DepthScope`, per-function timing enums |

### Model / Data Structures
| File | Role |
|------|------|
| `yggdrasil_decision_forests/model/decision_tree/decision_tree.h` | `NodeWithChildren`, `SelectedExamplesRollingBuffer`, tree structure |
| `yggdrasil_decision_forests/learner/decision_tree/decision_tree.proto` | Config proto: `sparse_oblique_split`, `mhld_oblique_split`, `growing_strategy`, `max_depth`, etc. |

## Tree Growth Architecture

The tree is grown **breadth-first** (BFS) via an iterative queue:

1. `DecisionTreeCoreTrain` → `GrowTreeLocalBFS` (for `kGrowingStrategyLocal`)
2. `GrowTreeLocalBFS` maintains a `std::deque<NodeAndExamples>` — push_back/pop_front = FIFO
3. Each iteration calls `NodeTrain` which:
   - Checks exit conditions (min_examples, max_depth, timeout)
   - Calls `FindBestCondition` → dispatches to oblique splitter
   - Calls `SplitExamplesInPlace` to partition examples
   - Enqueues both children onto the back of the queue

## Running Tests

```bash
# Oblique-specific tests
bazel test -c opt //yggdrasil_decision_forests/learner/random_forest:random_forest_test \
  --test_filter="*SparseOblique*"

bazel test -c opt //yggdrasil_decision_forests/learner/decision_tree:training_test \
  --test_filter="*Oblique*"

# Run binary directly for LOG output visibility
./bazel-bin/yggdrasil_decision_forests/learner/random_forest/random_forest_test \
  --gtest_filter="*SparseOblique*"
```

## Benchmarks (`benchmarks/`)

The `benchmarks/` directory is the experimental instrumentation for this research.

### Structure
```
benchmarks/
├── src/                            # Benchmark scripts and tooling
│   ├── bash_scripts/
│   │   ├── e2e_runtime.sh          # Main end-to-end runtime benchmark suite
│   │   ├── parallel_chrono_many_threads.sh  # CHRONO profiling across thread counts
│   │   └── exact_histogram_breakeven/       # Exact-vs-histogram breakeven experiments
│   │       ├── heatmaps/           # Row-scaling heatmap scripts (varying rows, threads)
│   │       └── chrono/             # CHRONO-enabled breakeven scripts
│   ├── parallel_chrono.py          # Python driver: runs CHRONO builds, parses per-tree-depth
│   │                               #   timing output into CSVs (thread-pivoted)
│   ├── plot_ydf_projection_matrix.ipynb  # Visualize oblique projection matrices
│   ├── microbenchmarks/            # Standalone C++ microbenchmarks (e.g. sampling algorithms)
│   └── utils/
│       ├── utils.py                # Shared CLI arg parser, E-core toggle, build helpers
│       ├── make_trunk_dataset.py   # Generate synthetic "trunk" datasets (rows x cols)
│       ├── set_cpu_e_features.sh   # Disable/enable Intel E-cores for stable benchmarking
│       └── download_cc18_datasets.ipynb  # Download OpenML CC18 benchmark suite
├── data/                           # Datasets
│   ├── cc18_binary_csv/            # 35 OpenML CC18 binary classification tasks (10-fold CV)
│   │   └── task_<id>_<name>/       # Each: repeat0_fold{0..9}_sample0_{train,test}.csv
│   ├── HIGGS_with_header.csv       # Large physics dataset (~11M rows)
│   ├── SUSY_with_header.csv        # Large physics dataset (~5M rows)
│   ├── epsilon_normalized_train.csv  # Large dense dataset (400K rows x 2000 features)
│   ├── haberman.csv                # Small dataset
│   └── jovo_50m/                   # Pickle-format dataset (50M?)
└── results/                        # Experimental output (committed for reference)
    ├── e2e_runtime/                # Wall-clock runtime logs (per method, dataset, ISA)
    ├── e2e_accuracy/               # OOB accuracy logs
    ├── per_function_timing/        # CHRONO CSVs organized by CPU
    │   ├── Intel(R) Core(TM) Ultra 9 185H/   # Laptop results
    │   │   └── <projection_mode> | Oblique | <split_method> | /
    │   │       └── <dataset>/<depth>Depth-<threads>Threads.csv
    │   └── Intel(R) Xeon(R) Platinum 8488C/   # AWS server results
    └── ydf_projection_matrices/    # Saved projection matrix visualizations
```

### Running Benchmarks

> **IMPORTANT — CPU E-features must be disabled for any timing.** Before running
> `perf stat`, `perf record`, `parallel_chrono.py`, `e2e_runtime.sh`, or any other
> benchmark/profiler command, you **must** run:
> ```bash
> sudo benchmarks/src/utils/set_cpu_e_features.sh --disable
> ```
> This is a hybrid Intel CPU (P-cores + E-cores). With E-cores / HT / turbo enabled,
> measurements are noisy and `perf` splits counters across `cpu_atom/*` and
> `cpu_core/*` PMUs (many events become `<not supported>` on the atom side), making
> A/B comparisons meaningless. The `parallel_chrono.py` driver and `e2e_runtime.sh`
> call this automatically (`disable_ecores=True` default); standalone `perf`
> invocations do not — run the script manually before profiling.

```bash
# Full end-to-end runtime suite (builds, disables E-cores, runs all methods/datasets)
bash benchmarks/src/bash_scripts/e2e_runtime.sh

# CHRONO per-function profiling (requires --config=multithreaded_chrono_profile build)
python benchmarks/src/parallel_chrono.py \
  --num_threads=48 --train_csv=benchmarks/data/trunk_data/100000x4096.csv \
  --label_col=target --tree_depth=-1 --num_trees=512

# Breakeven heatmaps (exact vs histogram across dataset sizes)
bash benchmarks/src/bash_scripts/exact_histogram_breakeven/heatmaps/oblique_multithr_-1_depth.sh

# Single run with the training binary
bazel build -c opt //examples:train_oblique_forest
./bazel-bin/examples/train_oblique_forest \
  --input_mode csv \
  --train_csv benchmarks/data/cc18_binary_csv/<dataset>/repeat0_fold0_sample0_train.csv \
  --label_col <label> \
  --feature_split_type "Oblique" \
  --num_trees 240 --num_threads -1
```

### Key Benchmark Parameters

The `e2e_runtime.sh` script sweeps over:
- **Split types**: Oblique, Axis Aligned
- **Numerical split methods**: Exact, Random, Dynamic Random Histogram
- **Vectorization**: None, AVX2, AVX512 (for Random/Dynamic methods)
- **Datasets**: HIGGS, SUSY, epsilon, CC18 tasks, synthetic trunk data
- **Dynamic split thresholds**: configurable sweep or fixed defaults

The `parallel_chrono.py` script:
- Builds with `--config=multithreaded_chrono_profile`
- Parses per-tree per-depth timing lines from stdout
- Outputs CSV with columns: thread, tree, depth, nodes, samples, then per-function times
- Functions timed: SampleProj, ProjEval, EvalProj, sort phases, histogram phases

## Conventions

- Data is **column-major** (`VerticalDataset`)
- Example indices use `UnsignedExampleIdx` (uint32)
- `SelectedExamplesRollingBuffer` is a non-owning span pair (active/inactive) for in-place partitioning
- Protobuf configs are the source of truth for hyperparameters
- Custom annotations in code: comments starting with `// Ariel:` are the fork author's notes
