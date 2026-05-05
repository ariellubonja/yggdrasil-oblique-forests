# AGENTS.md — Oblique Random Forest Performance Research

> **Scope**: oblique split methods only. Ignore other learners
> (gradient boosted trees, CART, etc.) unless explicitly asked.

The relevant files are:

- /home/ariel/prog/ydf/yggdrasil-oblique-forests/yggdrasil_decision_forests/learner/decision_tree/oblique.cc . Important functions: EvaluateProjection, ProjectionEvaluator::Evaluate , FindBestConditionSparseObliqueTemplate
- /home/ariel/prog/ydf/yggdrasil-oblique-forests/yggdrasil_decision_forests/learner/decision_tree/training.cc . Important functions: FindSplitLabelClassificationFeatureNumericalHistogram, FindSplitLabelClassificationFeatureNumericalCart
- /home/ariel/prog/ydf/yggdrasil-oblique-forests/yggdrasil_decision_forests/learner/decision_tree/splitter_scanner.h . Important function: FindBestSplit
- /home/ariel/prog/ydf/yggdrasil-oblique-forests/yggdrasil_decision_forests/learner/random_forest/random_forest.cc . Starts the whole forest training loop. Important function: TrainWithStatusImpl
- /home/ariel/prog/ydf/yggdrasil-oblique-forests/benchmarks/results/Function Time by Depth - Exact HWY + Random - 3m HWY + Random.csv - contains information about the runtime of each relevant function in the tree, by level. Highly useful to decide how to improve the algorithm.
- /home/ariel/prog/ydf/yggdrasil-oblique-forests/examples/train_oblique_forest.cc - the binary I call that decides RF vs SPO-RF, Exact vs. histogram, and many hyperparameters.

Do not pull in other files in context unless explicitly necessary. The whole math is in the above. Look at the includes of those files if you need e.g. VerticalDataset details

Goal:

1. **Reduce `ProjectionEvaluator::Evaluate` time (colloquially called `ApplyProjection`) .** 


**Crucial: All contributions need to be research grade - therefore only publishable speedup methods count. E.g. achieving speedup by turning off logging is unacceptable**

## Experiment workflow (mandatory pattern)

Empiricism is foundational here — if this loop isn't followed, the work is wasted.

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
   `benchmarks/experiments/apply_projection_experiments.md` for the
   shape). Every experiment — failed or successful — gets a row. **Mark
   successful experiments (≥ 20% speedup) prominently** (★, bold table,
   etc.) so they're easy to spot.
7. **5 consecutive failures → enter full plan mode.** Stop the loop, list
   what was tried and what was learned (cache pattern, FMA serialization,
   load-buffer occupancy, …), and design fundamentally different
   approaches before resuming. Do not just keep trying small variations.
8. **Iterate.** Loop back to step 2.

### A/B evaluation
Two tools, two jobs:

- **Verdict (end-of-experiment, deployable build):**
  `benchmarks/evaluation/runtime.sh` - end-to-end runtime on various datasets. Use its default Quick mode initially, then the full mode when an idea's benefits seem solid, to confirm. `benchmarks/evaluation/accuracy.sh` measures accuracy. This should match exactly. Baselines are named appropriately under `/home/ariel/prog/ydf/yggdrasil-oblique-forests/benchmarks/results` depending on the machine being used.
- **Insight (during the iteration loop, per-depth signal):** run
  `benchmarks/profiling/parallel_chrono.py` directly with `--bazel_config=NAME`
  for both A and B. gives you per-tree-depth
  ΣApplyProj / SortFillBuckets / etc., which is what you need to *understand*
  why a change moves a number.

### On guessing vs. measuring
Avoid speculative "why" stories about a result. A small autonomous-mode
guess is fine to keep moving, but if a result doesn't match the
hypothesis — or lands inside the noise floor — stop guessing and dig
deeper instead of trying another variation. Concretely: add `taskset`
pinning, run repeats with σ, take a per-depth breakdown, capture
`perf stat` (LLC misses, IPC, stall counters) and `perf annotate`
deltas between A and B, or strip the kernel into a microbench. Evidence
beats narrative.

### Hardware + measurement rules
- Intel Core Ultra 9 185H is hybrid (P-cores + E-cores). E-cores must be
  off for stable timing — `parallel_chrono.py` and `runtime.sh`
  toggle this automatically; for standalone `perf` runs, manually:
  `sudo benchmarks/utils/set_cpu_e_features.sh --disable`. Restore
  with `--enable` when done. With E-cores on, `perf` splits counters
  across `cpu_atom/*` and `cpu_core/*` PMUs and many events become
  `<not supported>` — A/B comparisons become meaningless. However, leave E-cores on for bazel builds to speed them up 3x. Note that parallel_chrono.py automatically manages this for per-function timings.
- For per-function timing runs, invoke `parallel_chrono.py` directly — it handles the bazel build (pass extra `--config=…` via `--bazel_config=NAME`) and the E-core toggle for you; don't run `bazel build` or `set_cpu_e_features.sh` separately.
- Sudo is available on request; just ask.
- Default workload: `rows=3000000`, `num_threads=1`. Never run
  simultaneous experiments (timing noise).
- **Never use sep5 / VTune / Advisor** — freezes this system. `perf` is fine.


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
| `enable_std_upper_bound_avx2` / `avx512` | Vectorized `upper_bound` |


## Conventions
- Data is **column-major** (`VerticalDataset`).
- Example indices: `UnsignedExampleIdx` (uint32).
- `SelectedExamplesRollingBuffer` is a non-owning span pair for in-place partitioning.
- Protobuf configs are the source of truth for hyperparameters.
- `// Ariel:` comments are the fork author's notes.
