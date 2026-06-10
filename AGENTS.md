# AGENTS.md — Oblique Random Forest Performance Research

> **Scope**: oblique split methods only. Ignore other learners
> (gradient boosted trees, CART, etc.) unless explicitly asked.

The relevant files are:

- `yggdrasil_decision_forests/learner/decision_tree/oblique.cc` . Important functions: `EvaluateProjection`, `ProjectionEvaluator::Evaluate`, `FindBestConditionSparseObliqueTemplate`
- `yggdrasil_decision_forests/learner/decision_tree/training.cc` . Important functions: `FindSplitLabelClassificationFeatureNumericalHistogram`, `FindSplitLabelClassificationFeatureNumericalCart`
- `yggdrasil_decision_forests/learner/decision_tree/splitter_scanner.h` . Important function: `FindBestSplit`
- `/home/ariel/prog/ydf/yggdrasil-oblique-forests/yggdrasil_decision_forests/learner/random_forest/random_forest.cc` . Starts the whole forest training loop. Important function: `TrainWithStatusImpl`
- `benchmarks/results/Function Time by Depth - Exact HWY + Random - 3m HWY + Random.csv` - contains information about the runtime of each relevant function in the tree, by level. Highly useful to decide how to improve the algorithm.
- `examples/train_oblique_forest.cc` - the binary I call that decides RF vs SPO-RF, Exact vs. histogram, and many hyperparameters.

Do not pull in other files in context unless explicitly necessary. The whole math is in the above. Look at the includes of those files if you need e.g. VerticalDataset details


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
5. **Significance gate.** If median speedup over baseline on the end-to-end runtime.sh measures is **< 15%**, this experiment is a **failed
   experiment** for the purpose of step 6.
6. **Log result.** Append to the branch's `*_experiments.md` (see
   `benchmarks/experiments/apply_projection_experiments.md` for the
   shape). Every experiment — failed or successful — gets a row. **Mark
   successful experiments (≥ 20% speedup) prominently** (★, bold table,
   etc.) so they're easy to spot.
7. **Iterate.** Loop back to step 2. Call the Advisor model liberally.

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
Avoid speculative "why" stories about a result. Measure instead. Guesses are necessary when coming up with a hypothesis, but not when answering questions that can be answered empirically.

### Hardware + measurement rules
- Intel Core Ultra 9 185H is hybrid (P-cores + E-cores). E-cores must be
  off for stable timing — `parallel_chrono.py` and `runtime.sh`
  toggle this automatically; for standalone `perf` runs, manually:
  `sudo benchmarks/utils/set_cpu_e_features.sh --disable`. Restore
  with `--enable` when done. With E-cores on, `perf` splits counters
  across `cpu_atom/*` and `cpu_core/*` PMUs and many events become
  `<not supported>` — A/B comparisons become meaningless. However, leave E-cores on for bazel builds to speed them up 3x. Note that parallel_chrono.py automatically manages this for per-function timings.
- Sudo is available on request; just ask.
- Never run simultaneous experiments (timing noise).
- **Never use Intel Advisor's Memory Access Pattern or Microarchitecture
  Exploration profiles** — those use the `sep5` driver and freeze this system.
  Other Advisor profile types, including Hotspots, are allowed. `perf` is fine.


### Artifacts
- **Tracked**: the `.md` log, structured CSVs from `parallel_chrono.py`,
  small text excerpts of `perf annotate` inside the `.md`.
- **Not tracked**: `*.perfdata` (large binary), raw `.log` files
  (verbose, redundant). These live under
  `benchmarks/experiments/<branch-name>/perf_runs/` and
  `benchmarks/experiments/<branch-name>/iteration_logs/` — both
  gitignored. Regeneratable from the binary on demand.


## Build flags

Different versions of the code are hidden behind compile flags. See .bazelrc

## Conventions
- Data is **column-major** (`VerticalDataset`).
- Example indices: `UnsignedExampleIdx` (uint32).
- `SelectedExamplesRollingBuffer` is a non-owning span pair for in-place partitioning.
- Protobuf configs are the source of truth for hyperparameters.
- `// Ariel:` comments are the fork author's notes.
