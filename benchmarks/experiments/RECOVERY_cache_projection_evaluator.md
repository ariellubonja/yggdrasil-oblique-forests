# Handoff — re-measure `cache_projection_evaluator` end-to-end

**Purpose.** This experiment was written, measured once on m7i, then its branch
was deleted. The code has now been **recovered** onto a fresh branch. This doc
is the contract for an agent on an **m7i (or other real Linux benchmark box)**
to rebuild it, run the standard e2e protocol, and report numbers. **Mac numbers
do not count** (per project rule) — this was recovered/compiled on macOS only to
confirm it builds; all reported timings must come from the Linux benchmark box.

## Where the code is

- **Branch:** `cache-projection-evaluator` (created off `rebased-main`).
- **Recovery commit:** `a70be692` "Recover cache_projection_evaluator (from
  deleted branch tip 3b5a9c2e)".
- **Safety tag:** `recover/cache-proj-eval` → `3b5a9c2e` (the original deleted
  branch tip, in case anything needs cross-checking).
- This branch **already contains all of `rebased-main`** (it was branched from
  it), so there is nothing to merge — the recovered code sits on top of current
  `rebased-main`. The feature had been removed from `rebased-main` in commit
  `fc577ad9` ("Big simplification of AP"); this reapplies it.

## What the change does

Stops rebuilding the per-node `ProjectionEvaluator` (its O(num_features)
per-attribute pointer tables + column-mean fills + one dynamic_cast per feature)
on **every node of every tree**. The evaluator only depends on
`(dataset, numerical_features)`, which are constant across all nodes under
`GLOBAL_IMPUTATION`, so it is built once per thread and reused. This is the
upstream `// TODO: Cache.` in `FindBestConditionSparseObliqueTemplate`.

- **Off by default.** Enable with `--config=cache_projection_evaluator`
  (defines `-DCACHE_PROJECTION_EVALUATOR=1`). Pure `#ifdef` control.
- **Bit-identical values** vs stock — same projected values, same RNG, same
  scheduler. Accuracy must match the control exactly.
- **Safety:** caches only under `GLOBAL_IMPUTATION` (the harness default). Under
  `RANDOM_LOCAL_IMPUTATION` each node trains on a fresh per-node dataset whose
  address could alias a freed one, so the cache is disabled there (rebuilds
  every call, keys cleared). This is the *fixed* version — the original used a
  raw `static thread_local` validated only on feature **count**; the recovered
  version validates on dataset pointer + `nrow` + full feature-vector equality
  and lives on `SplitterPerThreadCache`.

## The four pieces (for review / re-derivation)

1. `.bazelrc`: `build:cache_projection_evaluator --cxxopt="-DCACHE_PROJECTION_EVALUATOR=1"`
2. `training.h`: forward-decl `class ProjectionEvaluatorCache;` in
   `namespace internal` + `std::shared_ptr<internal::ProjectionEvaluatorCache>
   projection_evaluator_cache;` field on `SplitterPerThreadCache`.
3. `oblique.cc`: the `internal::ProjectionEvaluatorCache` class + anon-namespace
   `GetProjectionEvaluator(train_dataset, config_link, dt_config, cache)` helper.
4. `oblique.cc`: two `#ifdef CACHE_PROJECTION_EVALUATOR` call-site swaps in
   `FindBestConditionSparseObliqueTemplate` and
   `FindBestConditionMHLDObliqueTemplate`.

## Build check (already passed on macOS)

```bash
bazel build -c opt //yggdrasil_decision_forests/learner/decision_tree:training                                   # guard OFF
bazel build -c opt --config=cache_projection_evaluator //yggdrasil_decision_forests/learner/decision_tree:training  # guard ON
```
Both compiled clean on this recovery.

## E2E measurement protocol (run on m7i)

Standard harness (`benchmarks/evaluation/runtime.sh`): Oblique + Dynamic Random
Histogram, 64 bins, `dynamic_split_threshold=250`, `NUM_TREES = nproc*5`,
`num_threads=-1`, AVX2 build. The runner passes `EXTRA_BAZEL_CONFIGS` straight
into `bazel build`, so this is a clean A/B on one variable.

Pre-bench (m7i): disable E-core features and source oneAPI if used, then run
runtime + accuracy **sequentially, never concurrently**:
```bash
sudo benchmarks/utils/set_cpu_e_features.sh --disable    # if applicable on the box
# CONTROL (stock, guard off)
EXTRA_BAZEL_CONFIGS="" \
  benchmarks/evaluation/runtime.sh stock_control_m7i
# TREATMENT (cache on)
EXTRA_BAZEL_CONFIGS="--config=cache_projection_evaluator" \
  benchmarks/evaluation/runtime.sh cache_projeval_m7i
```
Outputs land at `benchmarks/results/.../{NUM_RUNS}runs_<suffix>.{csv,log}`; the
log header records `EXTRA_BAZEL_CONFIGS` for provenance. Prefer ≥3 runs and
report medians.

### Accuracy gate (must be exact, not just "close")
Values are bit-identical, so accuracy should match the control to the digit:
```bash
EXTRA_BAZEL_CONFIGS="--config=cache_projection_evaluator" \
  benchmarks/evaluation/accuracy.sh cache_projeval_m7i
```
Compare against a stock accuracy run on the same box. Any divergence = a bug in
the cache-invalidation keys, not an expected model change — stop and report.

## What to compare against (prior single-run m7i result)

The win scales with **feature count** (wider dataset ⇒ bigger per-node O(F)
setup saved). Prior 1-run numbers from the deleted branch, to confirm/refute:

| Dataset            | Expected direction | Prior Δ e2e |
|--------------------|--------------------|-------------|
| HIGGS 11m×29       | ~neutral (F tiny)  | small       |
| trunk ~1.5m×4096   | modest win         | ~ -6.8% @4k feats |
| trunk 150k×40k     | solid win          | ~ -21.7% @40k |
| trunk 10k×400k     | large win          | **-63% @400k** |

Headline claim to verify: **combined with `evaluate_4row` this was the
memory-safe leaderboard leader** (3m×4096: 576 vs 688 stock; 30k×400k: 73 vs
152 prior best). This branch has **only** the cache (one idea per branch); a
separate branch would be needed to reproduce the stacked result.

## Report back

For each dataset shape: control median, treatment median, Δ% e2e, and the
`kProjectionEvaluate` / `kFindObliqueSetup` chrono deltas if a
`CHRONO_PROFILE=1` run is also done. Confirm accuracy matched. Note the box,
core count, `NUM_TREES`, and `NUM_RUNS`. Attach the CSV paths.
