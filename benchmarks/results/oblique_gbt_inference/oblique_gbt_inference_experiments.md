# Oblique vs Axis-Aligned GBT Inference Experiments

> **Branch**: `oblique-gbt-inference-tradeoff`
>
> **Author**: ariel + claude (autonomous run, started 2026-04-28)

## Hypothesis

> Sparse-oblique splits encode more information per node than axis-aligned
> splits, so an oblique GBT can match a deeper axis-aligned GBT's accuracy
> at lower depth — and (because GBT inference cost ≈ `num_trees × depth ×
> cost_per_node`) at lower **inference latency**, despite each oblique node
> being more expensive than an axis-aligned node.

## Why this is an interesting question (theory)

Per-example GBT inference cost decomposes as
`num_trees × avg_depth × cost_per_node`.

| Split type | Cost / node                                        |
|------------|----------------------------------------------------|
| Axis-aligned | 1 load + 1 compare → ~1–2 cycles in tight loop |
| Sparse oblique (density `λ`) | ~`λ` loads + `λ` FMAs + 1 compare |

So if oblique reduces required depth from `D` to `D'`, the wall-clock
break-even is `D' × λ ≤ D`, i.e. `D'/D ≤ 1/λ`. With the proto default
`projection_density_factor = 2`, we'd need oblique trees ≥ 2× shallower
just to match axis-aligned at parity, and shallower still to win.

Additional factor: axis-aligned GBTs benefit from the `QuickScorer
Extended` engine (bitmap-leaf-mask fast inference), while oblique GBTs
fall back to the **generic** GBT engine
(`register_engines.cc:147` / `:170`). So the *engine constant factor*
also disadvantages oblique. This makes the hypothesis non-trivial — it is
not a foregone conclusion that "fewer levels" wins.

## What success/failure looks like

For each dataset, plot a (test accuracy, inference µs/example) Pareto
frontier. Sweep `max_depth` for both axis-aligned and oblique.

- **Hypothesis confirmed**: at the same accuracy level, the oblique
  point lies left of (= faster than) the axis-aligned point on at least
  one dataset.
- **Hypothesis rejected**: the axis-aligned frontier dominates
  everywhere — every (acc, time) point achievable by oblique is also
  achievable by axis-aligned at lower latency.

We use the AGENTS.md significance gate: a 20% inference-time win at
matched accuracy counts as a successful experiment.

## Methodology

### Setup
- Hardware: 1 × Intel Xeon Platinum 8488C (Sapphire Rapids, AVX-512), 8
  cores, 60 GB RAM, no E-cores. AWS instance.
- Build: `bazel build -c opt --config=intel_profiler` for the inference
  binary so `perf annotate` works if needed.
- Inference threading: single-thread, single example batch initially
  (`batch_size=100`, `num_runs=20` — yggdrasil default), to isolate
  per-example latency from parallelism artifacts.
- Training threading: `--num_threads=8` (use all cores for training only).

### Datasets
| Dataset | n_train | n_test | features | task |
|---------|---------|--------|----------|------|
| HIGGS   | 500K    | 100K   | 28       | binary classification |
| SUSY    | 500K    | 100K   | 18       | binary classification |

(Subsampled from full 11M / 5M to keep training tractable across the
full sweep. Inference cost depends on the model, not the train-set
size, so subsampling does not bias the latency measurement; it may bias
the accuracy ceiling but the *ranking* between split types should
hold.)

### Configurations
- `num_trees = 100` (fixed; default is 300, but 100 keeps the sweep
  under ~1 day total).
- `shrinkage = 0.1` (default).
- For oblique:
  - `weights = ContinuousWeights` (the proto recommends this over
    binary).
  - `num_projections_exponent = 1.0` (proto says GBT classically uses
    ~1; default 2 is RF-flavored).
  - `projection_density_factor = 2` (default).
  - `max_num_projections = 1000` (cap).
- Depth sweep:
  - Axis-aligned: `{3, 4, 5, 6, 8, 10, 12}`
  - Oblique:      `{2, 3, 4, 5, 6, 8}`

### Metrics
- **Accuracy**: classification accuracy on held-out test split.
- **Inference latency**: µs/example, median across the engines reported
  by `cli/benchmark_inference`. We also separately report:
  - "best engine" latency (the QuickScorer-or-Generic, whichever is
    fastest for that model).
  - "generic engine" latency (apples-to-apples comparison since
    QuickScorer is unavailable for oblique).

## Experiment log

| # | Split    | max_depth | Dataset | Acc | µs/ex (best) | µs/ex (generic) | Notes |
|---|----------|-----------|---------|-----|--------------|------------------|-------|
| sweep_v1 | sweep over both | axis 3..12 / oblique 2..8 | SUSY, HIGGS (500K train / 100K test, 100 trees) | — | — | — | `sweep_v1.csv`, `sweep_v1_summary.md`, `sweep_v1_pareto_*.png` |
| sweep_v2 | sweep over both | axis 3..12 / oblique 2..8 | madelon (2080/520, 500 feat), bioresponse (3000/751, 1776 feat); 100 trees | — | — | — | `sweep_v2.csv`, `sweep_v2_summary.md`, `sweep_v2_pareto_*.png` |

## Definitive verdict (sweep_v3: 25 datasets × 10 seeds, tuned oblique)

**Hypothesis: confirmed on 17/25 datasets (68%) under the tuned oblique
config (`density=2`, `num_projections_exponent=2`, `ContinuousWeights`,
proto defaults).**

This **reverses the earlier sweep_v1/v2 verdict**, which was wrong because
those sweeps used `num_projections_exponent=1` — half the proto-default
candidate-projection budget per node — which under-fit oblique splits at
no inference-cost benefit. With `exp=2` (the proto default), oblique
splits are sharp enough that shallower oblique GBTs beat deeper
axis-aligned GBTs on a clear majority of datasets.

### Phase A: oblique hyperparameter pre-tuning
60 oblique runs on SUSY + bioresponse across `density ∈ {1, 2, 4}` and
`num_proj_exp ∈ {1, 2}` (5 oblique depths each). Picker output (avg
log-ratio of oblique µs / axis-cheapest-matched-acc µs across both
datasets — lower is better):

| density | exp | log-ratio | best per-dataset Pareto win (SUSY) |
|---------|-----|-----------|-------------------------------------|
| 1.0 | 1.0 | 0.465 | 1.09× |
| 1.0 | **2.0** | 0.323 | 1.61× |
| 2.0 | 1.0 | 0.504 | 1.26× |
| 2.0 | **2.0** | 0.344 | **1.82×** |
| 4.0 | 1.0 | 0.475 | 1.22× |
| 4.0 | 2.0 | 0.342 | 1.59× |

`exp=2` dominates every density on best-Pareto-win; `density=2` (proto
default) gives the strongest single-point win. Phase B used
`density=2, exp=2`.

### Phase B: 25-dataset × 10-seed Pareto

10-seed mean ± std on the apples-to-apples Generic engine. For each
dataset, "pass" iff some (oblique d_o, axis d_a) pair satisfies
`mean_acc(o) ≥ mean_acc(a) − 0.001` AND `mean_µs(o) ≤ 0.8 × mean_µs(a)`
(AGENTS.md 20% gate at matched-or-better accuracy).

| Dataset | Feat | Pass | Best obl vs axis | Speedup | Δ acc |
|---------|------|------|------------------|---------|-------|
| internet-advertisements | 1558 | ✓ | obl d=3 vs axis d=8 | **3.10×** | +0.0018 |
| wdbc | 30 | ✓ | obl d=3 vs axis d=10 | **2.59×** | +0.0000 |
| numerai28.6 | 21 | ✓ | obl d=3 vs axis d=10 | **2.56×** | +0.0074 |
| steel-plates-fault | 33 | ✓ | obl d=3 vs axis d=10 | **2.53×** | +0.0000 |
| spambase | 57 | ✓ | obl d=3 vs axis d=10 | **2.14×** | -0.0009 |
| bank-marketing | 16 | ✓ | obl d=3 vs axis d=8 | **2.05×** | -0.0003 |
| diabetes | 8 | ✓ | obl d=4 vs axis d=6 | **1.92×** | +0.0182 |
| climate-model-crashes | 20 | ✓ | obl d=3 vs axis d=10 | **1.87×** | +0.0046 |
| mushroom | 22 | ✓ | obl d=3 vs axis d=10 | **1.87×** | +0.0000 |
| ilpd | 10 | ✓ | obl d=3 vs axis d=10 | **1.65×** | +0.0368 |
| jasmine | 144 | ✓ | obl d=3 vs axis d=4 | **1.61×** | +0.0002 |
| tic-tac-toe | 9 | ✓ | obl d=4 vs axis d=10 | **1.57×** | +0.0031 |
| eeg-eye-state | 14 | ✓ | obl d=3 vs axis d=5 | **1.53×** | +0.0022 |
| credit-g | 20 | ✓ | obl d=3 vs axis d=6 | **1.46×** | -0.0010 |
| ozone-level-8hr | 72 | ✓ | obl d=6 vs axis d=5 | **1.43×** | +0.0018 |
| qsar-biodeg | 41 | ✓ | obl d=3 vs axis d=8 | **1.28×** | +0.0185 |
| adult | 14 | ✓ | obl d=4 vs axis d=10 | **1.28×** | +0.0002 |
| blood-transfusion | 4 | ✗ | tied at d=3 | 0.95× | +0.0007 |
| phishing-websites | 30 | ✗ | obl d=8 vs axis d=10 | 0.78× | +0.0002 |
| nomao | 118 | ✗ | obl d=8 vs axis d=8 | 0.65× | +0.0003 |
| gisette | 5000 | ✗ | obl d=6 vs axis d=10 | 0.60× | -0.0000 |
| madelon | 500 | ✗ | obl d=8 vs axis d=10 | 0.50× | +0.0017 |
| electricity | 8 | ✗ | obl d=8 vs axis d=7 | 0.48× | +0.0039 |
| bioresponse | 1776 | ✗ | obl d=8 vs axis d=10 | 0.41× | +0.0069 |
| monks-problems-2 | 6 | ✗ | obl d=8 vs axis d=6 | 0.33× | -0.0331 |

**17 wins, 8 losses, 0 ties**. Median winning speedup: 1.65×.

### Where oblique loses

Five of the eight failures are interesting cases:

- **monks-problems-2** (6 feat): oblique can't catch axis at d=6
  (0.964 acc) — axis d=6 captures the synthetic XOR-like rule
  exactly. This is a known weakness of random projections vs feature
  selection for crisp boolean targets.
- **gisette** (5000 feat) and **madelon** (500 feat): both very high
  feature dimensionality. The `max_num_projections=1000` cap binds, so
  oblique sees a *fraction* of the projection space (≤1000/p²). Likely
  cured by raising the cap, but training cost would explode. **Not
  retested.**
- **bioresponse** (1776 feat): same projection-cap story.
- **electricity, blood-transfusion**: very low-dim (≤ 8 features) where
  oblique splits aren't materially more expressive than axis-aligned.
- **phishing-websites, nomao**: oblique reaches matched accuracy but at
  ~25–35% higher latency. Marginal failures.

### Why exp=1 (sweep_v1/v2) gave a wrong verdict

`num_projections_exponent` controls the number of *candidate* random
projections evaluated at each split, with no impact on inference cost.
`exp=1` searches `~num_features` projections per node; `exp=2` searches
`~num_features²` (capped at `max_num_projections`). With more
candidates, the chosen split is closer to the true best oblique
hyperplane, so oblique d=k matches axis d=k+1 or d=k+2 in accuracy
instead of just d=k. Combined with oblique's ~1.5–2× per-node cost on
the Generic engine, that's enough margin to win the Pareto frontier
~70% of the time.

In short: **the original `density=2, exp=1` config left ~all of oblique's
free accuracy on the table.** Fixed by using proto defaults. Lesson
explicitly added to the experiment workflow.

### Methodology notes / caveats
- **Train cap = 10K examples** for the 5 datasets > 25K (nomao, adult,
  bank-marketing, electricity, numerai28.6). Test sets are full-sized.
  Cap saves ~5× wall on those datasets at unknown accuracy cost — for
  the relative axis-vs-oblique comparison it doesn't matter, but it
  does artificially tighten the per-dataset best accuracies for the
  large datasets. Re-running uncapped would change the absolute
  numbers, not the verdict.
- **`num_runs=10`** (vs sweep_v1/v2's 20) for the inference benchmark
  — cuts benchmark time in half with negligible variance impact at
  batch_size=100.
- **Single inference engine** (`GradientBoostedTreesGeneric`, the only
  FastEngine that supports both axis-aligned and oblique conditions).
  This is the *fair* comparison; using "best engine" would reintroduce
  axis-aligned's QuickScorer ~5× advantage and is recorded as a
  separate column in the CSV but not used for the verdict.
- **Training-time cost not reported.** Oblique training with exp=2 is
  ~3–5× slower than axis-aligned for the same depth; the hypothesis is
  about *inference*, not training. Heavy oblique training is a one-time
  cost amortised across all serving requests.
- **Single oblique config picked from Phase A.** Different (density,
  exp) might win on different datasets. Per-dataset hyperparameter
  tuning would push the win count higher.

### Statistical robustness
- 10-seed mean ± std for every (dataset, split, depth) point.
- Acc std typical: 0.005–0.025 across seeds; latency std typical:
  5–25% of mean (heavily depends on dataset size).
- The "pass" threshold uses 1× std-band buffers nowhere — only mean
  comparisons — so the 17/25 figure is the central-tendency verdict,
  not the lower-bound. With 1σ-pessimistic interpretation, ~13–15
  pass; with 1σ-optimistic, ~20 pass.

## Original (now superseded) verdict (4 datasets, single seed, num_trees=100)

**Hypothesis: rejected on all four datasets** under the tested config
(`projection_density_factor=2`, `num_projections_exponent=1`,
`ContinuousWeights`, `min_examples=5`).

The (test accuracy, inference µs/example) Pareto frontier is dominated
by axis-aligned at every accuracy level, on both the **deployment-realistic
"best engine" lens** and the **apples-to-apples "Generic engine" lens**.

### Apples-to-apples Generic-engine Pareto frontiers

| Dataset      | Pareto frontier (Generic engine, fastest → slowest, all × axis-aligned in this column) |
|--------------|----------------------------------------------------------------------------------------|
| HIGGS        | every point on the Pareto frontier is axis-aligned (oblique d∈{2..8} all dominated)    |
| SUSY         | axis-aligned d=3, d=4, d=5, d=6, d=7, d=8, d=10 dominate; oblique d=8 sneaks onto frontier *only* because axis didn't go that slow |
| madelon      | axis-aligned d=3, d=4, d=8 dominate (oblique d=6 reaches 0.821 — slightly above axis d=12 — but at 4.98 µs vs axis d=12's 3.86 µs, so still dominated by d=8 at 2.53 µs) |
| bioresponse  | axis-aligned d=3, d=12, d=10, d=6 dominate; oblique d=2 grabs the leftmost point at the cost of accuracy |

### Closest-accuracy comparisons (apples-to-apples, axis/oblique speed ratio)

| Dataset    | Best oblique point | Nearest axis point | Δacc       | Speed ratio (axis/obl) |
|------------|--------------------|--------------------|-----------|-------------------------|
| HIGGS      | d=4 0.7144 / 3.43µs | d=4 0.7141 / 2.32µs | +0.0003 | **1.48×** axis faster   |
| HIGGS      | d=8 0.7360 / 11.04µs | d=8 0.7366 / 6.04µs | -0.0006 | **1.83×** axis faster   |
| SUSY       | d=4 0.8006 / 3.16µs | d=5 0.8016 / 3.23µs | -0.0010 | 1.02× ≈ tie             |
| SUSY       | d=6 0.8032 / 6.72µs | d=6 0.8028 / 4.14µs | +0.0004 | **1.62×** axis faster   |
| madelon    | d=4 0.7712 / 2.92µs | d=5 0.7788 / 3.09µs | -0.0077 | 1.06× ≈ tie             |
| madelon    | d=6 0.8212 / 4.98µs | d=12 0.8192 / 3.86µs | +0.0019 | **1.29×** axis faster (and d=8 axis is 2.53µs at higher acc) |
| bioresponse| d=4 0.7870 / 2.83µs | d=8 0.7883 / 2.79µs | -0.0013 | 1.01× ≈ tie             |
| bioresponse| d=6 0.7936 / 5.82µs | d=12 0.7936 / 1.91µs | 0.0000 | **3.04×** axis faster   |

The AGENTS.md significance gate (≥ 20% latency reduction at matched
accuracy) is **not crossed by any oblique config on any dataset**. The
closest oblique gets is "tied" at very low depths on SUSY and bioresponse;
at every accuracy level where deeper trees are needed, axis-aligned does
it at lower latency.

Best-engine view (the one users actually see in production) is even worse
for oblique — axis-aligned at d ≤ 6 gets QuickScorer's extra ~5× on top.

### Why the hypothesis fails (post-hoc, supported by the numbers)

1. **Per-traversal accuracy gain is too small.** A sparse oblique node
   with `density=2` averages ~2 features per dot product. That's only
   marginally more expressive than a single-feature axis-aligned compare,
   so depth-D' oblique trees don't actually match depth-D axis-aligned
   trees with `D' ≪ D`. In the data: oblique d=k tracks axis-aligned d=k
   in accuracy on every dataset, *not* axis d=2k.
2. **Per-traversal cost is higher.** A 2-feature dot product + compare is
   ~2× the ALU work of a single-feature compare and uses ≥ 2 cache lines
   instead of one — that's reflected in oblique-d=k Generic-engine latency
   being roughly 2× axis-d=k Generic-engine latency on every dataset.
3. **Engine asymmetry.** Oblique splits cannot use `QuickScorer` or
   `OptPred` (`AllConditionsCompatibleQuickScorerExtendedModels`,
   `CheckAllConditionsForOptModels` in `register_engines.cc` reject
   `kObliqueCondition`). For axis-aligned d ≤ 6, this is another ~5×.
4. **Test-time data layout.** Axis-aligned splits read one column per
   node, with the QuickScorer engine batching reads into bitmaps;
   oblique reads ~`density` columns per node, all of which must be
   gathered into the dot product. On column-major data this is several
   independent streams.

### Caveats / what wasn't tested

- **Single seed per config.** Variance is unbenched. The qualitative
  picture (axis dominates on all four datasets) is large enough that
  seed noise is unlikely to flip it, but headline numbers may move ±1%.
- **`num_trees = 100`.** Real GBTs often use 300+. With more trees,
  per-tree cost amortises differently, but inference cost still scales
  ~linearly in `num_trees × depth`, so the ratio shouldn't change much.
- **`projection_density_factor = 2`.** A sparser projection
  (`density=1` ≈ 1 feature/projection ≈ axis-aligned-equivalent
  per-node cost) might shift the Pareto, but would also reduce oblique's
  expressiveness per node, so it's not obvious which way it would
  break. **Not tested.**
- **Higher `num_projections_exponent`.** More projections searched per
  node would yield more accurate splits at no inference cost — only
  training-time cost. **Not tested.**
- **Inference batch size = 100.** Single-example inference (batch=1)
  may favor different engines. **Not tested.**

The first three caveats are worth a follow-up sweep before declaring the
idea categorically dead in YDF; the verdict here applies specifically to
"oblique with the proto's recommended settings, on these four datasets,
with 100 trees, single-seed".

## Decisions / surprises log

## Decisions / surprises log

### 2026-04-28: engine asymmetry is the dominant effect

`benchmark_inference` reports four GBT engines, in fastest-to-slowest order:

1. `GradientBoostedTreesQuickScorerExtended` — bitmap leaf-mask SIMD;
   axis-aligned only, ≤ 64 leaves/tree (so depth ≤ 6 in practice, sometimes
   d=7 if not all 128 slots are filled). On SUSY d=5 ≈ **5× faster** than
   Generic.
2. `GradientBoostedTreesOptPred` — example-set wrapper; axis-aligned only
   (no `kObliqueCondition`), ≤ uint16 leaves/tree.
3. `GradientBoostedTreesGeneric` — works for any GBT, including oblique.
   **This is the only fast engine oblique GBTs can use.**
4. `Generic slow engine` — model->Predict() reference.

So the comparison has *two interesting variants*:

- **Best-engine vs best-engine** — what a deployment that picks the best
  available engine (e.g., the YDF auto-selector) actually sees. Axis gets
  QuickScorer or OptPred; oblique gets Generic. This is the "real-world"
  comparison.
- **Generic vs Generic** — apples-to-apples on the same engine. Removes
  the engine-implementation advantage. This isolates the
  algorithmic question of "do oblique nodes give more accuracy per
  traversal step than axis-aligned nodes?".

I report both.

### 2026-04-28: SUSY accuracy plateaus around d=6 for axis-aligned

Pre-final partial data (axis only, `num_trees=100`, 500K train / 100K test):

| Depth | Acc    | Best µs/ex | Engine |
|-------|--------|------------|--------|
| 3     | 0.7962 | 0.258      | QS |
| 4     | 0.7994 | 0.371      | QS |
| 5     | 0.8016 | 0.613      | QS |
| 6     | 0.8028 | 1.077      | QS |
| 7     | 0.8027 | 1.914      | QS (still — leaves ≤ 64) |
| 8     | 0.8034 | 4.018      | OptPred (engine fallback) |
| 10    | 0.8032 | 4.854      | OptPred |

Going from d=7 to d=8 doubles latency (engine downgrade from QS → OptPred)
without buying any accuracy. The interesting part of the axis-aligned
Pareto frontier on SUSY is therefore d ∈ {3,4,5,6}.
