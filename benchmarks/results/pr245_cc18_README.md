# PR #245 (oblique histogram split) — CC18 accuracy validation, 2026-08-14

Upstream PR google/yggdrasil-decision-forests#245 "Add Histogram split search
supports for Oblique forests". Per reviewer request, implemented as a dedicated
proto field `SparseObliqueSplit.projection_split` (tag 12, type `NumericalSplit`)
with dispatch in `EvaluateProjection` (oblique.cc); branch
`feature-add-oblique-histogramming`.

**Question:** does routing oblique projected values through the histogram split
finder (equal-width and random) preserve accuracy vs the EXACT oblique split?

## Setup

34 binary CC18 tasks × 10-fold CV, RF 100 trees, upstream CLI train/evaluate,
zero errored runs. `num_projections_exponent=0.5` in all arms (upstream default
2.0 ⇒ ~F² projections/node; numerai28.6 exceeded 10 min for one fold).
`NumericalSplit.num_candidates` defaults: HISTOGRAM_RANDOM ⇒ 1,
HISTOGRAM_EQUAL_WIDTH ⇒ 255. Code paths verified by byte-comparing the model's
`nodes-*` files, not accuracy (small test folds collide); absent
`projection_split` gives byte-identical trees to EXACT (backward compat).

## Files & arms

Single file `pr245_cc18_accuracy.csv`, 34 ds × 10 folds × 4 arms:

| arm | meaning |
|---|---|
| `exact` | EXACT oblique split (control; shared across runs, verified identical) |
| `hist_eqw` | HISTOGRAM_EQUAL_WIDTH, 255 candidates |
| `hist_random_nc1` | HISTOGRAM_RANDOM, **num_candidates=1** (inherited axis-aligned default) |
| `hist_random_nc64` | HISTOGRAM_RANDOM, **num_candidates=64 explicit** (diagnostic re-run) |

## Results (per-dataset mean Δ accuracy vs `exact`, 95% CI over 34 datasets)

| arm | mean Δ | CI95 | wins/ties/losses |
|---|---|---|---|
| hist_eqw (255) | +0.0003 | [−0.0007, +0.0014] | 16/3/15 |
| hist_random, nc=1 | **−0.0099** | [−0.0165, −0.0032] | 7/2/25 (madelon −7.6pp, electricity −5.2pp, churn −4.9pp) |
| hist_random, nc=64 | −0.0001 | [−0.0016, +0.0013] | 13/3/18 |

## Conclusions

1. Equal-width histogram is statistically indistinguishable from EXACT.
2. Random histogram at the inherited `num_candidates=1` is a real regression;
   at 64 the gap closes entirely ⇒ the deficit is the candidate count, not the
   `projection_split` dispatch.
3. Therefore the oblique `projection_split` HISTOGRAM_RANDOM default is **64**,
   deliberately diverging from axis-aligned `numerical_split` (still 1, verified
   byte-identical / no regression). A default-plumbing check (num_candidates
   omitted ⇒ 64 from code) was bit-identical to the explicit-64 arm on the 5
   worst-offender datasets.

Open items: `num_candidates: 0` set explicitly yields a degenerate
majority-class model; regression tasks unvalidated (dispatch is
classification-only).
