# Split-finder accuracy study: Random Histogram vs Exact (SPO-RF + SPO-GBT)

Everything in this directory is one study, run 2026-08-11/12 on the m7i
(Intel Xeon Platinum 8488C, 48 cores, 377 GB), branch `hve-accuracy` =
`3a784958` + a small evaluation patch (see "Code changes" below). The question:
does Random Histogramming (64 bins) or the shipped Dynamic Random Histogram
config cost accuracy vs Exact split finding, for sparse-projection-oblique
Random Forest (Bagging) and GBT (Boosting)?

## The metric: held-out test accuracy, NOT OOB

Every number in these CSVs is **held-out test-set** performance. Each training
run is given `--test_csv`; the binary evaluates the trained model on that file
and logs one line (`test-accuracy: … test-auc: … test-logloss: …`), which
`benchmarks/utils/parse_log_to_csv.py` prefers over the RF-OOB / GBT
train-accuracy fallback. The fallback was never used here — every fold cell in
every CSV is test-set accuracy. OOB is still computed during RF training
(`--compute_oob_performances=true`) but does not appear in the CSVs.

Side files carry the extra metrics from the *same* evaluation pass:
`*_auc.csv` (positive-class one-vs-rest ROC AUC — both binary ROCs carry the
same discrimination) and `*_logloss.csv`. Same shape as the main CSV.

## How the runs were made

Two mechanisms, both driven by `benchmarks/evaluation/run_hve_m7i_all.sh`
(phases 0–4, resumable — finished CSVs are skipped on rerun):

1. **CC18 sweeps** (`accuracy_hve_<learner>_<arm>_s<seed>*.csv`): plain
   `benchmarks/evaluation/accuracy.sh <suffix>` invocations, one per
   learner × arm × seed, parameterized via `EXTRA_TRAIN_ARGS` /
   `EXTRA_BAZEL_CONFIGS` (recorded in each CSV's provenance header).
   accuracy.sh = 10-fold CV over the 34 OpenML CC18 binary tasks in
   `benchmarks/data/cc18_binary_csv/` (pre-split
   `repeat0_fold{0..9}_sample0_{train,test}.csv`; label = last header column).
   One training run per fold; `fold_k` column = fold k−1's test CSV.
   `--num_trees` is owned by accuracy.sh: **RF 240** (5 × nproc), **GBT 300**.
   The seed sweep is the outer loop of
   `benchmarks/evaluation/run_hist_vs_exact_accuracy.sh`.

2. **Physics held-out runs** (`accuracy_hve_physics*.csv`): NOT accuracy.sh —
   direct `train_oblique_forest` invocations (phase1 of the chain), one run
   per dataset × learner × arm at the full runtime-benchmark scale:
   HIGGS 10.5M-row train / last-500k test, SUSY 4.5M / 500k, RF 240 trees,
   GBT 300, seed 1. Single sample per row (`fold_1`); raw log kept as
   `accuracy_hve_physics.log`. These CSVs have no provenance header.

Build: icx (oneAPI 2025.2), `-c opt -O3 -march=native`, rebuilt by accuracy.sh
per invocation; `--num_threads=-1` (48). Runs were strictly serialized.

## The arms

| Arm in filename | Flags | Meaning |
|---|---|---|
| `exact` | `--numerical_split_type "Exact"` | baseline: exact split finding at every node |
| `rand` | `--numerical_split_type "Random" --histogram_num_bins=64` | random histogram at EVERY node — worst case, upper-bounds any hybrid's penalty |
| `dyn` | `--numerical_split_type "Dynamic Random Histogram" --histogram_num_bins=64 --dynamic_split_threshold=250` | the shipped runtime config: histogram ≥250 rows, Exact below |
| `rand_b{16,32,128,256}` | Random at other bin counts | bin-count ablation, seed 1 only |
| `rand_scalar` | rand + `--config=disable_std_upper_bound_vectorization` | scalar rebuild for the bit-identity check |

Learners: `rf` = SPORF (Bagging, the default) and `gbt` = `--ensemble_method
Boosting`. Seeds `s1`/`s2`/`s3` via `--seed=N`; seed 1 is the primary
analysis, seeds 2–3 exist to measure the seed-noise yardstick (is switching
the split finder distinguishable from switching the RNG seed?).

## Analysis

`hist_vs_exact_accuracy/` is the output of `benchmarks/utils/accuracy_stats.py`
over all these CSVs: `report.md` (all seeds; the physics rows join CC18 as 2
extra datasets → 36 total), `seed1_only/` (primary view), per-dataset CSVs,
LaTeX tables, `tidy.csv`. Unit of analysis is the dataset; per-dataset paired
fold deltas are aggregated with t/bootstrap CIs, Wilcoxon (Holm), TOST
equivalence AND one-sided non-inferiority at margins 0.001–0.02,
leave-one-out, Friedman ranks, and the seed yardstick.

## Code changes behind the numbers

`examples/train_oblique_forest.cc` gained `test-auc:`/`test-logloss:` on the
existing test-evaluation log line; `parse_log_to_csv.py` gained the `_auc`/
`_logloss` side-CSV emission; accuracy.sh prepends its provenance block to the
side files too. Training and timing paths untouched.

## Incidents / caveats (read before reusing the data)

- **Physics label-token bug (fixed; first pass quarantined).** The harness's
  single-pass train loader builds the label vocab from `FormatLabelKey(float)`
  ("0"/"1"), but `--test_csv` is loaded with the stock reader which
  **string-matches raw tokens** against that vocab. HIGGS/SUSY source CSVs
  store labels as `1.000000000000000000e+00` → every test row went
  out-of-vocabulary → test-accuracy exactly 0, logloss 36.0437 (−ln
  DBL_EPSILON). Fix: the label column of `HIGGS_test_500k.csv` /
  `SUSY_test_500k.csv` was rewritten to integer tokens (`0`/`1`) and phase1
  rerun; training semantics are unaffected (train labels parse numerically).
  The degenerate first-pass outputs are in `quarantine_labelmap_bug/`.
  Regenerating the test splits from `*_with_header.csv` reintroduces the bug.
  CC18 files already carry integer labels and were never affected.
- **Scalar-vs-vectorized bit-identity (phase3).** RF: scalar and vectorized
  builds are bit-identical on all 340 fold cells. GBT: 4/34 datasets differ
  (per-dataset mean |Δ| ≤ 0.01). Investigated: not the SIMD upper_bound (it is
  index-exact by construction, DCHECK-cross-checked in debug, and RF exercises
  the identical AVX2 64-bin path) and not nondeterminism (GBT re-runs are
  bit-identical at 48 threads). The `-D` flag changes training.cc's TU
  contents, and icx's default fast fp-model may re-associate float
  accumulation in GBT's gradient split scoring on recompile; RF's integer
  label-count accumulation is immune. So "vectorized ≡ scalar" is proven
  bit-exact for RF; for GBT the observed diffs are compiler fast-math codegen
  variation at seed-noise scale.
- The `_scalar_` CSVs duplicate the `rand_s1` arm (that is their point);
  accuracy_stats.py de-duplicates them (keeps the vectorized source), so
  passing both to the stats is safe.
- Logs: `hve_m7i.pass1.log` = full first pass of the chain,
  `hve_m7i.log` = the resume pass (phase1 rerun + phase4).

Headline results live in `hist_vs_exact_accuracy/report.md`; short version:
Random/Dynamic are non-inferior to Exact within 0.001–0.005 accuracy
(dataset-level, α=0.05), the only Holm-significant penalty case is
`task_219_electricity` (−0.005…−0.017 depending on learner/arm), cylinder-bands
is a large pro-Random outlier for RF (Exact degenerates on its missing values),
and on HIGGS/SUSY at full scale all arms agree within 0.002 accuracy/AUC.
