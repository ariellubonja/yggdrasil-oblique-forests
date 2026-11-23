# runs/2026-11-22_wide_real_dataset_candidates_spo_vs_aa_screen_hao

Measurement, stored once. Screening study, read by no float. Index and rules: `benchmarks/results/kdd28/README.md`.

- **Files:** `<profile>/seed<s>/{distacc_stamped,reg,reg_dist}/results.csv` as the driver wrote them, all rc = 0 (1151 rows).
  `kdd28_hao`: single:SPO on 17 sets (seed 1) and the 14 sets with a positive seed-1 effect (seeds 2-5), plus `{4,8,16,32}:SPO`
  (r = 0) on the 5 passers, seeds 1-5. `kdd28_ydf`: single:AA, same sets and seeds. Classification (binary, multiclass as
  majority-vs-rest) in `distacc_stamped` (test AUC), regression (`ct_slices`, `blogfeedback`) in `reg`/`reg_dist` (test RMSE).
  `derived/` = output of `select_wide_real_candidates.py` (rule F of
  `floats/accuracy/tab-accuracy-hao-vs-ydf-params-single-machine/select_spo_clearly_beats_aa.py`, BH over the new rows
  (a) and pooled with the 101 existing kdd28_hao rows (b), r = 0 gaps, column-replication grid cost) and `what_changed.csv`
- **Data:** `benchmarks/data/prep_wide_real_candidate_folds.py` (manifest `kdd28_wide_real_candidates_manifest.json`, 150
  raw + fold sha256 in `.sha256`). Reuses `encode_features_keep_nan`, `encode_binary_label`, `run_suite.make_folds` unchanged.
  11 OpenML sets (5 stratified CV folds seed 0), `ct_slices` (UCI, 5 KFold), and 5 shipped splits as one holdout fold
  (gisette train/valid, Fashion-MNIST 60k/10k, HAR subject split, ISOLET speaker split, BlogFeedback temporal split). No
  NaN in any source, so no cell was imputed. Self-check `benchmarks/data/check_wide_real_candidate_folds.py`: 17/17 OK. A
  second build from fresh downloads reproduced all 150 files byte-identically. Pinned venv numpy 2.5.2, pandas 3.0.5,
  scikit-learn 1.9.0, pyarrow 25.0.1
- **Profile:** SPO `kdd28_hao.distacc_single_spo`, AA `kdd28_ydf.distacc_single_aa` (Hao's settings do not touch AA),
  distributed `kdd28_hao.distacc_spo` at W = 4, 8, 16, 32, r = 0
- **Settings:** trees=300, max_depth=6, min_examples=5, split=EXACT, validation_set_ratio=0, early_stopping=NONE. SPO:
  exponent 1, max 1000, density 1.5, min 1, spo_weights=BINARY (logged effective-params). Threads 4 single, 2 per worker.
  Distributed: split_budget=true, local_projection_count=false, force_numerical, max_unique_discretized=1, in-process MULTI_THREAD
- **Eval suite:** `kdd28_wide_real_candidates` (new 2026-11-22). Seed 1 on every set, seeds 2-5 only where the seed-1 effect
  is > 0 (har = 0, gisette and blogfeedback < 0 got seed 1 only)
- **Machine:** EC2 ariel-ydf-4 (c8i.metal-96xl, i-07de26fdf094924be, SMT + turbo on, stopped after the run). Shared-box timing
- **Date:** 2026-11-22 20:00Z to 20:53Z
- **Code:** driver `accuracy_multi_thread/run_kdd28_ydf_ec2.py` of a6917d8a + working tree (new table `distacc_stamped` =
  `distacc` + bin_sha/machine columns), stamped `git=a6917d8a-dirty`. Binaries: `train_oblique_forest` ccd48156c4c4 and
  classification `train_distributed_gbt` 802b16fdf3b2 (`~/results/2026-11-06_hao_seeds/bin`), regression
  `train_distributed_gbt` ccfd771b6b1f (`--task` binary, `~/results/2026-11-21_where_spo_beats_aa_cc18_reg_distributed/bin_task`)
- **Checks:** pipeline check before new data. philippine (kdd28_wide_datasets, rebuilt here, 12/12 sha256 match) single:SPO
  seeds 1-2, single:AA seed 1, 4:SPO seed 1: 20/20 cells equal the recorded AUC to every printed digit (fold 0 seed 1
  0.880792). concrete_compressive_strength (kdd28_regression) single SPO/AA seed 1 and 4:SPO seed 3: 15/15 equal RMSE
- **Notes:** rule F passes (a) and (b): ct_slices, madeline, fabert, robert, jasmine. **Selected: madeline, fabert, robert,
  jasmine.** Pooling flips one existing row: QSAR-TID-11[regression] fails -> passes.
- **ct_slices excluded (user decision 2026-11-22):** it passes rule F under the standard KFold (+21.6 %, 25/25 cells), but
  the data is grouped by patient (97 patientId values), so KFold puts slices of one patient in train and test (leakage).
  Off-paper GroupKFold(5) by patient, seed 1: AA RMSE 6.36 vs SPO 6.12, +3.7 %, 4/5 folds, one-sided paired t p = 0.052.
  That is above q = 0.05, so the BH test of rule F rejects it at any family size. Seeds 2-5 were not run on the grouped
  folds. Cells, fold script and launcher: `offpaper_ct_slices_patient_grouped_folds/`. The selection script prints this
  test and writes `selected_a`, `selected_b`, `excluded` next to the untouched rule columns `pass_a`, `pass_b`
- **Other notes:** madeline may be artificial (unverified). musk molecule_name dropped
  as an ID column: its MUSK-/NON-MUSK- prefix equals the label in 6598/6598 rows
- **Raw cells:** gitignored `ariel_notes/results/2026-11-22_wide_real_dataset_candidates_spo_vs_aa_screen_hao/box/` (CSVs,
  logs, launch scripts, fold meta.json, `models_and_cells.tar.zst` sha256 a45dea98). Box copy:
  `~/results/2026-11-22_wide_real_dataset_candidates_spo_vs_aa_screen_hao/` on ariel-ydf-4 (fold files under `data/`)
- **Read by:** no float
