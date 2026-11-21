# runs/2026-11-20_regression_multiclass_tabarena_tabred

Measurement, stored once. Floats read it through their `inputs.txt`. Index and rules: `benchmarks/results/kdd28/README.md`.

- **Files:** `<profile>/seed<s>/{distacc,distacc_aaspo,reg}/results.csv`, profile kdd28_ydf and kdd28_hao, s = 1-5, as the driver
  wrote them, all rc = 0. Multiclass (distacc + distacc_aaspo, single machine only): 8 TabArena tasks x 5 folds, YDF arms
  single:SPO, single:AA, single:AA+SPO, Hao single:SPO, single:AA+SPO (Hao's settings do not touch AA). Regression (reg):
  13 TabArena x 5 folds + 5 TabReD x 1 chronological holdout, same arms, `--task regression`, test RMSE
- **Data:** `benchmarks/data/prep_extra_tabular_folds.py` (user request 2026-11-20). Multiclass -> majority class vs. rest
  (`encode_binary_label`), regression label passed through as float. Ordinal categoricals, train-fold NaN means, TabArena 5
  stratified (classification) or KFold (regression) folds seed 0, TabReD = HF harbor `train_full.csv` (40k-row subsets, sberbank
  23,674), first 80 % train. Manifest `benchmarks/data/kdd28_extra_datasets_manifest.sha256` (220 fold files). Independent
  Fable-xhigh verifier (2026-11-20): all 220 fold files re-derived from raw sources, OpenML ids and HF sha256 checked, TabReD
  timestamps non-decreasing, stock ydf reproduces harness AUC and RMSE on saved models
- **Profile:** `kdd28_ydf.distacc_single_{spo,aa,aa_spo}` and `kdd28_hao.distacc_single_{spo,aa_spo}`, seeds 1-5
- **Eval suites:** `kdd28_tabarena_multiclass`, `kdd28_regression` (new 2026-11-20)
- **Machine:** EC2 ariel-ydf-4 (c8i.metal-96xl, i-07de26fdf094924be, stopped), same binaries as the CC18 seeds (md5 2eaadddf).
  Smoke on ariel-ydf-1 (c8i.96xlarge) and ariel-ydf-7 (c8i.metal-96xl): 53/53 cells identical across the two. Shared-box timing
- **Date:** 2026-11-21 01:12Z to 01:44Z
- **Code:** driver `accuracy_multi_thread/run_kdd28_ydf_ec2.py` (new `reg` table) of the 2026-11-20 working tree, binaries
  0416519f-dirty (`bin_sha` ccd48156c4c4), stamped `git=57d9fd14-dirty`. Launcher `ariel_notes/results/2026-11-20_extra_datasets/box4/run_extra_seeds.sh`
- **Notes:** hiva_agnostic AUC < .5 in every arm. An independent sklearn check on the same majority-vs-rest target (5
  stratified folds, seed 0) gives logistic regression .444 and HistGradientBoosting .468, so it is the data, not this pipeline.
  sberbank-housing SPO RMSE varies with seed (YDF .79 to 1.17, Hao .45 to .65), AA .255 to .258. Why: hypothesis, untested
- **Raw cells:** gitignored `ariel_notes/results/2026-11-20_extra_datasets/` (`box4/` CSVs + logs, `extra_models.tar.zst`,
  `box1/` prep log, smoke, verifier scripts)
- **Read by:** tab:accuracy-hao-vs-ydf-params-single-machine (multiclass + regression continuation body)
