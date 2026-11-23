# runs/2026-11-22_categorical_dropped_vs_ordinal_aa_spo_gbt

Measurement, stored once. Floats read it through their `inputs.txt`. Index and rules: `benchmarks/results/kdd28/README.md`.

- **Files:** `results.csv` (80 rows: 4 datasets x 5 folds x arms `aa_ordinal`, `aa_dropped`, `spo_ordinal`, `spo_dropped`),
  `run.log`, `run.py` (writes `results.csv` next to itself, folds dir as argv[1])
- **What:** categorical columns dropped vs kept as ordinal codes (NUMERICAL), for AA-GBT and SPO-GBT. Extends
  `2026-11-19_ordinal_vs_native` (same folds, same categorical lists: its `cat_columns.json`). `aa_ordinal` is a
  reproduction check: it equals that run's `ordinal` arm on all 20 cells (|diff| < 1e-12, asserted by the generator)
- **Profile:** none (ydf Python package)
- **Settings:** ydf 0.16.1 GBT defaults, 300 trees, validation_ratio 0, early stopping NONE, seed 1, 4 threads. SPO =
  `split_axis=SPARSE_OBLIQUE`, `sparse_oblique_weights=BINARY` (explicit), other SPO params at proto defaults (exponent 2,
  max 6000, density 2, normalization NONE). The script asserts every input feature is NUMERICAL and 300 trees
- **Zero features left:** Amazon_employee_access is all categorical. Its dropped arms train nothing and score a constant
  (AUC .5, `note=no_features_constant_predictor`). in_vehicle_coupon_recommendation keeps 2 features (temperature, time)
  with 15 distinct pairs on fold 0. Its AA and SPO dropped AUCs are equal on all 5 folds although the SPO trees are oblique
  (100/100 sampled conditions use both features, fold 0)
- **Eval suite:** `kdd28_tabarena33` folds of credit-g, in_vehicle_coupon_recommendation, bank-marketing,
  Amazon_employee_access. Local copy `ariel_notes/results/2026-11-06_tabicl_baseline/folds`, all 40 CSVs match the sha256 in
  `benchmarks/results/runtime/speedup_map_by_dataset/MANIFEST/inputs.sha256`
- **Machine:** MacBook (macOS, local `.venv`, ydf 0.16.1 wheel). Accuracy only, timings not for reporting
- **Date:** 2026-11-22
- **Code:** `run.py` here (ydf 0.16.1, `.venv/bin/python run.py <folds dir>`)
- **Raw cells:** `run.log` and `results.csv` here (no mirror)
- **Read by:** tab:ordinal-vs-native
