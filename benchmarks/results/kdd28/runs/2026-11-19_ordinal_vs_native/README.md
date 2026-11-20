# runs/2026-11-19_ordinal_vs_native

Measurement, stored once. Floats read it through their `inputs.txt`. Index and rules: `benchmarks/results/kdd28/README.md`.

- **Files:** `results.csv` (40 rows: 4 datasets x 5 folds x arms ordinal, native), `run.log`, `run.py` (writes `results.csv`
  next to itself, folds dir as argv[1]), `cat_columns.json` (categorical columns per dataset)
- **Profile:** none (ydf Python package)
- **Settings:** AA-GBT, ydf 0.16.1 defaults, 300 trees, validation_ratio 0, early stopping NONE, seed 1, 4 threads. Arm
  `ordinal` = codes as NUMERICAL, `native` = the same codes as CATEGORICAL. The script asserts the trained semantics
- **Eval suite:** `kdd28_tabarena33` folds of credit-g, in_vehicle_coupon_recommendation, bank-marketing,
  Amazon_employee_access (box `~/spo_vs_gbt/folds`, sha256 per MANIFEST/inputs.sha256)
- **Machine:** not recorded. The mirror's `run.py` read the folds from `ariel_notes/results/2026-11-06_tabicl_baseline/folds`
- **Date:** 2026-11-19
- **Code:** `run.py` here (ydf 0.16.1)
- **Raw cells:** `run.log` here, mirror `ariel_notes/results/2026-11-19_ordinal_vs_native_cat/` (gitignored, same
  `results.csv`, `run.log`, `cat_columns.json` bytes)
- **Read by:** tab:ordinal-vs-native
