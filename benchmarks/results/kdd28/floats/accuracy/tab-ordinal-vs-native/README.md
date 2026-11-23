# tab:ordinal-vs-native

Results behind the KDD28 float below. Rebuild the body with the generator command. Index and rules:
`benchmarks/results/kdd28/README.md`.

- **Labels:** `tab:ordinal-vs-native`
  - `tab:ordinal-vs-native`: caption: see main.tex (the body now has AA dropped/ordinal/native and SPO dropped/ordinal, 2026-11-22)
- **Status:** current (appendix, Evaluation Setup). README added 2026-11-20 (the dir had none before)
- **Runs:** `runs/2026-11-19_ordinal_vs_native/` (AA ordinal, native) and `runs/2026-11-22_categorical_dropped_vs_ordinal_aa_spo_gbt/` (AA dropped, SPO dropped and
  ordinal, added 2026-11-22). Run facts: their READMEs
- **Profile:** none (the ydf Python package, not a `kdd28_params.json` profile)
- **Settings:** AA-GBT and SPO-GBT, ydf 0.16.1 defaults, 300 trees, no validation, no early stopping, seed 1, 4 threads. Arms: categoricals
  dropped, ordinal codes (NUMERICAL), native (CATEGORICAL, AA only). SPO = SPARSE_OBLIQUE, BINARY weights, proto defaults
- **Eval suite:** `kdd28_tabarena33` folds of credit-g, in_vehicle_coupon_recommendation, bank-marketing, Amazon_employee_access
- **Machine:** 2026-11-19 run not recorded. 2026-11-22 run: MacBook (macOS). Its AA ordinal arm is bit-equal to the
  2026-11-19 ordinal arm (all 20 cells)
- **Date:** 2026-11-19 (AA ordinal, native), 2026-11-22 (AA dropped, SPO dropped and ordinal)
- **Code:** `runs/2026-11-19_ordinal_vs_native/run.py`, `runs/2026-11-22_categorical_dropped_vs_ordinal_aa_spo_gbt/run.py`
- **Generator:** run from the repo root, writes into the paper dir (no `--out` flag)

```
python3 benchmarks/src/distributed_spo_gbt/paper/tables/make_ordinal_vs_native_table.py
```
- **Inputs:** `inputs.txt`
- **Raw cells:** `runs/2026-11-19_ordinal_vs_native/run.log`, `runs/2026-11-22_categorical_dropped_vs_ordinal_aa_spo_gbt/run.log`, mirror `ariel_notes/results/2026-11-19_ordinal_vs_native_cat/` (gitignored)
