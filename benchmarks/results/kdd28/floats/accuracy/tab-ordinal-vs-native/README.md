# tab:ordinal-vs-native

Results behind the KDD28 float below. Rebuild the body with the generator command. Index and rules:
`benchmarks/results/kdd28/README.md`.

- **Labels:** `tab:ordinal-vs-native`
  - `tab:ordinal-vs-native`: "Test AUC of single-machine AA-GBT with categorical features as ordinal codes vs.\ native YDF categoricals."
- **Status:** current (appendix, Evaluation Setup). README added 2026-11-20 (the dir had none before)
- **Runs:** `runs/2026-11-19_ordinal_vs_native/`. Run facts: its README
- **Profile:** none (the ydf Python package, not a `kdd28_params.json` profile)
- **Settings:** AA-GBT, ydf 0.16.1 defaults, 300 trees, no validation, no early stopping, seed 1, 4 threads. Arms `ordinal`
  (codes as NUMERICAL) and `native` (the same codes as CATEGORICAL)
- **Eval suite:** `kdd28_tabarena33` folds of credit-g, in_vehicle_coupon_recommendation, bank-marketing, Amazon_employee_access
- **Machine:** not recorded (see the run README)
- **Date:** 2026-11-19
- **Code:** `runs/2026-11-19_ordinal_vs_native/run.py`
- **Generator:** run from the repo root, writes into the paper dir (no `--out` flag)

```
python3 benchmarks/src/distributed_spo_gbt/paper/tables/make_ordinal_vs_native_table.py
```
- **Inputs:** `inputs.txt`
- **Raw cells:** `runs/2026-11-19_ordinal_vs_native/run.log`, mirror `ariel_notes/results/2026-11-19_ordinal_vs_native_cat/` (gitignored)
