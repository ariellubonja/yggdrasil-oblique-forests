# runs/2026-11-21_spo_gbt_feature_scaling_none_vs_trainstd_cc18_tabarena_tabred

Measurement, stored once. Floats read it through their `inputs.txt`. Index and rules: `benchmarks/results/kdd28/README.md`.

Off-paper probe (`--off_paper "SPO normalization probe: NONE vs train-std scaling"`): does dividing every feature by its
train-fold std change SPO accuracy? YDF's STANDARD_DEVIATION normalization is emulated by pre-scaled fold files
(`scale_folds*.py`: std with ddof 0, test reuses the train std, std < 1e-6 left as is, no centering, label copied).

- **Files:** three sub-runs, each with `results.csv` and the scripts that made it
  - `cc18_10fold/`: 1360 rows, 34 CC18 sets x 10 folds x arms `A_none`, `B_trainstd` x learners SPO-GBT, SPO-RF.
    `run.sh`, `run_cc18_spo_only.py` (driver filtered to the cc18_spo arm), `scale_folds_cc18.py`
  - `tabarena_tabred_mixed_categorical_numeric/`: 308 rows, 15 TabArena sets x 5 folds + 2 TabReD sets x 1 fold, same
    arms and learners. `run.sh`, `scale_folds.py`, `column_types.py`, `column_types.json`
  - `tabarena_categorical_heavy/`: 200 rows, 5 TabArena sets x 5 folds, learners AA-GBT (sanity), SPO-GBT (`run` 1 and a
    repeat `run` 2), SPO-RF. `run_gbt.sh`, `scale_folds.py`
- **Profile:** GBT `kdd28_ydf` (via `run_kdd28_ydf_ec2.py`, `--tables cc18` resp. `distacc --dist_axes=`). SPO-RF:
  `train_oblique_forest` harness defaults
- **Settings:** GBT: YDF defaults, 300 trees, no validation, early stopping off, seed 1, 4 threads. SPO-RF: seed 1,
  8 threads, 3 h timeout per cell. Categoricals are ordinal codes in both arms (`n_ordinal_cols`)
- **Eval suite:** `kdd28_cc18_10fold` (sha256 checked by `run.sh`) and the `kdd28_tabarena33` / TabReD fold files
  (`~/spo_vs_gbt/folds` on the box). Off-paper, so no eval-suite stop
- **Machine:** `cc18_10fold/` on ariel-ydf-7 (i-04668dc60810fb749, c8i.metal-96xl). The two TabArena sub-runs on
  ariel-ydf-1 (i-0f90f7764082bf21a, c8i.96xlarge, `host` column says its old name ariel-dist-accuracy-1)
- **Date:** 2026-11-21
- **Code:** binary 0416519f-dirty (`/home/ubuntu/results/2026-11-06_hao_seeds/bin`), driver ee27dcdb (CC18) and
  23ce9a55 (TabArena), as stamped in the `commit` column
- **Raw cells:** mirrors `ariel_notes/results/2026-11-21_spo_normalization_none_vs_trainstd_cc18_10fold/`,
  `.../2026-11-21_spo_normalization_none_vs_trainstd_mixed_categorical_numeric_tabarena/`,
  `.../2026-11-21_spo_gbt_normalization_none_vs_trainstd_categorical_heavy_tabarena/` (gitignored, logs and `code.tgz`)
- **Read by:** tab:spo-ordinal-feature-scaling (SPO-GBT, `run` 1 only)
