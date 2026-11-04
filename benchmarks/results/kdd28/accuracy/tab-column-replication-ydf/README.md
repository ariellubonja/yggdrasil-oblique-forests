# tab:column-replication-ydf

Results behind the KDD28 float(s) below. Rebuild the body with the generator command. Index and rules:
`benchmarks/results/kdd28/README.md`.

- **Labels:** `tab:column-replication-ydf`, `fig:column-replication-ydf`
  - `tab:column-replication-ydf`: "Test accuracy of distributed SPO-GBT vs.\ column replication $r$."
  - `fig:column-replication-ydf`: "Table~\ref{tab:column-replication-ydf} as charts: mean (top) and per-dataset (bottom) $\Delta$ test accuracy vs.\ \texttt{single}."
- **Status:** current (main body, Accuracy section), bodies pushed to Overleaf 2026-11-04 (e57222c). Metric: test ROC AUC. Replaces the CC18 fold-0 bodies of 2026-11-03 (inputs now in `outdated/tab-column-replication-ydf-cc18/`)
- **Profile:** `kdd28_ydf.colrep_distacc_spo`, `kdd28_ydf.distacc_spo`, `kdd28_ydf.distacc_single_spo` (`benchmarks/src/distributed_spo_gbt/params/kdd28_params.json`)
- **Settings:** effective values from the profile (`params.py show <arm>`), checked against the CSV stamp columns
  by the generator where the CSV has them
  - `kdd28_ydf.colrep_distacc_spo`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=2, spo_max_projections=6000, spo_density=2, spo_min_projections=1, split_budget=true, local_projection_count=false, column_replication=0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1, max_unique_discretized=1, workers=4, machine=m7i.metal-24xl in-process, folds=5 CV folds (chronological sets: 1 holdout), seeds=1, force_numerical=true, profile=kdd28_ydf.colrep_distacc_spo
  - `kdd28_ydf.distacc_spo`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=2, spo_max_projections=6000, spo_density=2, spo_min_projections=1, split_budget=true, local_projection_count=false, column_replication=0, max_unique_discretized=1, workers=1 4 8 16 32, machine=m7i.metal-24xl in-process, folds=5 CV folds (chronological sets: 1 holdout), seeds=1, force_numerical=true, profile=kdd28_ydf.distacc_spo
  - `kdd28_ydf.distacc_single_spo`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=2, spo_max_projections=6000, spo_density=2, spo_min_projections=1, validation_set_ratio=0, early_stopping=NONE, folds=5 CV folds (chronological sets: 1 holdout), seeds=1, include_axis_aligned=false, num_candidate_attributes=-1, profile=kdd28_ydf.distacc_single_spo
- **Eval suite:** `kdd28_tabarena33` (the datasets and folds of tab:dist-accuracy-ydf). `colrep_distacc/` stamped, `params.py check-eval --suite kdd28_tabarena33` clean (1533 rows). The reused distacc rows are unstamped
- **Machine:** `colrep_distacc/`: EC2 c8i.metal-96xl (Xeon 6975P-C), SMT and turbo on, W = 4 in-process workers, 8 threads. r = 0 and single reuse the 2026-11-03 m7i.metal-24xl distacc rows (SMT off)
- **Date:** 2026-11-04 (r = 0 and single rows: 2026-11-03)
- **Code:** driver `c3b01f56-dirty`, binaries built at e3279f20 (`bin_sha` 168e934e2d19)
- **Generator:** run from the repo root, writes into the paper dir (`--out` to redirect)

```
python3 benchmarks/src/distributed_spo_gbt/make_column_replication_table.py benchmarks/results/kdd28/accuracy/tab-column-replication-ydf/colrep_distacc \
  --out paper/distributed_spo_gbt_kdd28/tables/table_column_replication_ydf_body.tex \
  "IN-PROCESS rerun, c8i.metal-96xl SMT+turbo, W=4; r=0 and single = 2026-11-03 m7i distacc rows" --metric test_auc \
  --single_csv benchmarks/results/kdd28/accuracy/tab-dist-accuracy-ydf/distacc/results.csv --single_filter arm=single:SPO --single_arms distacc_single_spo \
  --dist_arm colrep_distacc_spo --r0_csv benchmarks/results/kdd28/accuracy/tab-dist-accuracy-ydf/distacc/results.csv --r0_filter arm=4:SPO --r0_arm distacc_spo \
  --profile kdd28_ydf --eval_suite kdd28_tabarena33
python3 benchmarks/src/distributed_spo_gbt/make_column_replication_fig.py --csv benchmarks/results/kdd28/accuracy/tab-column-replication-ydf/colrep_distacc/results.csv --suffix _ydf --metric test_auc \
  --dist_arm colrep_distacc_spo --single_csv benchmarks/results/kdd28/accuracy/tab-dist-accuracy-ydf/distacc/results.csv --single_filter arm=single:SPO \
  --single_arms distacc_single_spo --r0_csv benchmarks/results/kdd28/accuracy/tab-dist-accuracy-ydf/distacc/results.csv --r0_filter arm=4:SPO --r0_arm distacc_spo \
  --profile kdd28_ydf --eval_suite kdd28_tabarena33
```
- **Inputs:** `inputs.txt` (one repo-relative path per line)
  - `benchmarks/results/kdd28/accuracy/tab-column-replication-ydf/colrep_distacc/results.csv`
  - `benchmarks/results/kdd28/accuracy/tab-dist-accuracy-ydf/distacc/results.csv`
- **Raw cells:** box `~/results/2026-11-04_colrep_distacc/` (EC2 ariel-dist-accuracy-1), mirrored with logs and models to the gitignored `ariel_notes/results/2026-11-04_colrep_distacc/box/colrep_distacc/`. REPLICABILITY §J

`colrep_distacc/results.csv`: 1530 cells (153 dataset-folds x r = 0.1..1.0) plus 3 r = 0
checks (blood-transfusion fold 1, diabetes fold 2, Fitness_Club fold 0), all rc = 0. The checks give the same test AUC as
the distacc W = 4 SPO rows (box: models byte-identical). The generator drops them and uses the 153 distacc W = 4 rows as
r = 0. Consistency (2026-11-04): the single row (.8301) and the r = 0 row (.8264) equal the single-machine SPO and W = 4
SPO means of tab:dist-accuracy-ydf.
