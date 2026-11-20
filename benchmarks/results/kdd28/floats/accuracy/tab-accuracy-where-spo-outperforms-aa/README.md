# tab:accuracy-where-spo-outperforms-aa

Results behind the KDD28 floats below. Rebuild the bodies with the commands. Index and rules: `benchmarks/results/kdd28/README.md`.

- **Labels:** `tab:accuracy-where-spo-outperforms-aa`, `fig:accuracy-where-spo-outperforms-aa`, `fig:column-replication-where-spo-outperforms-aa` (section
  `sec:accuracy-where-spo-outperforms-aa`, "Accuracy on Datasets Where SPO Outperforms AA"). Added 2026-11-06 (Overleaf 8c3e81d). Metric: test ROC AUC
- **Status:** current (main body)
- **What:** no new runs. Row subsets of the SPAA paper eval and the wide dataset eval, 41 datasets. `select_spo_advantage.py`
  keeps the 11 where single-machine SPO-GBT minus single-machine AA-GBT test AUC (fold means) > 0.005
  (`derived/spo_advantage_datasets.txt`, edge per dataset). Selection uses the same test folds the table reports. Of the 612
  rows of `derived/distacc/results.csv`, 540 come from `runs/2026-11-03_distacc/` and 72 (epsilon, philippine) from
  `runs/2026-11-04_wide_eval/`
- **Runs:** `runs/2026-11-03_distacc/`, `runs/2026-11-04_distacc_aaspo/`, `runs/2026-11-04_colrep_distacc/`, `runs/2026-11-04_wide_eval/`.
  Run facts: their READMEs
- **Profile:** `kdd28_ydf.distacc_single_spo`, `kdd28_ydf.distacc_single_aa`, `kdd28_ydf.distacc_spo`, `kdd28_ydf.distacc_aa`,
  `kdd28_ydf.distacc_single_aa_spo`, `kdd28_ydf.distacc_single_spo_harness_exact300`, `kdd28_ydf.colrep_distacc_spo`
  (`benchmarks/src/distributed_spo_gbt/params/kdd28_params.json`)
- **Settings:** effective values from the profile (`params.py show <arm>`), checked against the CSV stamp columns. Threads 4
  single, 2 per worker
- **Eval suite:** `kdd28_spo_advantage` (union of 9 `kdd28_tabarena33` + 2 `kdd28_wide_datasets` sets). Rows keep their member
  suite's stamp
- **Machine:** as in the run READMEs (2026-11-03_distacc: m7i.metal-24xl, the others: c8i.metal-96xl), distributed arms in-process
- **Date:** runs 2026-11-03 to 2026-11-06, selection 2026-11-06
- **Code:** as in the run READMEs
- **Generator:** the commands below, run from the repo root, write `_where_spo_outperforms_aa` bodies into the paper dir

```
E=benchmarks/results/kdd28/floats/accuracy/tab-accuracy-where-spo-outperforms-aa
python3 $E/select_spo_advantage.py
python3 benchmarks/src/distributed_spo_gbt/paper/tables/make_paper_tables.py --accuracy_csv $E/derived/distacc/results.csv --dataset_shapes $E/derived/spo_advantage_shapes.csv \
  --accuracy_note "IN-PROCESS reruns (workers = threads of one process): SPAA paper eval rows on AWS m7i.metal-24xl, wide dataset eval rows on c8i.metal-96xl; datasets where single SPO-GBT beats AA-GBT by >0.005 AUC" \
  --arms distacc_single_spo,distacc_single_aa,distacc_spo,distacc_aa --profile kdd28_ydf --suffix _where_spo_outperforms_aa --purple --std_below \
  --extra aaspo=$E/derived/distacc_aaspo/results.csv,exact300=$E/derived/distacc_exact300/results.csv --extra_first --eval_suite kdd28_spo_advantage \
  --std_below --shape_below --group_header --no_figure --time_row  # table (2026-11-06), figure: drop these flags
python3 benchmarks/src/distributed_spo_gbt/paper/figures/make_column_replication_fig.py --csv $E/derived/colrep_distacc/results.csv --suffix _where_spo_outperforms_aa --metric test_auc \
  --dist_arm colrep_distacc_spo --single_csv $E/derived/distacc/results.csv --single_filter arm=single:SPO \
  --single_arms distacc_single_spo --r0_csv $E/derived/distacc/results.csv --r0_filter arm=4:SPO --r0_arm distacc_spo \
  --profile kdd28_ydf --eval_suite kdd28_spo_advantage
```
- **Inputs:** `inputs.txt`. The shapes of the 2 wide datasets come from the over-100-columns float's
  `derived/wide_shapes.csv` (marked `# cross-float` there)
- **Raw cells:** as in the run READMEs

Every `derived/` file was rebuilt byte-identically by `select_spo_advantage.py` on 2026-11-20. Expected warnings: inputs mix
the member suites, 45 SPAA rows predate the `oblique_also_axis_aligned` stamp.
