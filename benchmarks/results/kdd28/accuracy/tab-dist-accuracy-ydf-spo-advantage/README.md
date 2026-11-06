# tab:dist-accuracy-ydf-spo-advantage

Results behind the KDD28 floats below. Rebuild the bodies with the commands. Index and rules: `benchmarks/results/kdd28/README.md`.

- **Labels:** `tab:dist-accuracy-ydf-spo-advantage`, `fig:dist-accuracy-mean-ydf-spo-advantage`, `fig:column-replication-ydf-spo-advantage` (section
  `sec:accuracy-spo-advantage`, "Accuracy on Datasets Where SPO Outperforms AA"). Added 2026-11-06 (Overleaf 8c3e81d). Metric: test ROC AUC
- **What:** no new runs. Row subsets of the SPAA paper eval (`tab-dist-accuracy-ydf/`, `tab-column-replication-ydf/`) and the
  wide dataset eval (`tab-dist-accuracy-ydf-wide/wide_eval/`), 41 datasets. `select_spo_advantage.py` keeps the 11 where single-machine
  SPO-GBT minus single-machine AA-GBT test AUC (fold means) > 0.005 (`spo_advantage_datasets.txt`, edge per dataset). Selection uses the
  same test folds the table reports.
- **Eval suite:** `kdd28_spo_advantage` (union of 9 `kdd28_tabarena33` + 2 `kdd28_wide_datasets` sets). Profile, machine, code, raw
  cells: as in the source READMEs.
- **Commands** (repo root):

```
E=benchmarks/results/kdd28/accuracy/tab-dist-accuracy-ydf-spo-advantage
python3 $E/select_spo_advantage.py
python3 benchmarks/src/distributed_spo_gbt/make_paper_tables.py --accuracy_csv $E/distacc/results.csv --dataset_shapes $E/spo_advantage_shapes.csv \
  --accuracy_note "IN-PROCESS reruns (workers = threads of one process): SPAA paper eval rows on AWS m7i.metal-24xl, wide dataset eval rows on c8i.metal-96xl; datasets where single SPO-GBT beats AA-GBT by >0.005 AUC" \
  --arms distacc_single_spo,distacc_single_aa,distacc_spo,distacc_aa --profile kdd28_ydf --suffix _ydf_spo_advantage --purple --std_below \
  --extra aaspo=$E/distacc_aaspo/results.csv,exact300=$E/distacc_exact300/results.csv --extra_first --eval_suite kdd28_spo_advantage \
  --std_below --shape_below --group_header --no_figure --time_row  # table (2026-11-06), figure: drop these flags
python3 benchmarks/src/distributed_spo_gbt/make_column_replication_fig.py --csv $E/colrep_distacc/results.csv --suffix _ydf_spo_advantage --metric test_auc \
  --dist_arm colrep_distacc_spo --single_csv $E/distacc/results.csv --single_filter arm=single:SPO \
  --single_arms distacc_single_spo --r0_csv $E/distacc/results.csv --r0_filter arm=4:SPO --r0_arm distacc_spo \
  --profile kdd28_ydf --eval_suite kdd28_spo_advantage
```

Expected warnings: inputs mix the member suites, 45 SPAA rows predate the `oblique_also_axis_aligned` stamp.

