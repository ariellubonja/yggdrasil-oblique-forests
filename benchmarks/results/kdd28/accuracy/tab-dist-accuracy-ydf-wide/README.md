# tab:dist-accuracy-ydf-wide

Results behind the KDD28 float(s) below. Rebuild the bodies with the commands. Index and rules:
`benchmarks/results/kdd28/README.md`.

- **Labels:** `tab:dist-accuracy-ydf-wide`, `fig:dist-accuracy-mean-ydf-wide`, `fig:column-replication-ydf-wide`,
  `tab:column-replication-ydf-wide` (section `sec:accuracy-wide`, "Accuracy on Datasets with >100 Columns")
- **Status:** current (main body), added 2026-11-04. Metric: test ROC AUC
- **What:** the floats of `tab-dist-accuracy-ydf/` and `tab-column-replication-ydf/` restricted to the 6 datasets with
  more than 100 features: APSFailure (170), Bioresponse (1776), ecom-offers (119), homecredit-default (696),
  homesite-insurance (299), kddcup09_appetency (212). No new runs. The CSVs here are row subsets of those inputs,
  made by `filter_wide.py` with the feature counts of `make_paper_tables.SHAPES_CSV`
- **Profile, settings, machine, date, code, raw cells:** as in `../tab-dist-accuracy-ydf/README.md` and
  `../tab-column-replication-ydf/README.md`
- **Commands:** run from the repo root, write `_ydf_wide` bodies into the paper dir

```
D=benchmarks/results/kdd28/accuracy/tab-dist-accuracy-ydf-wide; A=benchmarks/results/kdd28/accuracy
SH=benchmarks/results/runtime/speedup_map_by_dataset/analysis/per_dataset_summary.csv
python3 $D/filter_wide.py $SH $A/tab-dist-accuracy-ydf/distacc/results.csv $D/distacc/results.csv
python3 $D/filter_wide.py $SH $A/tab-dist-accuracy-ydf/distacc_aaspo/distacc_aaspo.csv $D/distacc_aaspo/distacc_aaspo.csv
python3 $D/filter_wide.py $SH $A/tab-dist-accuracy-ydf/distacc_aaspo/distacc_exact300.csv $D/distacc_aaspo/distacc_exact300.csv
python3 $D/filter_wide.py $SH $A/tab-column-replication-ydf/colrep_distacc/results.csv $D/colrep_distacc/results.csv
python3 benchmarks/src/distributed_spo_gbt/make_paper_tables.py --accuracy_csv $D/distacc/results.csv \
  --accuracy_note "IN-PROCESS rerun on AWS m7i.metal-24xl (Xeon 8488C; workers = threads of one process), datasets with F>100" \
  --arms distacc_single_spo,distacc_single_aa,distacc_spo,distacc_aa --profile kdd28_ydf --suffix _ydf_wide --purple \
  --extra aaspo=$D/distacc_aaspo/distacc_aaspo.csv,exact300=$D/distacc_aaspo/distacc_exact300.csv --extra_first
python3 benchmarks/src/distributed_spo_gbt/make_column_replication_table.py $D/colrep_distacc \
  --out paper/distributed_spo_gbt_kdd28/tables/table_column_replication_ydf_wide_body.tex \
  "IN-PROCESS rerun, c8i.metal-96xl SMT+turbo, W=4; r=0 and single = 2026-11-03 m7i distacc rows; datasets with F>100" --metric test_auc \
  --single_csv $D/distacc/results.csv --single_filter arm=single:SPO --single_arms distacc_single_spo \
  --dist_arm colrep_distacc_spo --r0_csv $D/distacc/results.csv --r0_filter arm=4:SPO --r0_arm distacc_spo \
  --profile kdd28_ydf --eval_suite kdd28_tabarena33
python3 benchmarks/src/distributed_spo_gbt/make_column_replication_fig.py --csv $D/colrep_distacc/results.csv --suffix _ydf_wide --metric test_auc \
  --dist_arm colrep_distacc_spo --single_csv $D/distacc/results.csv --single_filter arm=single:SPO \
  --single_arms distacc_single_spo --r0_csv $D/distacc/results.csv --r0_filter arm=4:SPO --r0_arm distacc_spo \
  --profile kdd28_ydf --eval_suite kdd28_tabarena33
```

The generators warn that 27 of the 33 `kdd28_tabarena33` datasets are missing. That is the intended filter.
- **Inputs:** `inputs.txt`
