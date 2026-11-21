# tab:accuracy-on-datasets-over-100-columns

Results behind the KDD28 float(s) below. Rebuild the bodies with the commands. Index and rules:
`benchmarks/results/kdd28/README.md`.

- **Labels:** `tab:accuracy-on-datasets-over-100-columns`, `fig:accuracy-on-datasets-over-100-columns`, `fig:column-replication-over-100-columns` (tab:column-replication-over-100-columns dropped from the paper 2026-11-20, user) (section `sec:accuracy-on-datasets-over-100-columns`, "Accuracy on Datasets with >=100 Columns")
- **Status:** current (appendix, moved 2026-11-20), added 2026-11-04. Metric: test ROC AUC
- **What (union, user directive 2026-11-05):** 14 datasets, eval suite `kdd28_wide_union` (`params/eval_suites.json`). (a) The 6
  SPAA paper eval datasets with more than 100 features: APSFailure (170), Bioresponse (1776), ecom-offers (119),
  homecredit-default (696), homesite-insurance (299), kddcup09_appetency (212). Row subsets made by `filter_wide.py`:
  `derived/distacc/` of `runs/2026-11-03_distacc/`, `derived/distacc_aaspo/` of `runs/2026-11-04_distacc_aaspo/`,
  `derived/colrep_distacc/` of `runs/2026-11-04_colrep_distacc/`. `derived/distacc/` was recut on 2026-11-20 from today's run
  CSV: the same 216 rows in the same order as the first cut, plus the `eval_suite,data_sha,metric_impl` stamps that
  `params.py backfill-eval` (0e34d94e) added to the run CSV after it (the 216 x 27 older cells are equal, the regenerated
  bodies are byte-identical). (b) The wide dataset eval `kdd28_wide_datasets` (8 sets,
  F >= 100): epsilon, santander_transaction, hcdr_main, dota2, vehicle_sensIT, guillermo, philippine, christine, in
  `runs/2026-11-04_wide_eval/`. YouTube-8M dropped (too slow, see the float's footnote). `union_wide.py` concatenates (a) + (b)
  into `derived/union/<table>/results.csv`, `make_wide_shapes.py` writes `derived/wide_shapes.csv` (rows, cols) for the table
- **Runs:** `runs/2026-11-03_distacc/`, `runs/2026-11-04_distacc_aaspo/`,
  `runs/2026-11-04_colrep_distacc/`, `runs/2026-11-04_wide_eval/`. Run facts: their READMEs
- **Profile:** `kdd28_ydf.distacc_single_spo`, `kdd28_ydf.distacc_single_aa`, `kdd28_ydf.distacc_spo`, `kdd28_ydf.distacc_aa`,
  `kdd28_ydf.distacc_single_aa_spo`, `kdd28_ydf.distacc_single_spo_harness_exact300`, `kdd28_ydf.colrep_distacc_spo`
- **Settings:** effective values from the profile (`params.py show <arm>`), checked against the CSV stamp columns. Threads 4
  single, 2 per worker in both parts
- **Eval suite:** `kdd28_wide_union` = the 6 `kdd28_tabarena33` sets above + all of `kdd28_wide_datasets` (5 CV folds, seed 0
  fold split; Epsilon and the 3 chronological sets: 1 holdout; seed 1; no validation, no early stopping). Rows keep their
  member suite's stamp. Check: `params.py check-eval <csv> --suite kdd28_wide_union`
- **Machine:** (a) as in the run READMEs (distacc: m7i.metal-24xl, aaspo/exact300/colrep: c8i.metal-96xl). (b)
  c8i.metal-96xl SMT+turbo (Xeon 6975P-C), distributed arms in-process (MULTI_THREAD workers)
- **Date:** (a) 2026-11-03/04, (b) 2026-11-04 to 2026-11-06 (UTC)
- **Code:** (a) as in the run READMEs. (b) driver and binaries `0416519f-dirty` (`ariel_notes/results/2026-11-04_wide_eval/ariel-ydf-1/code.diff`),
  folds by `benchmarks/src/distributed_spo_gbt/rebuild_wide_data.py` (sha256 pins `benchmarks/data/kdd28_wide_datasets_manifest.sha256`)
- **Generator:** the commands below, run from the repo root, write into the paper dir
- **Raw cells:** (a) as in the run READMEs. (b) EC2 ariel-ydf-1 (i-0f90f7764082bf21a) and ariel-ydf-7 (i-04668dc60810fb749)
  `~/results/2026-11-04_wide_eval/`, mirrored to `ariel_notes/results/2026-11-04_wide_eval/{ariel-ydf-1,ariel-ydf-7}`;
  shard scripts `ariel_notes/results/2026-11-05_wide_eval_shards/`
- **Commands:** run from the repo root, write `_over_100_columns` bodies into the paper dir

```
D=benchmarks/results/kdd28/floats/accuracy/tab-accuracy-on-datasets-over-100-columns; R=benchmarks/results/kdd28/runs; U=$D/derived/union
SH=benchmarks/results/runtime/speedup_map_by_dataset/analysis/per_dataset_summary.csv
python3 $D/filter_wide.py $SH $R/2026-11-04_distacc_aaspo/distacc_aaspo.csv $D/derived/distacc_aaspo/distacc_aaspo.csv
python3 $D/filter_wide.py $SH $R/2026-11-04_distacc_aaspo/distacc_exact300.csv $D/derived/distacc_aaspo/distacc_exact300.csv
python3 $D/filter_wide.py $SH $R/2026-11-04_colrep_distacc/results.csv $D/derived/colrep_distacc/results.csv
python3 $D/filter_wide.py $SH $R/2026-11-03_distacc/results.csv $D/derived/distacc/results.csv
python3 $D/union_wide.py && python3 $D/make_wide_shapes.py
python3 benchmarks/src/distributed_spo_gbt/paper/tables/make_paper_tables.py --accuracy_csv $U/distacc/results.csv --dataset_shapes $D/derived/wide_shapes.csv \
  --accuracy_note "IN-PROCESS reruns (workers = threads of one process): SPAA paper eval rows on AWS m7i.metal-24xl, wide dataset eval rows on c8i.metal-96xl; datasets with F>=100" \
  --arms distacc_single_spo,distacc_single_aa,distacc_spo,distacc_aa --profile kdd28_ydf --suffix _over_100_columns --purple \
  --extra aaspo=$U/distacc_aaspo/results.csv,exact300=$U/distacc_exact300/results.csv --extra_first --eval_suite kdd28_wide_union \
  --std_below --shape_below --group_header --no_figure --time_row --bold_datasets benchmarks/results/kdd28/floats/accuracy/tab-accuracy-where-spo-outperforms-aa/derived/spo_advantage_datasets.txt  # table only (2026-11-06), the figure needs the command without the last 4 flags
python3 benchmarks/src/distributed_spo_gbt/paper/figures/make_column_replication_fig.py --csv $U/colrep_distacc/results.csv --suffix _over_100_columns --metric test_auc \
  --dist_arm colrep_distacc_spo --single_csv $U/distacc/results.csv --single_filter arm=single:SPO \
  --single_arms distacc_single_spo --r0_csv $U/distacc/results.csv --r0_filter arm=4:SPO --r0_arm distacc_spo \
  --profile kdd28_ydf --eval_suite kdd28_wide_union
```

Every `derived/` file was rebuilt byte-identically by these scripts on 2026-11-20. Since 2026-11-20 `--bold_datasets` bolds the
names of the datasets where single-machine SPO-GBT beats AA-GBT (philippine, epsilon here). Only those 2 name cells changed. The
section is now subsection D.1 of sec:accuracy-on-spaa-datasets (Overleaf dfd8cf8). Expected warnings: the inputs mix the member suites `kdd28_tabarena33` and `kdd28_wide_datasets`,
and 18 SPAA aaspo rows predate the `oblique_also_axis_aligned` stamp ("empty, not recorded").
- **Inputs:** `inputs.txt`

