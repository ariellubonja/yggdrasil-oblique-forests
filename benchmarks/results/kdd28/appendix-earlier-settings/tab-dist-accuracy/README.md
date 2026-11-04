# tab:dist-accuracy

Results behind the KDD28 float(s) below. Rebuild the body with the generator command. Index and rules:
`benchmarks/results/kdd28/README.md`.

- **Labels:** `tab:dist-accuracy`, `fig:dist-accuracy-mean`
  - `tab:dist-accuracy`: "Test AUC of single-machine vs. $W$-distributed GBT. Single-machine SPO-GBT samples $F$ projections per node, distributed SPO-GBT workers sample $F/W$ projections from their $F/W$ columns."
  - `fig:dist-accuracy-mean`: "Mean test AUC over the datasets of Table~\ref{tab:dist-accuracy}."
- **Status:** appendix (Accuracy results under the earlier settings), frozen
- **Profile:** `table8.single_spo`, `table8.single_aa`, `table8.dist_spo`, `table8.dist_aa` (`benchmarks/src/distributed_spo_gbt/params/kdd28_params.json`)
- **Settings:** effective values from the profile (`params.py show <arm>`), checked against the CSV stamp columns
  by the generator where the CSV has them
  - `table8.single_spo`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=1, spo_max_projections=2147483647, spo_density=1.5, spo_min_projections=1, folds=5 CV folds (chronological sets: 1 holdout), seeds=1, profile=table8.single_spo
  - `table8.single_aa`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, folds=5 CV folds (chronological sets: 1 holdout), seeds=1, num_candidate_attributes=-1, profile=table8.single_aa
  - `table8.dist_spo`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=1, spo_max_projections=2147483647, spo_density=1.5, spo_min_projections=1, split_budget=false, local_projection_count=true, column_replication=0, force_numerical=false, max_unique_discretized=1, workers=1 4 8 16 32, folds=5 CV folds (chronological sets: 1 holdout), seeds=123456, profile=table8.dist_spo
  - `table8.dist_aa`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, force_numerical=false, max_unique_discretized=1, workers=1 4 8 16 32, folds=5 CV folds (chronological sets: 1 holdout), seeds=123456, profile=table8.dist_aa
- **Eval suite:** `kdd28_tabarena33` folds (unstamped CSV: no eval columns)
- **Machine:** EC2 m7i.metal-24xl (Xeon 8488C), in-process. SMT/turbo: unknown, not recorded in the CSV
- **Date:** single rows 2026-10-12 (SPAA27 suite), distributed SPO 2026-11-01, distributed AA 2026-10-30
- **Code:** SPO dist. dgbt-spo-full-fixes 10509539, AA dist. `train_distributed_gbt` 8ddb03f7 (`raw/table8_rerun_20261030/PROVENANCE.txt`), single = `benchmarks/results/runtime/speedup_map_by_dataset/suite_results.csv` (methods spo_gbt_exact_hwy_fullproj, aa_gbt_exact)
- **Generator:** run from the repo root, writes into the paper dir (`--out` to redirect)

```
python3 benchmarks/src/distributed_spo_gbt/make_paper_tables.py --profile table8 --accuracy_csv benchmarks/results/kdd28/appendix-earlier-settings/tab-dist-accuracy/table8_results.csv \
  --accuracy_note "IN-PROCESS runs on AWS m7i.metal-24xl (Xeon 8488C; workers = threads of one process); SPO dist. = dgbt-spo-full-fixes 10509539"
```
- **Inputs:** `inputs.txt` (one repo-relative path per line)
  - `benchmarks/results/kdd28/appendix-earlier-settings/tab-dist-accuracy/table8_results.csv`
- **Raw cells:** `benchmarks/results/kdd28/appendix-earlier-settings/tab-dist-accuracy/raw/table8_fullfix/` (SPO), `raw/table8_rerun_20261030/` (AA, pre-7c0eebd3 but AA is unaffected), built by `fixed_reruns/build_paper_inputs.py` (reproduces `table8_results.csv` byte-identically, checked 2026-11-04)

Regenerated 2026-11-04: equal to the Overleaf copies in every non-`%` line.
