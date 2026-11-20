# runs/2026-11-01_colrep_vertex_split_budget

Measurement, stored once. Floats read it through their `inputs.txt`. Index and rules: `benchmarks/results/kdd28/README.md`.

- **Files:** `results.csv` (1870 rows, 34 CC18 datasets, r = 0..1, seeds 1-5), `launcher.log`, `summary_by_dataset.{csv,md}`,
  `summary_pooled.csv`. Old place: the `table2/fullfix_splitbudget/` part of the 2026-11-01 fixed reruns
  (`accuracy_multi_thread/fixed_reruns/layout.py`)
- **Profile:** `table2.dist_split` (`benchmarks/src/distributed_spo_gbt/params/kdd28_params.json`)
- **Settings:** effective values (`params.py show <arm>`)
  - `table2.dist_split`: trees=10, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=1, spo_max_projections=2147483647, spo_density=1.5, spo_min_projections=1, split_budget=true, local_projection_count=false, column_replication=0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1, max_unique_discretized=1, workers=4, machine=n2-highmem-4, folds=CC18 fold 0, seeds=1 2 3 4 5, profile=table2.dist_split
- **Eval suite:** `kdd28_cc18_fold0` (34 CC18, fold 0, seeds 1-5)
- **Machine:** Vertex AI n2-highmem-4, W = 4. SMT/turbo: unknown, not recorded
- **Date:** 2026-11-01
- **Code:** dgbt-spo-full-fixes 10509539 + `--split_budget` (Vertex run colrep-fullparity_fullparity20261101)
- **Raw cells:** these files. GCS URIs not recorded
- **Read by:** tab:column-replication, fig:column-replication
