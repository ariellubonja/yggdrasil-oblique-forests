# runs/2026-11-01_cross_network

Measurement, stored once. Floats read it through their `inputs.txt`. Index and rules: `benchmarks/results/kdd28/README.md`.

- **Files:** `table3_fullfix/g0..g3/` (`results.csv` + `launcher_g*.log`, 68 rows, seed 1) and `table3_fullfix_seeds/g0..g3/`
  (`results.csv`, 340 rows, seeds 1-5). One run in 4 launcher groups, as the float README treats it.
  The 68 seed-1 rows are stored twice: `table3_fullfix/` equals the seed-1 rows of `table3_fullfix_seeds/` on all 30 columns
  (checked 2026-11-20). Both stay: the launcher logs are only in `table3_fullfix/` and `layout.py` maps both dirs. Subdir names carry the
  old table number and are kept (files are only moved)
- **Profile:** `table3` (`benchmarks/src/distributed_spo_gbt/params/kdd28_params.json`)
- **Settings:** trees=300, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=1, spo_max_projections=2147483647,
  spo_density=1.5, spo_min_projections=1, split_budget=false, local_projection_count=true, max_unique_discretized=1, workers=4,
  machine=n2-highmem-4, folds=CC18 fold 0, seeds=1 2 3 4 5 (`params.py show table3`)
- **Eval suite:** `kdd28_cc18_fold0` (34 CC18 binary, fold 0, seeds 1-5, test accuracy)
- **Machine:** Vertex AI n2-highmem-4, W = 4 (`machine` column). SMT/turbo: unknown, not recorded
- **Date:** 2026-11-01
- **Code:** fix-disc-colcomms 1ad2062f (image `train-ydf:colship-fullfix-1ad2062f`, `git` column)
- **Raw cells:** Vertex runs `colship-g*_fullfix20261101`. GCS URIs not recorded. REPLICABILITY §G
- **Read by:** tab:cross-network, fig:cross-network, through `floats/runtime/tab-cross-network/derived/table3_results_5seed.csv`
  (`accuracy_multi_thread/fixed_reruns/build_paper_inputs.py`, rebuilt byte-identically 2026-11-20)
