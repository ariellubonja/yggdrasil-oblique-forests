# tab:cross-network

Results behind the KDD28 float(s) below. Rebuild the body with the generator command. Index and rules:
`benchmarks/results/kdd28/README.md`.

- **Labels:** `tab:cross-network`, `fig:cross-network`
  - `tab:cross-network`: "Tree-training time, communication and test accuracy of in-worker vs.\ cross-network projections per CC18 dataset."
  - `fig:cross-network`: "Table~\ref{tab:cross-network} as charts: (a) speedup vs.\ dataset size, (b) time breakdown, (c) test accuracy, means over 5 seeds."
- **Status:** current main-body float on the OLD profile `table3` (P = F, density 1.5, no split budget). The kdd28_ydf rerun (arm `kdd28_ydf.cross_network_ship`) has not run yet (REPLICABILITY §J)
- **Profile:** `table3` (`benchmarks/src/distributed_spo_gbt/params/kdd28_params.json`)
- **Settings:** effective values from the profile (`params.py show <arm>`), checked against the CSV stamp columns
  by the generator where the CSV has them
  - `table3`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=1, spo_max_projections=2147483647, spo_density=1.5, spo_min_projections=1, split_budget=false, local_projection_count=true, max_unique_discretized=1, workers=4, machine=n2-highmem-4, folds=CC18 fold 0, seeds=1 2 3 4 5, profile=table3
- **Eval suite:** `kdd28_cc18_fold0` (34 CC18 binary, fold 0, seeds 1-5, test accuracy)
- **Machine:** Vertex AI n2-highmem-4, W = 4 (`machine` column). SMT/turbo: unknown, not recorded
- **Date:** 2026-11-01
- **Code:** fix-disc-colcomms 1ad2062f (image `train-ydf:colship-fullfix-1ad2062f`, `git` column)
- **Generator:** run from the repo root, writes into the paper dir (`--out` to redirect)

```
python3 benchmarks/src/distributed_spo_gbt/make_cross_network_table.py   # default src = benchmarks/results/kdd28/runtime/tab-cross-network/table3_results_5seed.csv
python3 benchmarks/src/distributed_spo_gbt/make_cross_network_fig.py
```
- **Inputs:** `inputs.txt` (one repo-relative path per line)
  - `benchmarks/results/kdd28/runtime/tab-cross-network/table3_results_5seed.csv`
- **Raw cells:** `benchmarks/results/kdd28/runtime/tab-cross-network/raw/table3_fullfix{,_seeds}/g0..g3/` (Vertex runs `colship-g*_fullfix20261101`, built into the 5-seed CSV by `fixed_reruns/build_paper_inputs.py`). GCS URIs not recorded. REPLICABILITY §G

Regenerated 2026-11-04: table, figure and `cross_network/*.dat` equal the Overleaf copies in every non-`%` line.
