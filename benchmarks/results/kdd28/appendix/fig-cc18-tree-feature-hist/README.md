# fig:cc18-tree-feature-hist

Results behind the KDD28 float(s) below. Rebuild the body with the generator command. Index and rules:
`benchmarks/results/kdd28/README.md`.

- **Labels:** `fig:cc18-tree-feature-hist`
  - `fig:cc18-tree-feature-hist`: "Features split on by the first tree of a AA-GBT. Top: most Pareto-like. Bottom: least-pareto like. See animation \href{https://drive.google.com/file/d/1BLq98ab_YEgCxTeb6eLGhEEZw2BVsqCO/view?usp=dri..."
- **Status:** appendix
- **Profile:** `cc18_feature_study` (`benchmarks/src/distributed_spo_gbt/params/kdd28_params.json`)
- **Settings:** effective values from the profile (`params.py show <arm>`), checked against the CSV stamp columns
  by the generator where the CSV has them
  - `cc18_feature_study`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, num_candidate_attributes=-1, shrinkage=0.1, growing_strategy=Local, folds=0 1 2 3 4 5 6 7 8 9, seeds=1, profile=cc18_feature_study
- **Eval suite:** `kdd28_cc18_10fold` datasets (34 CC18 binary, sick skipped, 10 CV folds, seed 1), AA-GBT only
- **Machine:** EC2 box (icx build), single machine. SMT/turbo: unknown, not recorded
- **Date:** 2026-10-26
- **Code:** `train_oblique_forest` + `examples/dump_forest_nodes.cc` (commit: unknown, the study README names none)
- **Generator:** run from the repo root, writes into the paper dir (`--out` to redirect)

```
python3 benchmarks/src/cc18_gbt_feature_study/make_fig_tree_feature_hist.py --out paper/distributed_spo_gbt_kdd28 \
  --trees 1,44,86,129,172,215,257,300 --hsep 0.45cm --no_counts --k 6
```
- **Inputs:** `inputs.txt` (one repo-relative path per line)
  - `benchmarks/results/kdd28/appendix/fig-cc18-tree-feature-hist/cc18_gbt_feature_study/nodes_all.tar.gz`
  - `benchmarks/results/kdd28/appendix/fig-cc18-tree-feature-hist/cc18_gbt_feature_study/per_dataset.csv`
  - `benchmarks/data/cc18_binary_csv`
- **Raw cells:** `benchmarks/results/kdd28/appendix/fig-cc18-tree-feature-hist/cc18_gbt_feature_study/` (study README, per-depth CSVs, movie)

Regenerated 2026-11-04: body and 48 `.dat` files equal the Overleaf copies in every non-`%` line.
