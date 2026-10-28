# CC18: which features does an axis-aligned GBT pick at each node? (2026-10-26)

Scripts: `benchmarks/src/cc18_gbt_feature_study/`; data here (moved from `ariel_notes/results/` 2026-10-28). Figure scripts also need the CC18 CSV headers: `benchmarks/data/download_cc18_datasets.py`.

**Setup.** YDF GBT via `train_oblique_forest` (built from source on the EC2 box, icx): 300 trees,
`max_depth 6` (YDF counts the root level, so 5 split levels, ≤ 31 internal nodes), shrinkage 0.1,
exact numeric splits, **every attribute is a candidate at every node** (`--num_candidate_attributes -1`),
min 5 examples, no validation split / early stopping, seed 1. 34 CC18 binary datasets (`sick` skipped,
manifest status "issue") × 10 CV folds = 340 models, 102k trees, 2.07 M internal nodes. Every split
node was dumped with the new `examples/dump_forest_nodes.cc` (tree, depth, parent, attribute,
condition type, threshold, node size, split score). Medians below are across the 34 datasets
(median F = 21 features, each dataset first averaged over its 10 folds). Independently verified
2026-10-26: tree structure of all 2.07 M nodes, brute-force recomputation of every metric on 8 models, re-aggregation.

## Findings

1. **Trees are far from full.** Median 19.3 internal nodes per tree (max 31). Depth 4 holds 40 % of all
   splits (share by depth 0..4: 5 / 10 / 17 / 27 / 40 %).
2. **The root rotates after the first ~30 iterations.** Median: the most common root feature heads 20 %
   of the 300 trees; 15 distinct root features per model. In trees 1-30 the top root feature has a 63 %
   share, in trees 271-300 only 30 %. Only 13/34 datasets keep the same favourite root early vs late.
3. **Below depth 1 the choice is nearly uniform over the feature set.** Top-1 feature share per depth
   0.20 → 0.16 → 0.14 → 0.13 → 0.13; normalised entropy 0.73 → 0.87 → 0.91 → 0.93 → 0.93;
   effective number of features 11 → 16 → 17 → 17 (of F = 21).
4. **Almost every feature gets used.** 24/34 datasets split on 100 % of their features in every fold
   (29/34 pooled over folds; 31/34 reach ≥ 90 % per fold). madelon uses 496 of its 500 per fold and
   all 500 pooled, 480 of them noise. Only the very wide sets leave many out: Internet-Ads 19 % per
   fold (39 % pooled), Bioresponse 41 % (70 %), cylinder-bands 85 %; kr-vs-kp, spambase, qsar-biodeg
   and nomao sit at 92-98 %. A single tree touches a median 10 distinct features (≈ F/2); 18 for F ≥ 500.
5. **Concentration scales with F.** Top-5 features take a median 46 % of all splits: 83 % for F ≤ 10,
   43 % for F 11-40, 30 % for 41-130, 8 % for F > 130. The most-split feature is also the highest-gain
   feature in only 11/34 datasets.
6. **Path reuse is modest and grows with depth.** 19 % of splits reuse a feature already on the root
   path (depth 1: 13 %, depth 4: 22 %); 48-57 % on the F ≤ 5 datasets, 0 % on kr-vs-kp (binary
   categoricals cannot be re-split).
7. **GBT trees are not oblivious.** Sibling internal nodes pick the same feature only 9-11 % of the
   time at every depth. Among trees with ≥ 2 internal nodes at a level, the whole level shares one
   feature in 10.6 % of trees at depth 1, 1.4 % at depth 2, < 0.5 % deeper. At depth 4 the number of
   distinct features is 77 % of the number of nodes. CatBoost-style symmetric trees are therefore a
   strong structural constraint relative to what depth-wise GBT chooses freely.
8. **Condition types.** 82 % numeric thresholds, 17 % categorical sets (`contains_bitmap`), 1 % boolean
   (`true_value`; YDF infers 0/1 columns as BOOLEAN, e.g. Internet-Ads). 11/34 datasets have any
   non-numeric split (9/34 with categorical sets; Internet-Ads and pc4 only boolean); kr-vs-kp and
   tic-tac-toe are 100 % categorical. `categorical_share` in the CSVs counts both kinds.

## By feature count (medians)

| F bucket | n | top-1 share | top-5 share | root top-1 share | distinct roots | features / tree | features used | path reuse |
|---|---|---|---|---|---|---|---|---|
| ≤ 10 | 9 | 0.22 | 0.83 | 0.42 | 6.6 | 6.5 | 1.00 | 0.46 |
| 11-40 | 18 | 0.12 | 0.43 | 0.19 | 17.5 | 10.9 | 1.00 | 0.18 |
| 41-130 | 4 | 0.07 | 0.30 | 0.15 | 32.8 | 14.4 | 0.96 | 0.16 |
| > 130 | 3 | 0.02 | 0.08 | 0.07 | 86.6 | 18.1 | 0.41 | 0.05 |

## By depth (median across datasets; IQR in `per_depth.csv`)

| depth | nodes (of 300 trees) | distinct feats | top-1 share | top-3 share | norm. entropy | eff. #feats | path reuse | sibling same feat | level all-same | distinct/nodes |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 300 | 15.4 | 0.20 | 0.51 | 0.73 | 11.0 | 0 | – | – | – |
| 1 | 581 | 20.7 | 0.16 | 0.35 | 0.87 | 16.3 | 0.13 | 0.11 | 0.11 | 0.95 |
| 2 | 1026 | 21.0 | 0.14 | 0.35 | 0.91 | 17.0 | 0.17 | 0.10 | 0.014 | 0.90 |
| 3 | 1618 | 21.0 | 0.13 | 0.34 | 0.93 | 17.1 | 0.18 | 0.09 | 0.004 | 0.85 |
| 4 | 2316 | 21.0 | 0.13 | 0.33 | 0.93 | 17.1 | 0.22 | 0.09 | 0.002 | 0.77 |

## Single iterations: the 100th, 200th and 300th tree

Same metrics on exactly the k-th tree of every fold, pooled per dataset (10 trees per dataset per
iteration), medians across the 34 datasets (`tree_slices_pooled.csv`, per-dataset values in
`per_depth_tree_slices.csv` / `per_model_tree_slices.csv`, script `benchmarks/src/cc18_gbt_feature_study/tree_slices_cc18_gbt.py`).
**Compare the three columns with each other only:** 10 trees give far fewer samples than 300, so
distinct counts, top-k shares and entropies are biased relative to the full-model table above
(10 roots cap "distinct" at 10 and the normalised entropy at ln 10 / ln F).

**Finding 9. Feature choice is stationary from tree 100 on.** Top-1 share, entropy, sibling agreement
and node counts are indistinguishable among trees 100, 200 and 300; only path reuse creeps up
(0.18 → 0.21 → 0.22) and distinct features per tree creeps down (10.5 → 9.9 → 9.3). At a fixed
iteration the 10 folds agree little on the root: a median 5-6 different root features, the most
common one shared by 3 folds of 10.

| depth | splits (10 trees) 100 / 200 / 300 | distinct 100 / 200 / 300 | top-1 share 100 / 200 / 300 | norm. entropy 100 / 200 / 300 | path reuse 100 / 200 / 300 | sibling same 100 / 200 / 300 |
|---|---|---|---|---|---|---|
| 0 | 10 / 10 / 10 | 5 / 6 / 6 | 0.30 / 0.30 / 0.30 | 0.51 / 0.54 / 0.54 | 0 / 0 / 0 | – |
| 1 | 20 / 19 / 20 | 11 / 11 / 9 | 0.24 / 0.25 / 0.22 | 0.71 / 0.71 / 0.68 | 0.11 / 0.16 / 0.15 | 0.05 / 0.11 / 0.11 |
| 2 | 34.5 / 33.5 / 34.5 | 15 / 13.5 / 14.5 | 0.16 / 0.16 / 0.17 | 0.84 / 0.80 / 0.79 | 0.19 / 0.18 / 0.18 | 0.08 / 0.10 / 0.10 |
| 3 | 52 / 51 / 53 | 16 / 15.5 / 15 | 0.16 / 0.17 / 0.16 | 0.85 / 0.84 / 0.82 | 0.19 / 0.19 / 0.21 | 0.12 / 0.10 / 0.13 |
| 4 | 70 / 70 / 73.5 | 17.5 / 16.5 / 17.5 | 0.16 / 0.16 / 0.15 | 0.87 / 0.85 / 0.87 | 0.20 / 0.24 / 0.25 | 0.11 / 0.08 / 0.11 |

Per tree (medians across datasets), tree 100 / 200 / 300: internal nodes 18.8 / 18.3 / 18.9; distinct
features per tree 10.5 / 9.9 / 9.3; distinct roots over the 10 folds 5 / 6 / 6; top root share
0.30 / 0.30 / 0.30; path reuse 0.18 / 0.21 / 0.22.

## Files

- `per_dataset.csv` – one row per dataset: F, n_train, nodes/tree, distinct features/tree, coverage,
  top-k shares (by count and gain), root statistics (all / first 30 / last 30 trees), path reuse,
  categorical share.
- `per_depth.csv` – one row per (dataset, depth); `per_depth_pooled.csv` – median/mean over datasets.
- `top_features.csv` – top 10 features per dataset with split-count share and gain share.
- `per_depth_tree_slices.csv`, `per_model_tree_slices.csv`, `tree_slices_pooled.csv` – the single-iteration slices (trees 100/200/300).
- `nodes_all.tar.gz` – raw dumps, one `nodes.csv` per model (117 MB extracted):
  `tree,depth,node_id,parent_id,attribute,attribute_name,cond_type,threshold,num_examples,split_score`.
- Paper figures (DGBT paper, `paper/distributed_spo_gbt_kdd27/figures/evaluation/`): `benchmarks/src/cc18_gbt_feature_study/make_fig_tree1_pareto.py` =
  `fig:cc18-tree1-pareto`, split share by feature RANK in trees 1/100/200/300 for 6 datasets sampled uniformly (seed 0, shown in
  increasing F: diabetes, breast-w, tic-tac-toe, dresses-sales, pc4, Bioresponse; `--select extremes` gives the 3 most /
  3 least Pareto-like by tree-1 normalised entropy instead); `benchmarks/src/cc18_gbt_feature_study/make_fig_tree_feature_hist.py` = `fig:cc18-tree-feature-hist`, the
  same counts on a fixed x-axis (features in header order, unused at 0, shared y range per dataset). Both pool the
  10 folds; both write body-only `.tex` + `.dat`, captions live in `main.tex`.
- `benchmarks/src/cc18_gbt_feature_study/dump_forest_nodes.cc` (+ `cc_binary` in `examples/BUILD`, uncommitted on the box),
  `benchmarks/src/cc18_gbt_feature_study/run_cc18_gbt.py` (driver, 8 jobs × 6 threads, 1232 job-seconds total, Bioresponse 44 s/fold),
  `benchmarks/src/cc18_gbt_feature_study/aggregate_cc18_gbt.py`.
- Exact paper body of `fig:cc18-tree-feature-hist` (byte-identical, checked 2026-10-28): `python3 benchmarks/src/cc18_gbt_feature_study/make_fig_tree_feature_hist.py --out <repo>/paper/distributed_spo_gbt_kdd27 --trees 1,44,86,129,172,215,257,300 --hsep 0.45cm --no_counts --k 6`
- `benchmarks/src/cc18_gbt_feature_study/make_fig_tree1_pareto.py --out <paper dir>` – first-tree feature Pareto figure (3 most / 3 least
  concentrated datasets by normalised entropy, pooled over the 10 folds): writes `figures/evaluation/cc18_tree1_pareto/*.dat`
  + `cc18_tree1_pareto_body.tex`; the `figure*`/caption/label live in `main.tex`. Replaced the per-depth table in the KDD27 paper on 2026-10-26.

## Reproduce

Box `ec2-100-54-250-152` (hostname changes on stop/start), repo `~/yggdrasil-oblique-forests`, work dir `~/featstudy`:

```bash
source /opt/intel/oneapi/setvars.sh && bazel build //examples:train_oblique_forest //examples:dump_forest_nodes
python3 ~/featstudy/run_cc18_gbt.py --folds 0-9 --jobs 8 --threads 6
~/gbt_venv/bin/python ~/featstudy/aggregate_cc18_gbt.py
```

Locally from the tarball: `python3 benchmarks/src/cc18_gbt_feature_study/aggregate_cc18_gbt.py --runs <extracted>/runs --out . --data benchmarks/data/cc18_binary_csv`.

## Caveats

- F counts every non-label column, constant ones included, so normalised entropy is a slight underestimate.
- Early/late root shares use the first and last 30 trees; "same root early vs late" compares the modes.
- Entropy, top-k shares and every per-depth metric are computed per fold (300 trees) and averaged over the
  10 folds; only `top_features.csv` and the `*_pooled` coverage column pool folds.
- Only the 34 binary CC18 tasks (the project's benchmark subset), not the full 72-task suite.
