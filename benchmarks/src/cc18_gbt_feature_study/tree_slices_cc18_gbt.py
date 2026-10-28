#!/usr/bin/env python3
"""Per-depth feature-choice stats at single boosting iterations: the k-th tree
(k in --trees, 1-based) of every fold, pooled per dataset (10 trees), then
medians across datasets. Reuses the metric helpers of aggregate_cc18_gbt.py.
Usage: tree_slices_cc18_gbt.py --runs DIR --out DIR --data DIR [--trees 100,200,300]
"""
import argparse, glob, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from aggregate_cc18_gbt import ancestor_reuse, norm_entropy, sibling_same, topk_share


def per_depth(df, F):
    rows = []
    for d, g in df.groupby("depth"):
        counts = g["attribute"].value_counts()
        ne, eff = norm_entropy(counts.values, F)
        td = g.groupby("tree")["attribute"].agg(["size", "nunique"])
        multi = td[td["size"] >= 2]  # a 1-node level is trivially oblivious
        rows.append(dict(depth=d, n_nodes=len(g), n_distinct=counts.size,
                         frac_features_used=counts.size / F,
                         top1_share=topk_share(counts.values, 1),
                         top3_share=topk_share(counts.values, 3),
                         norm_entropy=ne, eff_num_features=eff,
                         reuse_rate=g["reuse"].mean(),
                         within_tree_distinct_ratio=(multi["nunique"] / multi["size"]).mean() if len(multi) else np.nan,
                         within_tree_oblivious=(multi["nunique"] == 1).mean() if len(multi) else np.nan))
    p = pd.DataFrame(rows).set_index("depth")
    p["sibling_same_feature"] = sibling_same(df).reindex(p.index)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", required=True, help="cc18_binary_csv dir (for F)")
    ap.add_argument("--trees", default="100,200,300", help="1-based tree numbers")
    a = ap.parse_args()
    trees = [int(t) for t in a.trees.split(",")]
    os.makedirs(a.out, exist_ok=True)
    by_ds = {}
    for p in sorted(glob.glob(f"{a.runs}/*/nodes.csv")):
        ds, fold = os.path.basename(os.path.dirname(p)).rsplit("__fold", 1)
        by_ds.setdefault(ds, []).append((int(fold), p))
    depth_rows, model_rows = [], []
    for ds, lst in by_ds.items():
        hdr = open(f"{a.data}/{ds}/repeat0_fold0_sample0_train.csv").readline().rstrip("\n").split(",")
        F = len(hdr) - 1
        dfs = {fold: pd.read_csv(p) for fold, p in lst}
        for k in trees:
            parts = []
            for fold, df in dfs.items():
                t = df[df["tree"] == k - 1].copy()
                assert len(t) > 0, (ds, fold, k)
                t["tree"] = fold  # re-key so the 10 folds' trees stay distinct
                parts.append(t)
            sl = pd.concat(parts, ignore_index=True)
            sl["reuse"] = ancestor_reuse(sl)
            pdp = per_depth(sl, F)
            pdp.insert(0, "dataset", ds); pdp.insert(1, "tree_k", k); pdp.insert(2, "F", F)
            depth_rows.append(pdp.reset_index())
            per_tree = sl.groupby("tree")["attribute"].agg(["size", "nunique"])
            roots = sl[sl["depth"] == 0]["attribute"].value_counts()
            model_rows.append(dict(dataset=ds, tree_k=k, F=F, n_trees=sl["tree"].nunique(),
                                   internal_nodes_per_tree=per_tree["size"].mean(),
                                   distinct_features_per_tree=per_tree["nunique"].mean(),
                                   n_distinct_roots=roots.size,
                                   root_top1_share=roots.iloc[0] / roots.sum(),
                                   reuse_rate=sl["reuse"].mean(),
                                   max_depth_reached=sl["depth"].max()))
        print(ds, flush=True)
    pdd = pd.concat(depth_rows)
    pm = pd.DataFrame(model_rows)
    pdd.to_csv(f"{a.out}/per_depth_tree_slices.csv", index=False)
    pm.to_csv(f"{a.out}/per_model_tree_slices.csv", index=False)
    cols = ["n_nodes", "n_distinct", "top1_share", "top3_share", "norm_entropy", "eff_num_features",
            "reuse_rate", "sibling_same_feature", "within_tree_oblivious", "within_tree_distinct_ratio"]
    pooled = pdd.groupby(["tree_k", "depth"])[cols].median()
    pooled.to_csv(f"{a.out}/tree_slices_pooled.csv")
    print("\n=== per-depth medians across datasets, by tree number (10 trees per dataset) ===")
    print(pooled.round(3).to_string())
    mcols = ["internal_nodes_per_tree", "distinct_features_per_tree", "n_distinct_roots", "root_top1_share", "reuse_rate"]
    print("\n=== per-tree medians across datasets, by tree number ===")
    print(pm.groupby("tree_k")[mcols].median().round(3).to_string())


if __name__ == "__main__":
    main()
