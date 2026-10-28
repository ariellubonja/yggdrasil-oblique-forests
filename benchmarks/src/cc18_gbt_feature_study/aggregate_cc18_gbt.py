#!/usr/bin/env python3
"""Aggregate per-node feature dumps (featstudy/runs/*/nodes.csv) into
per-dataset, per-depth, and pooled summaries of which features GBT picks.
Usage: aggregate_cc18_gbt.py [--runs DIR] [--out DIR]
"""
import argparse, glob, os, re, sys
import numpy as np
import pandas as pd

HOME = os.path.expanduser("~")
DATA = f"{HOME}/yggdrasil-oblique-forests/benchmarks/data/cc18_binary_csv"
MAX_DEPTH = 6


def norm_entropy(counts, F):
    p = np.asarray(counts, dtype=float); p = p[p > 0]; p /= p.sum()
    H = -(p * np.log(p)).sum()
    return H / np.log(F) if F > 1 else 0.0, float(np.exp(H))


def topk_share(counts, k):
    c = np.sort(np.asarray(counts, dtype=float))[::-1]
    return c[:k].sum() / c.sum() if c.sum() else np.nan


def ancestor_reuse(df):
    """Per node: does its attribute already appear on the root path?"""
    key = df["tree"].astype(np.int64) * 1_000_000 + df["node_id"]
    pkey = df["tree"].astype(np.int64) * 1_000_000 + df["parent_id"]
    attr_of = pd.Series(df["attribute"].values, index=key.values)
    parent_of = pd.Series(pkey.values, index=key.values)
    reuse = np.zeros(len(df), dtype=bool)
    cur = pkey.copy()
    for _ in range(MAX_DEPTH):
        anc = cur.map(attr_of)
        reuse |= (anc.values == df["attribute"].values)
        cur = cur.map(parent_of)
        if cur.isna().all():
            break
    return reuse


def sibling_same(df):
    """Per depth: fraction of internal sibling pairs splitting on the same feature."""
    inner = df[df["parent_id"] >= 0]
    g = inner.groupby(["tree", "parent_id"])["attribute"]
    pairs = g.agg(["size", "nunique"])
    pairs = pairs[pairs["size"] == 2]
    dep = inner.groupby(["tree", "parent_id"])["depth"].first()
    pairs["depth"] = dep.reindex(pairs.index).values
    return pairs.groupby("depth")["nunique"].apply(lambda s: (s == 1).mean())


def per_run(path, F):
    df = pd.read_csv(path)
    df["reuse"] = ancestor_reuse(df)
    out = {}
    # --- per depth ---
    rows = []
    for d, g in df.groupby("depth"):
        counts = g["attribute"].value_counts()
        ne, eff = norm_entropy(counts.values, F)
        tree_depth = g.groupby("tree")["attribute"].agg(["size", "nunique"])
        multi = tree_depth[tree_depth["size"] >= 2]
        rows.append(dict(depth=d, n_nodes=len(g), n_distinct=counts.size,
                         frac_features_used=counts.size / F,
                         top1_share=topk_share(counts.values, 1),
                         top3_share=topk_share(counts.values, 3),
                         norm_entropy=ne, eff_num_features=eff,
                         reuse_rate=g["reuse"].mean(),
                         categorical_share=(g["cond_type"] != "higher").mean(),
                         # per-tree: distinct features at this depth / nodes at this depth
                         # per-tree, over levels with >=2 internal nodes (1 node is trivially oblivious)
                         within_tree_distinct_ratio=(multi["nunique"] / multi["size"]).mean(),
                         within_tree_oblivious=(multi["nunique"] == 1).mean(),
                         mean_num_examples=g["num_examples"].mean()))
    pdep = pd.DataFrame(rows).set_index("depth")
    sib = sibling_same(df)
    pdep["sibling_same_feature"] = sib.reindex(pdep.index)
    out["per_depth"] = pdep
    # --- whole-model ---
    counts = df["attribute_name"].value_counts()
    gain = df.groupby("attribute_name")["split_score"].sum().sort_values(ascending=False)
    roots = df[df["depth"] == 0].sort_values("tree")
    rc = roots["attribute_name"].value_counts()
    n_trees = df["tree"].nunique()
    early = roots[roots["tree"] < 30]["attribute_name"].value_counts()
    late = roots[roots["tree"] >= n_trees - 30]["attribute_name"].value_counts()
    per_tree = df.groupby("tree")["attribute"].agg(["size", "nunique"])
    out["model"] = dict(
        n_trees=n_trees, F=F, n_internal_nodes=len(df),
        internal_nodes_per_tree=per_tree["size"].mean(),
        distinct_features_per_tree=per_tree["nunique"].mean(),
        n_features_used=counts.size, frac_features_used=counts.size / F,
        top1_feature=counts.index[0], top1_share=counts.iloc[0] / counts.sum(),
        top3_share=topk_share(counts.values, 3), top5_share=topk_share(counts.values, 5),
        norm_entropy_all=norm_entropy(counts.values, F)[0],
        top1_gain_feature=gain.index[0], top1_gain_share=gain.iloc[0] / gain.sum(),
        n_distinct_roots=rc.size, root_top1=rc.index[0], root_top1_share=rc.iloc[0] / rc.sum(),
        root_top1_early=early.index[0], root_top1_early_share=early.iloc[0] / early.sum(),
        root_top1_late=late.index[0], root_top1_late_share=late.iloc[0] / late.sum(),
        reuse_rate=df["reuse"].mean(),
        categorical_share=(df["cond_type"] != "higher").mean(),
        max_depth_reached=df["depth"].max(),
    )
    out["feature_counts"] = counts
    out["feature_gain"] = gain
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=f"{HOME}/featstudy/runs")
    ap.add_argument("--out", default=f"{HOME}/featstudy/results")
    ap.add_argument("--data", default=DATA, help="cc18_binary_csv dir (for F and n_train)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = sorted(glob.glob(f"{a.runs}/*/nodes.csv"))
    by_ds = {}
    for p in runs:
        tag = os.path.basename(os.path.dirname(p))
        ds, fold = tag.rsplit("__fold", 1)
        by_ds.setdefault(ds, []).append((int(fold), p))
    ds_rows, depth_rows, feat_rows = [], [], []
    for ds, lst in by_ds.items():
        hdr = open(f"{a.data}/{ds}/repeat0_fold0_sample0_train.csv").readline().rstrip("\n").split(",")
        F = len(hdr) - 1
        n_train = sum(1 for _ in open(f"{a.data}/{ds}/repeat0_fold0_sample0_train.csv")) - 1
        models, depths, fc, fg = [], [], None, None
        for fold, p in lst:
            r = per_run(p, F)
            r["model"].update(dataset=ds, fold=fold)
            models.append(r["model"]); depths.append(r["per_depth"].assign(fold=fold))
            fc = r["feature_counts"] if fc is None else fc.add(r["feature_counts"], fill_value=0)
            fg = r["feature_gain"] if fg is None else fg.add(r["feature_gain"], fill_value=0)
        m = pd.DataFrame(models)
        num = m.select_dtypes("number").drop(columns=["fold"]).mean()
        row = dict(dataset=ds, n_folds=len(lst), n_train=n_train, **num.to_dict())
        for c in ["top1_feature", "top1_gain_feature", "root_top1", "root_top1_early", "root_top1_late"]:
            row[c] = m[c].mode().iloc[0]
        # pooled-over-folds feature usage (a feature counts as used if split on in any fold)
        row["n_features_used_pooled"] = int((fc > 0).sum()); row["frac_features_used_pooled"] = row["n_features_used_pooled"] / F
        ds_rows.append(row)
        dd = pd.concat(depths).reset_index().groupby("depth").mean(numeric_only=True).drop(columns=["fold"])
        dd.insert(0, "dataset", ds); dd.insert(1, "F", F)
        depth_rows.append(dd.reset_index())
        fc = fc.sort_values(ascending=False)
        for rank, (name, c) in enumerate(fc.head(10).items(), 1):
            feat_rows.append(dict(dataset=ds, rank=rank, feature=name, split_count=int(c),
                                  split_share=c / fc.sum(), gain_share=fg.get(name, 0) / fg.sum()))
        print(f"{ds}: F={F} folds={len(lst)} used={row['n_features_used_pooled']}/{F} "
              f"root_top1={row['root_top1']} ({row['root_top1_share']:.2f}) reuse={row['reuse_rate']:.2f}", flush=True)
    per_ds = pd.DataFrame(ds_rows).sort_values("F")
    per_ds.to_csv(f"{a.out}/per_dataset.csv", index=False)
    pd.concat(depth_rows).to_csv(f"{a.out}/per_depth.csv", index=False)
    pd.DataFrame(feat_rows).to_csv(f"{a.out}/top_features.csv", index=False)
    # pooled per-depth medians across datasets
    pdm = pd.concat(depth_rows).drop(columns=["dataset"]).groupby("depth").agg(["median", "mean"])
    pdm.to_csv(f"{a.out}/per_depth_pooled.csv")
    print("\n=== pooled per-depth medians across datasets ===")
    cols = ["n_distinct", "frac_features_used", "top1_share", "top3_share", "norm_entropy",
            "eff_num_features", "reuse_rate", "sibling_same_feature", "within_tree_distinct_ratio",
            "within_tree_oblivious", "categorical_share"]
    print(pd.concat(depth_rows).groupby("depth")[cols].median().round(3).to_string())
    print("\n=== per-dataset medians ===")
    cols = ["F", "n_features_used", "frac_features_used", "frac_features_used_pooled", "top1_share",
            "top3_share", "top5_share", "norm_entropy_all", "n_distinct_roots", "root_top1_share",
            "reuse_rate", "internal_nodes_per_tree", "distinct_features_per_tree", "categorical_share"]
    print(per_ds[cols].median().round(3).to_string())


if __name__ == "__main__":
    main()
