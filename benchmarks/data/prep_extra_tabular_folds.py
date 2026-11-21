#!/usr/bin/env python3
"""TabArena multiclass (majority-vs-rest) and TabArena/TabReD regression folds, 2026-11-20 (Hao vs YDF table).
Same rules as the SPAA paper eval (rebuild_table8_data.py): ordinal categoricals, NaN -> train-fold column mean,
TabArena 5 CV folds seed 0 (stratified for classification, KFold for regression), TabReD = HF harbor train_full.csv,
chronological 80/20 holdout. Run with the pinned gbt_venv:
  python prep_extra_tabular_folds.py <folds_dir> [--only a,b]
Writes <folds_dir>/<name>/{fold<k>_train,fold<k>_test}.csv, folds.json, meta.json and <folds_dir>/{shapes,categoricals}.csv."""
import argparse, csv, json, os, subprocess, sys

import pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(D))
sys.path.insert(0, D)
sys.path.insert(0, os.path.join(REPO, "benchmarks/src/spaa27_experiments"))
from tabular_suite_prep import encode_binary_label  # noqa: E402
from make_train_nan import encode_features_keep_nan  # noqa: E402
from run_suite import make_folds  # noqa: E402

TABRED_REG = ["cooking-time", "delivery-eta", "maps-routing", "sberbank-housing", "weather"]
HF_REPO = "danil-e/harbor-datasets-tabred"


def tabarena(kind):
    m = pd.read_csv(os.path.join(D, "tabarena/_meta/curated_tabarena_dataset_metadata.csv"))
    return sorted(m[m.problem_type == kind].dataset_name)


def load(name, suite):
    """(X, y, n_categorical, problem) from the raw source, categorical counted as in build_kdd28_categoricals.py."""
    if suite == "tabred":
        raw = pd.read_csv(os.path.join(D, "tabred_raw_mirror", f"{name}_train_full.csv"))
        X = raw.drop(columns=[c for c in ("id", "timestamp", "target") if c in raw.columns])
        return X, raw["target"], sum(c.startswith("cat_") for c in X.columns), "regression"
    d = os.path.join(D, "tabarena", name)
    info = json.load(open(os.path.join(d, "info.json")))
    df = pd.read_parquet(os.path.join(d, "data.parquet"))
    t = info["target_feature"]
    X = df.drop(columns=[t])
    ncat = sum(info["columns"].get(c, {}).get("is_categorical", False) or X[c].dtype.kind not in "iuf" for c in X.columns)
    return X, df[t], ncat, info.get("problem_type")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folds_dir")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    jobs = [(n, "tabarena", "multiclass") for n in tabarena("multiclass")] + \
           [(n, "tabarena", "regression") for n in tabarena("regression")] + [(n, "tabred", "regression") for n in TABRED_REG]
    only = set(filter(None, a.only.split(",")))
    jobs = [j for j in jobs if not only or j[0] in only]
    subprocess.run([sys.executable, os.path.join(D, "download_tabarena_datasets.py"),
                    *[n for n, s, _ in jobs if s == "tabarena"]], check=True)
    for n, s, _ in jobs:
        if s == "tabred" and not os.path.isfile(os.path.join(D, "tabred_raw_mirror", f"{n}_train_full.csv")):
            sys.exit(f"missing tabred_raw_mirror/{n}_train_full.csv (download {HF_REPO} datasets/tabred-real/{n}/environment/data/train_full.csv)")
    shapes, cats = [], []
    for n, s, kind in jobs:
        X, y, ncat, problem = load(n, s)
        if s == "tabarena" and problem != kind:
            sys.exit(f"{n}: info.json problem_type {problem} != metadata {kind}")
        if y.isna().any():
            sys.exit(f"{n}: label has {int(y.isna().sum())} NaN")
        feats = encode_features_keep_nan(X)
        if kind == "regression":
            lab, lmeta = y.astype("float64"), {"task": "regression", "mean": float(y.mean()), "std": float(y.std())}
        else:
            lab, lmeta = encode_binary_label(y)
            lmeta["task"] = "classification"
        feats["label"] = lab.to_numpy()
        src = os.path.join(a.folds_dir, "_src", n)
        os.makedirs(src, exist_ok=True)
        feats.to_csv(os.path.join(src, "train_nan.csv"), index=False, float_format="%.9g", na_rep="")
        make_folds(n, os.path.join(src, "train.csv"), "label", 5, 0, a.folds_dir, chrono=s == "tabred",
                   holdout_frac=0.2, regression=kind == "regression")
        meta = {"dataset": n, "suite": s, "problem_type": kind, "rows": len(feats), "features": feats.shape[1] - 1,
                "categorical": int(ncat), "nan_cells": int(feats.drop(columns="label").isna().sum().sum()), "label": lmeta}
        json.dump(meta, open(os.path.join(a.folds_dir, n, "meta.json"), "w"), indent=1)
        shapes.append((n, len(feats), feats.shape[1] - 1)); cats.append((n, int(ncat), feats.shape[1] - 1, s))
        print(f"{n}: {kind} rows={len(feats)} feats={feats.shape[1] - 1} cat={ncat} nan={meta['nan_cells']}", flush=True)
    with open(os.path.join(a.folds_dir, "shapes.csv"), "w", newline="") as f:
        csv.writer(f).writerows([("dataset", "rows", "features")] + shapes)
    with open(os.path.join(a.folds_dir, "categoricals.csv"), "w", newline="") as f:
        csv.writer(f).writerows([("dataset", "categorical", "features", "source")] + cats)


if __name__ == "__main__":
    main()
