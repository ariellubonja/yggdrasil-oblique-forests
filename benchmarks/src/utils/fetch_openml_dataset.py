#!/usr/bin/env python3
"""Fetch an OpenML dataset by ID, write it as a header'd CSV with the
target column renamed to 'class'. Optionally split into train/test.

Examples (CC18-relevant, binary, high-dim):
  Madelon         id=1485   500 feat  / 2600 ex
  Bioresponse     id=4134  1776 feat  / 3751 ex
  Gisette         id=41026 5000 feat  / 7000 ex
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import openml
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, required=True,
                    help="OpenML dataset id")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--name", default=None,
                    help="File stem (default: dataset name from OpenML)")
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching dataset {args.id} ...")
    ds = openml.datasets.get_dataset(
        args.id, download_data=True, download_qualities=False,
        download_features_meta_data=False)
    name = args.name or ds.name.lower().replace(" ", "_")
    X, y, _, _ = ds.get_data(
        target=ds.default_target_attribute,
        dataset_format="dataframe")

    # Force-cast to float for any string-formatted numeric features.
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    # Replace NaN with 0 (rare for binary classification tabular).
    X = X.fillna(0)

    # Map labels to {0,1} ints regardless of original encoding.
    if y.dtype.kind in "OSU":
        cats = sorted(y.unique())
        if len(cats) != 2:
            print(f"[WARN] non-binary target: {cats}", file=sys.stderr)
        y = y.astype("category").cat.codes
    else:
        # Numeric labels — re-encode to 0/1 just in case.
        cats = sorted(np.unique(y))
        if len(cats) == 2 and set(cats) != {0, 1}:
            mapping = {cats[0]: 0, cats[1]: 1}
            y = y.map(mapping)

    df = X.copy()
    df["class"] = y.astype(int).values
    print(f"  shape={df.shape}  columns[:3]={list(df.columns)[:3]}  "
          f"label balance={df['class'].value_counts().to_dict()}")

    # Save full CSV.
    full = out_dir / f"{name}.csv"
    df.to_csv(full, index=False)
    print(f"  wrote {full}")

    # Train/test split (deterministic by seed).
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_tr = int(args.train_frac * len(df))
    tr = df.iloc[idx[:n_tr]]
    te = df.iloc[idx[n_tr:]]
    tr_path = out_dir / f"{name}_train.csv"
    te_path = out_dir / f"{name}_test.csv"
    tr.to_csv(tr_path, index=False)
    te.to_csv(te_path, index=False)
    print(f"  wrote {tr_path} ({len(tr)} rows)")
    print(f"  wrote {te_path} ({len(te)} rows)")


if __name__ == "__main__":
    main()
