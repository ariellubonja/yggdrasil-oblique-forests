#!/usr/bin/env python3
"""Report per-dataset column-type and missing-value properties for TabArena / TabReD.

Numerical column counting rules:
  TabArena -- OpenML ships an is_categorical mask per attribute. A column counts as
    numerical if it is NOT flagged categorical AND its pandas dtype is a numeric
    kind (int/float/uint). The target column is excluded from the feature counts
    and reported separately.
  TabReD  -- the preprocessed format already splits features into x_num / x_bin /
    x_cat, so "numerical" is exactly x_num.shape[1] (x_bin = 0/1 columns, counted
    separately; they are numeric-valued but semantically binary).

Missing entries are counted over feature cells only (target excluded), reported as
an absolute count, a percentage of all feature cells, and the number of affected
columns/rows.

  .venv/bin/python benchmarks/data/summarize_datasets.py [--csv out.csv]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TABARENA = os.path.join(REPO_ROOT, "benchmarks", "data", "tabarena")
TABRED = os.path.join(REPO_ROOT, "benchmarks", "data", "tabred")

NUMERIC_KINDS = set("iuf")


def summarize_tabarena():
    out = []
    if not os.path.isdir(TABARENA):
        return out
    for name in sorted(os.listdir(TABARENA), key=str.lower):
        ddir = os.path.join(TABARENA, name)
        info_p = os.path.join(ddir, "info.json")
        pq = os.path.join(ddir, "data.parquet")
        if name.startswith("_") or not os.path.exists(info_p):
            continue
        info = json.load(open(info_p))
        df = pd.read_parquet(pq)
        target = info["target_feature"]
        feats = [c for c in df.columns if c != target]
        cols = info["columns"]

        n_num = n_cat = n_bool_other = 0
        for c in feats:
            is_cat = cols.get(c, {}).get("is_categorical", False)
            kind = df[c].dtype.kind
            if not is_cat and kind in NUMERIC_KINDS:
                n_num += 1
            elif is_cat:
                n_cat += 1
            else:
                n_bool_other += 1

        sub = df[feats]
        na = sub.isna()
        n_missing = int(na.to_numpy().sum())
        cells = int(sub.shape[0]) * int(sub.shape[1])
        num_feats = [c for c in feats
                     if not cols.get(c, {}).get("is_categorical", False)
                     and df[c].dtype.kind in NUMERIC_KINDS]
        n_missing_num = int(df[num_feats].isna().to_numpy().sum()) if num_feats else 0
        out.append(dict(
            benchmark="TabArena", dataset=name, task=info["problem_type"],
            n_rows=len(df), n_features=len(feats),
            n_numerical=n_num, n_categorical=n_cat, n_other=n_bool_other,
            n_missing=n_missing,
            pct_missing=100.0 * n_missing / cells if cells else 0.0,
            n_missing_in_numerical=n_missing_num,
            cols_with_missing=int((na.sum(axis=0) > 0).sum()),
            rows_with_missing=int((na.sum(axis=1) > 0).sum()),
            target_missing=int(df[target].isna().sum()),
        ))
    return out


def summarize_tabred():
    out = []
    if not os.path.isdir(TABRED):
        return out
    for name in sorted(os.listdir(TABRED), key=str.lower):
        ddir = os.path.join(TABRED, name)
        info_p = os.path.join(ddir, "info.json")
        if name.startswith("_") or not os.path.exists(info_p):
            continue
        info = json.load(open(info_p))

        def load(fn):
            p = os.path.join(ddir, fn)
            return np.load(p, allow_pickle=True) if os.path.exists(p) else None

        x_num, x_bin, x_cat = load("x_num.npy"), load("x_bin.npy"), load("x_cat.npy")
        y = load("y.npy")

        def ncols(a):
            return 0 if a is None else (1 if a.ndim == 1 else a.shape[1])

        n_num, n_bin, n_cat = ncols(x_num), ncols(x_bin), ncols(x_cat)
        n_rows = len(y) if y is not None else (0 if x_num is None else len(x_num))

        # Missing: NaN in float blocks; TabReD encodes missing categoricals as a
        # dedicated category, so count them only where the array is float-typed.
        miss = 0
        miss_num = 0
        cols_miss = 0
        row_mask = np.zeros(n_rows, dtype=bool)
        for a, is_num in ((x_num, True), (x_bin, False), (x_cat, False)):
            if a is None:
                continue
            if a.dtype.kind != "f":
                continue
            m = np.isnan(a)
            if m.ndim == 1:
                m = m[:, None]
            miss += int(m.sum())
            if is_num:
                miss_num += int(m.sum())
            cols_miss += int((m.sum(axis=0) > 0).sum())
            row_mask |= m.any(axis=1)

        cells = n_rows * (n_num + n_bin + n_cat)
        out.append(dict(
            benchmark="TabReD", dataset=name,
            task=info.get("task", {}).get("type", info.get("problem_type", "?")),
            n_rows=n_rows, n_features=n_num + n_bin + n_cat,
            n_numerical=n_num, n_categorical=n_cat, n_other=n_bin,
            n_missing=miss,
            pct_missing=100.0 * miss / cells if cells else 0.0,
            n_missing_in_numerical=miss_num,
            cols_with_missing=cols_miss,
            rows_with_missing=int(row_mask.sum()),
            target_missing=int(np.isnan(y).sum()) if y is not None and y.dtype.kind == "f" else 0,
        ))
    return out


FIELDS = ["benchmark", "dataset", "task", "n_rows", "n_features", "n_numerical",
          "n_categorical", "n_other", "n_missing", "pct_missing",
          "n_missing_in_numerical", "cols_with_missing", "rows_with_missing",
          "target_missing"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    rows = summarize_tabarena() + summarize_tabred()
    if not rows:
        sys.exit("no datasets found")

    hdr = (f"{'benchmark':9s} {'dataset':38s} {'task':11s} {'rows':>9s} {'feat':>5s} "
           f"{'num':>5s} {'cat':>5s} {'bin':>4s} {'missing':>12s} {'miss%':>7s} "
           f"{'miss(num)':>11s} {'cols_na':>7s} {'rows_na':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['benchmark']:9s} {r['dataset'][:38]:38s} {r['task'][:11]:11s} "
              f"{r['n_rows']:>9,d} {r['n_features']:>5d} {r['n_numerical']:>5d} "
              f"{r['n_categorical']:>5d} {r['n_other']:>4d} {r['n_missing']:>12,d} "
              f"{r['pct_missing']:>7.3f} {r['n_missing_in_numerical']:>11,d} "
              f"{r['cols_with_missing']:>7d} {r['rows_with_missing']:>9,d}")

    for bench in ("TabArena", "TabReD"):
        sel = [r for r in rows if r["benchmark"] == bench]
        if not sel:
            continue
        clean = sum(1 for r in sel if r["n_missing"] == 0)
        print(f"\n{bench}: {len(sel)} datasets | "
              f"{sum(r['n_numerical'] for r in sel):,} numerical cols total "
              f"(of {sum(r['n_features'] for r in sel):,} features) | "
              f"{clean} datasets with zero missing values, {len(sel)-clean} with missing")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
