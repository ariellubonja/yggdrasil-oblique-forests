#!/usr/bin/env python3
"""Numeric fold-0 copies of the CC18 binary tasks for the distributed GBT drivers.

Reads benchmarks/data/cc18_binary_csv/task_<id>_<name>/repeat0_fold0_sample0_{train,test}.csv
(written by download_cc18_datasets.py) and writes
benchmarks/data/cc18_fold0_numeric/<name>/fold0_{train,test}.csv + meta.json, following the
dataset policy (CLAUDE.md "Dataset policy"):
  - categorical / string columns -> ordinal codes over the sorted distinct values of train+test
    (same codes on both sides); bool -> 0/1; numeric kept as float32;
  - NaN feature cells -> the TRAIN fold's column mean (test reuses it; all-NaN column -> 0.0);
  - label -> 0/1 tokens via tabular_suite_prep.encode_binary_label on train+test (two classes:
    majority -> 0; more classes: majority-vs-rest with the majority class positive).
meta.json records rows, features, what was encoded / imputed and the label mapping.

  python3 benchmarks/data/prep_cc18_fold0.py [--only name,name]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tabular_suite_prep import encode_binary_label  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
SRC_DIR = os.path.join(REPO_ROOT, "benchmarks", "data", "cc18_binary_csv")
DST_DIR = os.path.join(REPO_ROOT, "benchmarks", "data", "cc18_fold0_numeric")
MANIFEST = os.path.join(REPO_ROOT, "benchmarks", "data", "cc18_manifest.json")


def encode_split(train: pd.DataFrame, test: pd.DataFrame, label: str):
    """Encode features jointly, impute with train means, tokenize the label."""
    both = pd.concat([train, test], keys=["train", "test"])
    n_train = len(train)
    stats = {"n_numeric": 0, "n_bool": 0, "n_categorical_encoded": 0,
             "n_all_nan": 0, "nan_cells_imputed_train": 0,
             "nan_cells_imputed_test": 0, "categorical_columns": []}
    out = {}
    for name in both.columns:
        if name == label:
            continue
        col = both[name]
        if pd.api.types.is_bool_dtype(col):
            vals = col.astype("float32").to_numpy()
            stats["n_bool"] += 1
        elif pd.api.types.is_numeric_dtype(col):
            vals = col.astype("float32").to_numpy()
            stats["n_numeric"] += 1
        else:
            s = col.astype("string")
            uniq = sorted(s.dropna().unique())
            codes = {v: float(i) for i, v in enumerate(uniq)}
            vals = s.map(codes).astype("float32").to_numpy(dtype="float32", na_value=np.nan)
            stats["n_categorical_encoded"] += 1
            stats["categorical_columns"].append(name)
        vals = np.asarray(vals, dtype="float32")
        nan = np.isnan(vals)
        if nan.any():
            stats["nan_cells_imputed_train"] += int(nan[:n_train].sum())
            stats["nan_cells_imputed_test"] += int(nan[n_train:].sum())
            mean = float(np.nanmean(vals[:n_train])) if not nan[:n_train].all() else 0.0
            if not np.isfinite(mean):
                mean = 0.0
            if nan[:n_train].all():
                stats["n_all_nan"] += 1
            vals = np.where(nan, np.float32(mean), vals).astype("float32")
        out[name] = vals
    feats = pd.DataFrame(out)
    tokens, lstats = encode_binary_label(both[label].reset_index(drop=True))
    feats[label] = tokens.to_numpy().astype("int8")
    return feats.iloc[:n_train].reset_index(drop=True), feats.iloc[n_train:].reset_index(drop=True), stats, lstats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated dataset names")
    args = ap.parse_args()
    only = {s for s in args.only.split(",") if s}
    with open(MANIFEST) as f:
        entries = [e for e in json.load(f)["entries"] if e["status"] == "ok"]
    done = []
    for e in entries:
        name = e["dataset_name"]
        if only and name not in only:
            continue
        task_dir = os.path.join(SRC_DIR, f"task_{e['task_id']}_{name}")
        src_train = os.path.join(task_dir, "repeat0_fold0_sample0_train.csv")
        src_test = os.path.join(task_dir, "repeat0_fold0_sample0_test.csv")
        if not (os.path.isfile(src_train) and os.path.isfile(src_test)):
            print(f"skip {name}: fold0 CSVs missing in {task_dir}", file=sys.stderr)
            continue
        label = e["label_column"]
        train = pd.read_csv(src_train)
        test = pd.read_csv(src_test)
        tr, te, fstats, lstats = encode_split(train, test, label)
        out_dir = os.path.join(DST_DIR, name)
        os.makedirs(out_dir, exist_ok=True)
        tr.to_csv(os.path.join(out_dir, "fold0_train.csv"), index=False, float_format="%.9g")
        te.to_csv(os.path.join(out_dir, "fold0_test.csv"), index=False, float_format="%.9g")
        meta = {"dataset": name, "task_id": e["task_id"], "label_col": label,
                "rows_train": int(len(tr)), "rows_test": int(len(te)),
                "features": int(tr.shape[1] - 1), "label": lstats,
                "features_encoding": fstats, "source": {"train": src_train, "test": src_test}}
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        done.append(name)
        print(f"{name}: {len(tr)}x{tr.shape[1]-1} train, {len(te)} test, "
              f"cat={fstats['n_categorical_encoded']} nan_train={fstats['nan_cells_imputed_train']} "
              f"pos_rate={lstats['positive_rate']:.3f}")
    print(f"done: {len(done)} datasets -> {DST_DIR}")


if __name__ == "__main__":
    main()
