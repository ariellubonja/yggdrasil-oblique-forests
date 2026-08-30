#!/usr/bin/env python3
"""Download the TabArena-v0.1 benchmark datasets (51 curated OpenML tasks).

Dataset list comes from the pinned metadata CSV copied out of
autogluon/tabarena (packages/tabarena/.../curated_tabarena_dataset_metadata.csv),
kept at benchmarks/data/tabarena/_meta/. Each dataset is fetched from OpenML by
its pinned dataset_id and written as parquet:

  benchmarks/data/tabarena/<dataset_name>/
  ├── data.parquet   # full X + target column
  └── info.json      # openml ids, target, problem type, per-column dtype/NaN stats

  .venv/bin/python benchmarks/data/download_tabarena_datasets.py
"""
import json
import os
import sys
import csv
import traceback

import openml
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO_ROOT, "benchmarks", "data", "tabarena")
META_CSV = os.path.join(OUT_DIR, "_meta", "curated_tabarena_dataset_metadata.csv")

openml.config.cache_directory = os.path.join(OUT_DIR, "_meta", "openml_cache")


def main():
    rows = list(csv.DictReader(open(META_CSV)))
    only = set(sys.argv[1:])
    failures = []
    for i, r in enumerate(rows, 1):
        name = r["dataset_name"]
        if only and name not in only:
            continue
        ddir = os.path.join(OUT_DIR, name)
        pq = os.path.join(ddir, "data.parquet")
        if os.path.exists(pq) and os.path.exists(os.path.join(ddir, "info.json")):
            print(f"[{i}/{len(rows)}] {name}: cached", flush=True)
            continue
        print(f"[{i}/{len(rows)}] {name}: downloading did={r['dataset_id']}", flush=True)
        try:
            ds = openml.datasets.get_dataset(
                int(r["dataset_id"]), download_data=True,
                download_qualities=False, download_features_meta_data=True)
            X, _, cat_mask, attr_names = ds.get_data(dataset_format="dataframe")
            target = r["target_feature"]
            os.makedirs(ddir, exist_ok=True)
            X.to_parquet(pq, index=False)
            info = {
                "name": name,
                "openml_dataset_id": int(r["dataset_id"]),
                "openml_task_id": int(r["task_id"]),
                "openml_dataset_name": r["openml_dataset_name"],
                "target_feature": target,
                "problem_type": r["problem_type"],
                "n_rows": int(len(X)),
                "n_cols_total": int(X.shape[1]),
                "columns": {
                    c: {"dtype": str(X[c].dtype),
                        "is_categorical": bool(m),
                        "n_missing": int(X[c].isna().sum())}
                    for c, m in zip(attr_names, cat_mask)
                },
            }
            with open(os.path.join(ddir, "info.json"), "w") as f:
                json.dump(info, f, indent=1)
        except Exception:
            traceback.print_exc()
            failures.append(name)
    if failures:
        print("FAILED:", failures, file=sys.stderr)
        sys.exit(1)
    print("done")


if __name__ == "__main__":
    main()
