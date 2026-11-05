#!/usr/bin/env python3
"""Rebuild kdd28_wide_datasets (pins/rules: benchmarks/data/kdd28_wide_datasets_manifest.json) + folds, sha256-checked.
<venv>/bin/python rebuild_wide_data.py <folds_dir> [--only a,b] [--force [--redownload]]
Venv as rebuild_table8_data.py: numpy 2.5.2, pandas 3.0.5, scikit-learn 1.9.0, pyarrow 25.0.1."""
import argparse
import collections
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = os.path.join(REPO, "benchmarks", "data")
sys.path.insert(0, DATA)
sys.path.insert(0, os.path.join(REPO, "benchmarks", "src", "spaa27_experiments"))
from tabular_suite_prep import encode_binary_label, write_dataset  # noqa: E402
from make_train_nan import encode_features_keep_nan  # noqa: E402
from run_suite import make_folds  # noqa: E402

MANIFEST = os.path.join(DATA, "kdd28_wide_datasets_manifest.json")
SHA = os.path.join(DATA, "kdd28_wide_datasets_manifest.sha256")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def curl(url, out):
    subprocess.run(["curl", "-sSfL", "--retry", "3", "-o", out, url], check=True)


def check_pin(e):
    """The OpenML API must still describe the pinned dataset (name, version, active, ARFF md5)."""
    meta_p = os.path.join(os.path.dirname(e["_raw"]), "openml_data.json")
    curl(f"https://www.openml.org/api/v1/json/data/{e['openml_data_id']}", meta_p)
    d = json.load(open(meta_p))["data_set_description"]
    got = {"name": d["name"], "version": int(d["version"]), "status": d["status"], "md5": d["md5_checksum"]}
    want = {"name": e["openml_name"], "version": e["openml_version"], "status": "active", "md5": e["openml_md5"]}
    if got != want:
        sys.exit(f"{e['name']}: OpenML pin drift, got {got}, want {want}")


def build(e, data_dir, folds_dir):
    name = e["name"]
    if not os.path.isfile(e["_raw"]):
        curl(e["parquet_url"], e["_raw"])
    df = pd.read_parquet(e["_raw"])
    if list(df.shape) != e["openml_shape"]:
        sys.exit(f"{name}: raw shape {list(df.shape)} != manifest {e['openml_shape']}")
    target = e["target"]
    X = df.drop(columns=[target] + e["drop_columns"])
    y = df[target]
    if [X.shape[0], X.shape[1]] != [e["expected_rows"], e["expected_features"]]:
        sys.exit(f"{name}: features {X.shape} != manifest ({e['expected_rows']}, {e['expected_features']})")
    meta = write_dataset(data_dir, name, X, y, extra_meta={
        "suite": "kdd28_wide_datasets", "openml_data_id": e["openml_data_id"],
        "openml_version": e["openml_version"], "dropped_columns": e["drop_columns"]})
    if meta["label"]["counts"][1] != e["expected_minority"]:
        sys.exit(f"{name}: label 1 count {meta['label']['counts'][1]} != manifest {e['expected_minority']}")
    if meta["features_encoding"]["nan_cells_imputed"]:
        feats = encode_features_keep_nan(X)
        tokens, _ = encode_binary_label(y)
        feats["label"] = tokens.to_numpy()
        feats.to_csv(os.path.join(data_dir, name, "train_nan.csv"), index=False, float_format="%.9g", na_rep="")
    make_folds(name, os.path.join(data_dir, name, "train.csv"), "label", 5, 0, folds_dir)
    print(f"built {name}: rows={meta['rows']} features={meta['features']} label1={meta['label']['counts'][1]} "
          f"nan_cells={meta['features_encoding']['nan_cells_imputed']}", flush=True)


def build_hf(e, folds_dir):
    """Shipped train/test parquet shards -> one holdout fold0_{train,test}.csv, the bytes download_epsilon.py wrote
    (float64 repr, label as shipped). Epsilon: the files the Vertex runs read (gs://ariel_ydf_bucket/big/EPSILON)."""
    name, out = e["name"], os.path.join(folds_dir, e["name"])
    os.makedirs(out, exist_ok=True)
    counts = {}
    for split in ("train", "test"):
        dst = os.path.join(out, f"fold0_{split}.csv")
        rows, labels = 0, collections.Counter()
        for i in range(e["shards"][split]):
            p = os.path.join(e["_raw"], split, f"{i:04d}.parquet")
            if not os.path.isfile(p):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                curl(e["parquet_url"].format(split=split, shard=i), p)
            t = pq.read_table(p)  # inputs: list<list<double>>, one single-element list per feature
            X = t.column(e["features_column"]).combine_chunks().flatten().flatten().to_numpy().reshape(t.num_rows, -1)
            df = t.select([e["target"]]).to_pandas()
            if X.shape[1] != e["expected_features"]:
                sys.exit(f"{name} {split} shard {i}: {X.shape[1]} features != manifest {e['expected_features']}")
            flat = pd.DataFrame(X, columns=[f"f{j}" for j in range(X.shape[1])])
            flat["label"] = df[e["target"]].values
            flat.to_csv(dst + ".part", index=False, mode="w" if i == 0 else "a", header=i == 0)
            rows += len(flat)
            labels.update(flat["label"].tolist())
        if rows != e["expected_rows"][split]:
            sys.exit(f"{name} {split}: {rows} rows != manifest {e['expected_rows'][split]}")
        os.replace(dst + ".part", dst)
        counts[split] = {str(k): v for k, v in sorted(labels.items())}
    with open(os.path.join(out, "folds.json"), "w") as f:
        json.dump({"dataset": name, "protocol": "shipped train/test split, one holdout fold (k=0)",
                   "indices": [{"fold": 0, "train_rows": e["expected_rows"]["train"],
                                "test_rows": e["expected_rows"]["test"]}], "label_counts": counts}, f, indent=1)
    print(f"built {name}: train={e['expected_rows']['train']} test={e['expected_rows']['test']} labels={counts}", flush=True)


def outputs(e, data_dir, folds_dir):
    name = e["name"]
    if e.get("source") == "hf_parquet":
        return {f"folds/{name}/fold0_{s}.csv": os.path.join(folds_dir, name, f"fold0_{s}.csv") for s in ("train", "test")}
    files = {f"raw/{name}/{os.path.basename(e['_raw'])}": e["_raw"]}
    for f in ("train.csv", "train_nan.csv"):
        p = os.path.join(data_dir, name, f)
        if os.path.isfile(p):
            files[f"data/{name}/{f}"] = p
    for k in range(5):
        for s in ("train", "test"):
            files[f"folds/{name}/fold{k}_{s}.csv"] = os.path.join(folds_dir, name, f"fold{k}_{s}.csv")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folds_dir")
    ap.add_argument("--data_dir", help="default <folds_dir>/../wide_binary_csv")
    ap.add_argument("--raw_dir", help="default <folds_dir>/../wide_raw (one new dir per dataset)")
    ap.add_argument("--only", default="", help="comma-separated dataset names")
    ap.add_argument("--force", action="store_true", help="delete data and fold files, rebuild, then check")
    ap.add_argument("--redownload", action="store_true", help="with --force, also re-download the raw parquet")
    a = ap.parse_args()
    folds_dir = os.path.abspath(a.folds_dir)
    parent = os.path.dirname(folds_dir)
    data_dir = os.path.abspath(a.data_dir or os.path.join(parent, "wide_binary_csv"))
    raw_dir = os.path.abspath(a.raw_dir or os.path.join(parent, "wide_raw"))
    only = set(filter(None, a.only.split(",")))
    entries = json.load(open(MANIFEST))["datasets"]
    todo = [e for e in entries if e["status"] == "built" and (not only or e["name"] in only)]
    for e in entries:
        if e["status"] != "built" and (not only or e["name"] in only):
            print(f"skip {e['name']}: status {e['status']} ({e.get('unbuilt_reason', '')})")
    for e in todo:
        os.makedirs(os.path.join(raw_dir, e["name"]), exist_ok=True)
        hf = e.get("source") == "hf_parquet"
        e["_raw"] = os.path.join(raw_dir, e["name"]) if hf else \
            os.path.join(raw_dir, e["name"], f"dataset_{e['openml_data_id']}.pq")
        if a.force:
            for p in (os.path.join(data_dir, e["name"]), os.path.join(folds_dir, e["name"])):
                shutil.rmtree(p, ignore_errors=True)
            if a.redownload and os.path.exists(e["_raw"]):
                shutil.rmtree(e["_raw"]) if hf else os.remove(e["_raw"])
        if hf:
            if not all(os.path.isfile(p) for p in outputs(e, data_dir, folds_dir).values()):
                build_hf(e, folds_dir)
        else:
            check_pin(e)
            build(e, data_dir, folds_dir)

    have = {}
    if os.path.isfile(SHA):
        for line in open(SHA):
            h, k = line.split()
            have[k] = h
    ok = bad = new = 0
    for e in todo:
        for k, p in outputs(e, data_dir, folds_dir).items():
            h = sha256(p)
            if k not in have:
                have[k] = h
                new += 1
            elif have[k] == h:
                ok += 1
            else:
                bad += 1
                print("MISMATCH", k)
    if new:
        with open(SHA, "w") as f:
            for k in sorted(have):
                f.write(f"{have[k]}  {k}\n")
    print(f"sha256 vs {os.path.relpath(SHA, REPO)}: {ok} match, {bad} mismatch, {new} new (recorded)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
