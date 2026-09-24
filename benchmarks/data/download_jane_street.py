#!/usr/bin/env python3
"""Jane Street Real-Time Market Data Forecasting (Kaggle, 2024) -> all-numeric binary CSVs.

Source: https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting (competition
rules; needs a Kaggle token: ~/.kaggle/access_token or KAGGLE_API_TOKEN, and `kaggle` in the venv).
train.parquet is partitioned into partition_id=0..9 (chronological by date_id; ~47 M rows, ~11.5 GB zipped).
Columns: date_id, time_id, symbol_id, weight, feature_00..feature_78 (anonymized, fp32, NaNs in many),
responder_0..responder_8 (targets; the competition scores responder_6).

Massage (dataset policy 2026-09-21):
  * target: class = 1 iff responder_6 > median(responder_6 over TRAIN) (median ~0); train median reused
    for test, recorded in the meta JSON.
  * dropped: weight (a sample weight, not a feature), responder_0..8 except the target (other targets).
  * features (82): date_id, time_id, symbol_id, feature_00..feature_78; NaN -> TRAIN column mean.
  * split: partitions 0..8 = train, partition 9 = test (the last ~10 % of dates). Row order = parquet
    order (date_id ascending), so `head -n K` = earliest-K prefix.
Outputs (benchmarks/data/jane_street/, gitignored): jane_street_train.csv, jane_street_test.csv,
jane_street_meta.json.
"""
import argparse, glob, os, subprocess, sys, time, zipfile
import numpy as np, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from large_csv_utils import *

COMP = "jane-street-real-time-market-data-forecasting"


def ensure_data(out_dir):
    if glob.glob(os.path.join(out_dir, "train.parquet", "partition_id=*", "*.parquet")):
        return
    z = os.path.join(out_dir, COMP + ".zip")
    if not os.path.exists(z):
        log("downloading via kaggle"); subprocess.check_call([sys.executable.replace("python", "kaggle"), "competitions", "download", "-c", COMP, "-p", out_dir])
    log("unzipping"); zipfile.ZipFile(z).extractall(out_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "jane_street"))
    ap.add_argument("--target", default="responder_6")
    ap.add_argument("--test_partitions", default="9")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True); ensure_data(a.out_dir)
    test_parts = {int(x) for x in a.test_partitions.split(",")}
    t0 = time.time()
    parts = sorted(glob.glob(os.path.join(a.out_dir, "train.parquet", "partition_id=*", "*.parquet")), key=lambda p: int(p.split("partition_id=")[1].split("/")[0]))
    tabs = {"train": [], "test": []}
    for p in parts:
        pid = int(p.split("partition_id=")[1].split("/")[0]); t = pq.read_table(p)
        t = t.sort_by([("date_id", "ascending"), ("time_id", "ascending"), ("symbol_id", "ascending")])
        tabs["test" if pid in test_parts else "train"].append(t); log(f"partition {pid}: {t.num_rows} rows")
    res = {}
    for k in ("train", "test"):
        t = pa.concat_tables(tabs[k]); y = t.column(a.target).to_numpy(zero_copy_only=False).astype(np.float64)
        drop = ["weight"] + [c for c in t.column_names if c.startswith("responder_")]
        res[k] = (t.drop(drop), y); log(f"{k}: {t.num_rows} rows, {len(res[k][0].column_names)} features, {time.time()-t0:.0f}s")
    Xtr, ytr = res["train"]; Xte, yte = res["test"]
    thr = float(np.median(ytr)); means = column_means(Xtr); nans_tr = nan_counts(Xtr); nans_te = nan_counts(Xte)
    paths = {}
    for k, (X, y) in res.items():
        tab = impute_and_binarize(X, y, means, thr); p = os.path.join(a.out_dir, f"jane_street_{k}.csv"); write_csv_unquoted(tab, p); paths[k] = (p, tab.num_rows, float(np.mean(y > thr)))
        log(f"wrote {p} ({os.path.getsize(p)/1e9:.1f} GB) in {time.time()-t0:.0f}s")
    meta = {"name": "jane_street", "source": "kaggle competition " + COMP, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target": {"name": a.target, "rule": f"class = 1 iff {a.target} > train median {thr}", "train_median": thr},
            "train": {"csv": paths["train"][0], "rows": paths["train"][1], "positive_rate": paths["train"][2], "partitions": sorted(set(range(10)) - test_parts)},
            "test": {"csv": paths["test"][0], "rows": paths["test"][1], "positive_rate": paths["test"][2], "partitions": sorted(test_parts)},
            "dropped": ["weight"] + [f"responder_{i}" for i in range(9) if f"responder_{i}" != a.target], "features": Xtr.column_names,
            "imputation": {"rule": "NaN -> TRAIN column mean", "train_nan_cells": nans_tr, "test_nan_cells": nans_te, "train_means": means}}
    write_meta(os.path.join(a.out_dir, "jane_street_meta.json"), meta)
    log(f"train positives {paths['train'][2]:.4f}; nan cells train {sum(nans_tr.values())} test {sum(nans_te.values())}")


if __name__ == "__main__":
    main()
