#!/usr/bin/env python3
"""Yandex Shifts *Weather Prediction* dataset (canonical partition) -> all-numeric binary CSVs.

Source: https://research.yandex.com/shifts/weather  (Shifts Project, CC BY-NC-SA 4.0 per LICENSE.md
in the archive), 7.5 GB tar `canonical-partitioned-dataset.tar` from storage.yandexcloud.net.
Splits inside: train 3,129,592 rows (Sept 2018 - July 2019), dev_in / dev_out 50,000 each,
eval_in 561,105, eval_out 576,626. 129 columns: fact_time, fact_latitude, fact_longitude,
fact_temperature (the regression target), fact_cwsm_class (a second observed label), climate
(string: 5 climate types), then 123 forecast/climatology features (cmc_*, gfs_*, wrf_*, ...).

Massage (dataset policy 2026-09-21):
  * target: class = 1 iff fact_temperature > median(fact_temperature over TRAIN); the train median is
    reused for the test split and recorded in the meta JSON.
  * dropped: fact_temperature (target), fact_cwsm_class (an observed label, not a forecast).
  * features (127): fact_time (epoch seconds, fp32 => 128 s grid), fact_latitude, fact_longitude,
    climate (ordinal codes, alphabetical), and the 123 numeric columns; NaN -> TRAIN column mean.
  * rows: train sorted by fact_time (stable), so `head -n K` = earliest-K prefix; test = eval_in + eval_out
    (the time-shifted / climate-shifted evaluation partitions), also sorted by fact_time.
Outputs (benchmarks/data/shifts_weather/, gitignored): shifts_weather_train.csv, shifts_weather_test.csv,
shifts_weather_meta.json.
"""
import argparse, os, subprocess, sys, tarfile, time
import numpy as np, pyarrow as pa, pyarrow.compute as pc, pyarrow.csv as pacsv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from large_csv_utils import *

URL = "https://storage.yandexcloud.net/yandex-research/shifts/weather/canonical-partitioned-dataset.tar"
D = "canonical-paritioned-dataset"  # sic, the archive's spelling


def load(path):
    t = pacsv.read_csv(path, read_options=pacsv.ReadOptions(block_size=64 << 20))
    t = t.take(pc.sort_indices(t, sort_keys=[("fact_time", "ascending")]))
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "shifts_weather"))
    ap.add_argument("--also_regression", action="store_true", help="also write <name>_{train,test}_reg.csv: raw float target as first column 'target' (for --task=regression runs)")
    ap.add_argument("--only_regression", action="store_true", help="with --also_regression: do not rewrite the classification CSVs (they may be in use)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    tar = os.path.join(a.out_dir, "canonical-partitioned-dataset.tar")
    if not os.path.exists(os.path.join(a.out_dir, D, "shifts_canonical_train.csv")):
        if not os.path.exists(tar):
            log("downloading " + URL); subprocess.check_call(["curl", "-sL", "-C", "-", "-o", tar, URL])
        log("extracting"); tarfile.open(tar).extractall(a.out_dir)
    t0 = time.time()
    train = load(os.path.join(a.out_dir, D, "shifts_canonical_train.csv"))
    test = pa.concat_tables([pacsv.read_csv(os.path.join(a.out_dir, D, f"shifts_canonical_{s}.csv")) for s in ("eval_in", "eval_out")])
    test = test.take(pc.sort_indices(test, sort_keys=[("fact_time", "ascending")]))
    log(f"loaded train {train.num_rows} test {test.num_rows} cols {train.num_columns} in {time.time()-t0:.0f}s")
    drop = ["fact_temperature", "fact_cwsm_class"]
    enc_train, cmap = ordinal_encode(train.column("climate"))
    enc_test = pc.cast(pc.index_in(test.column("climate"), value_set=pa.array(sorted(cmap, key=cmap.get))), pa.float32())
    Xtr = train.drop(drop).set_column(train.drop(drop).schema.get_field_index("climate"), "climate", enc_train)
    Xte = test.drop(drop).set_column(test.drop(drop).schema.get_field_index("climate"), "climate", enc_test)
    ytr = train.column("fact_temperature").to_numpy(zero_copy_only=False).astype(np.float64)
    yte = test.column("fact_temperature").to_numpy(zero_copy_only=False).astype(np.float64)
    thr = float(np.median(ytr))
    means = column_means(Xtr); nans_tr = nan_counts(Xtr); nans_te = nan_counts(Xte)
    out_tr = impute_and_binarize(Xtr, ytr, means, thr); out_te = impute_and_binarize(Xte, yte, means, thr)
    p_tr = os.path.join(a.out_dir, "shifts_weather_train.csv"); p_te = os.path.join(a.out_dir, "shifts_weather_test.csv")
    if not a.only_regression:
        write_csv_unquoted(out_tr, p_tr); write_csv_unquoted(out_te, p_te)
    if a.also_regression:
        for split, X, y in (("train", Xtr, ytr), ("test", Xte, yte)):
            pr = os.path.join(a.out_dir, f"shifts_weather_{split}_reg.csv"); write_csv_unquoted(impute_with_target(X, y, means), pr); log(f"wrote {pr}")
    log(f"wrote {p_tr} ({os.path.getsize(p_tr)/1e9:.1f} GB) and {p_te} in {time.time()-t0:.0f}s")
    meta = {"name": "shifts_weather", "source": URL, "license": "see LICENSE.md in the archive (Shifts Project)",
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_rule": f"class = 1 iff fact_temperature > train median {thr}", "train_median_fact_temperature": thr,
            "train": {"csv": p_tr, "rows": out_tr.num_rows, "positive_rate": float(np.mean(ytr > thr)), "source_split": "shifts_canonical_train.csv sorted by fact_time"},
            "test": {"csv": p_te, "rows": out_te.num_rows, "positive_rate": float(np.mean(yte > thr)), "source_split": "eval_in + eval_out sorted by fact_time"},
            "dropped_columns": {"fact_temperature": "target", "fact_cwsm_class": "observed weather-class label"},
            "encodings": {"climate": cmap, "fact_time": "epoch seconds stored fp32 (128 s grid)"},
            "imputation": {"rule": "NaN -> TRAIN column mean", "train_means": means, "train_nan_cells": nans_tr, "test_nan_cells": nans_te},
            "columns": out_tr.column_names}
    if not a.only_regression: write_meta(os.path.join(a.out_dir, "shifts_weather_meta.json"), meta)
    log(f"train positives {meta['train']['positive_rate']:.4f}, nan cells train {sum(nans_tr.values())} test {sum(nans_te.values())}")


if __name__ == "__main__":
    main()
