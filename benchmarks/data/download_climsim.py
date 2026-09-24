#!/usr/bin/env python3
"""ClimSim low-res, the paper's subsampled + pre-normalized tabular split -> binary CSVs.

Source: HuggingFace LEAP/subsampled_low_res (CC-BY-4.0; the 744 GB raw ClimSim_low-res netCDF set
subsampled and normalized by the ClimSim authors, https://arxiv.org/abs/2306.08754). Files used:
train_input.parquet 10,091,520 x 124, train_target.parquet 10,091,520 x 128, val_input/val_target
1,441,920 rows (the paper's validation split, used here as the test split). Columns are unnamed
("0".."123"); the ClimSim ordering gives them names:
  inputs : state_t_0..59, state_q0001_0..59, state_ps, pbuf_SOLIN, pbuf_LHFLX, pbuf_SHFLX
  targets: ptend_t_0..59, ptend_q0001_0..59, cam_out_NETSW, cam_out_FLWDS, cam_out_PRECSC,
           cam_out_PRECC, cam_out_SOLS, cam_out_SOLL, cam_out_SOLSD, cam_out_SOLLD
The task is multi-output regression; per the user's instruction (2026-09-24) we regress on ONE target:
`cam_out_FLWDS` (target index 121, downward longwave flux at the surface) - chosen because it is
continuous with no zero-inflation (NETSW/PRECC/SOL* are 0 for ~50 % of rows: night / no rain;
ptend_q0001 levels 60-71 are identically 0). Override with --target_index.
Massage: class = 1 iff target > median(target over TRAIN); all 124 inputs kept as fp32 features
(pre-normalized, no NaNs); row order as in the parquet (time-major: the authors wrote the
subsampled timesteps in order, 384 grid columns per timestep).
Outputs (benchmarks/data/climsim/, gitignored): climsim_train.csv, climsim_test.csv, climsim_meta.json.
"""
import argparse, os, subprocess, sys, time
import numpy as np, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from large_csv_utils import *

BASE = "https://huggingface.co/datasets/LEAP/subsampled_low_res/resolve/main/"
INPUTS = [f"state_t_{i}" for i in range(60)] + [f"state_q0001_{i}" for i in range(60)] + ["state_ps", "pbuf_SOLIN", "pbuf_LHFLX", "pbuf_SHFLX"]
TARGETS = [f"ptend_t_{i}" for i in range(60)] + [f"ptend_q0001_{i}" for i in range(60)] + ["cam_out_NETSW", "cam_out_FLWDS", "cam_out_PRECSC", "cam_out_PRECC", "cam_out_SOLS", "cam_out_SOLL", "cam_out_SOLSD", "cam_out_SOLLD"]


def fetch(out_dir, f):
    p = os.path.join(out_dir, f)
    if not os.path.exists(p):
        log("downloading " + f); subprocess.check_call(["curl", "-sL", "-C", "-", "-o", p, BASE + f])
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "climsim"))
    ap.add_argument("--target_index", type=int, default=121)
    ap.add_argument("--also_regression", action="store_true", help="also write <name>_{train,test}_reg.csv: raw float target as first column 'target' (for --task=regression runs)")
    ap.add_argument("--only_regression", action="store_true", help="with --also_regression: do not rewrite the classification CSVs (they may be in use)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    tname = TARGETS[a.target_index]
    t0 = time.time()
    res = {}
    for split, out in (("train", "train"), ("val", "test")):
        X = pq.read_table(fetch(a.out_dir, f"{split}_input.parquet")).rename_columns(INPUTS)
        y = pq.read_table(fetch(a.out_dir, f"{split}_target.parquet"), columns=[str(a.target_index)]).column(0).to_numpy(zero_copy_only=False).astype(np.float64)
        res[out] = (X, y); log(f"loaded {split}: {X.num_rows} rows in {time.time()-t0:.0f}s")
    Xtr, ytr = res["train"]; Xte, yte = res["test"]
    thr = float(np.median(ytr)); means = column_means(Xtr); nans_tr = nan_counts(Xtr); nans_te = nan_counts(Xte)
    paths = {}
    for out, (X, y) in res.items():
        tab = impute_and_binarize(X, y, means, thr); p = os.path.join(a.out_dir, f"climsim_{out}.csv"); paths[out] = (p, tab.num_rows, float(np.mean(y > thr)))
        if not a.only_regression:
            write_csv_unquoted(tab, p); log(f"wrote {p} ({os.path.getsize(p)/1e9:.1f} GB) in {time.time()-t0:.0f}s")
        if a.also_regression:
            pr = os.path.join(a.out_dir, f"climsim_{out}_reg.csv"); write_csv_unquoted(impute_with_target(X, y, means), pr); log(f"wrote {pr}")
    meta = {"name": "climsim", "source": BASE, "license": "CC-BY-4.0", "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target": {"name": tname, "index": a.target_index, "rule": f"class = 1 iff {tname} > train median {thr}", "train_median": thr,
                       "why": "single continuous, non-zero-inflated output; user directive: regress on one target"},
            "train": {"csv": paths["train"][0], "rows": paths["train"][1], "positive_rate": paths["train"][2], "source": "train_input/train_target.parquet"},
            "test": {"csv": paths["test"][0], "rows": paths["test"][1], "positive_rate": paths["test"][2], "source": "val_input/val_target.parquet"},
            "features": INPUTS, "dropped": "the other 127 target columns", "imputation": {"rule": "NaN -> TRAIN column mean", "train_nan_cells": nans_tr, "test_nan_cells": nans_te, "train_means": means}}
    if not a.only_regression: write_meta(os.path.join(a.out_dir, "climsim_meta.json"), meta)
    log(f"train positives {paths['train'][2]:.4f}; nan cells train {sum(nans_tr.values())} test {sum(nans_te.values())}")


if __name__ == "__main__":
    main()
