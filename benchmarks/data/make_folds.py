#!/usr/bin/env python3
"""Turn process_datasets.py output (X.npy / y.npy / info.json, NaN kept) into the
CC18 fold-CSV layout that accuracy.sh and gbdt_baselines.py both consume:

    <out>/task_<benchmark>_<dataset>/repeat0_fold{k}_sample0_{train,test}.csv

Splits: TabReD ships time-based upstream splits (train/val/test) -> one fold,
train = train+val, test = test.  TabArena -> stratified K-fold (shuffled, fixed
seed).  Missing values: every NaN cell is imputed with the mean of that column
over the fold's *training* rows (test rows use the same means; an all-NaN column
gets 0).  This is the na_replacement rule YDF's oblique splits apply anyway, so
SPO-YDF and the GBDT baselines see identical, NaN-free matrices.

Binary tasks only by default: the SPO harness has no regression path.  Pass
--include-regression to also emit regression folds (label = raw float) for the
GBDT runner.  A folds_manifest.csv records rows / features / imputed cells.
"""
import argparse
import csv
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_IN = os.path.join(REPO_ROOT, "benchmarks", "data", "processed")
DEFAULT_OUT = os.path.join(REPO_ROOT, "benchmarks", "data", "tabular_folds")
BINARY_TASKS = {"binary", "binclass"}


def safe_name(s):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(s))


def impute(Xtr, Xte):
    """Train-column means fill NaN in both splits; returns (Xtr, Xte, #imputed)."""
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN column -> 0
        means = np.nanmean(Xtr, axis=0)
    means = np.where(np.isfinite(means), means, 0.0).astype(np.float32)
    n = 0
    out = []
    for X in (Xtr, Xte):
        m = np.isnan(X)
        n += int(m.sum())
        X = X.copy()
        X[m] = np.broadcast_to(means, X.shape)[m]
        out.append(X)
    return out[0], out[1], n


def write_split(path, X, y, names, label_name):
    df = pd.DataFrame(X, columns=names)
    df[label_name] = y
    df.to_csv(path, index=False, float_format="%.9g")


def folds_for(ddir, y, task, k, seed):
    """Yields (train_idx, test_idx). TabReD upstream splits win when present."""
    sdir = os.path.join(ddir, "splits")
    if os.path.isdir(sdir):
        splits = sorted(os.listdir(sdir))
        sp = os.path.join(sdir, splits[0])
        tr = np.load(os.path.join(sp, "train_idx.npy"))
        va = os.path.join(sp, "val_idx.npy")
        if os.path.exists(va):
            tr = np.concatenate([tr, np.load(va)])
        te = np.load(os.path.join(sp, "test_idx.npy"))
        yield np.sort(tr), np.sort(te)
        return
    if task in BINARY_TASKS:
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    else:
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    for tr, te in kf.split(np.zeros(len(y)), y):
        yield tr, te


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", default=DEFAULT_IN)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--include-regression", action="store_true")
    ap.add_argument("--min-rows", type=int, default=0, help="skip smaller datasets")
    args = ap.parse_args()

    recs = []
    for bench in sorted(os.listdir(args.processed)) if os.path.isdir(args.processed) else []:
        bdir = os.path.join(args.processed, bench)
        for name in sorted(os.listdir(bdir), key=str.lower):
            ddir = os.path.join(bdir, name)
            info_p = os.path.join(ddir, "info.json")
            if not os.path.exists(info_p) or (args.only and name not in args.only):
                continue
            info = json.load(open(info_p))
            task = info["task"]
            if task not in BINARY_TASKS and not args.include_regression:
                print(f"[skip] {bench}/{name}: {task} (harness is binary-only)")
                continue
            X = np.load(os.path.join(ddir, "X.npy"), mmap_mode="r")
            y = np.load(os.path.join(ddir, "y.npy"))
            if len(y) < args.min_rows:
                print(f"[skip] {bench}/{name}: {len(y)} rows < --min-rows")
                continue
            names = [safe_name(c) for c in info.get("feature_names", [])]
            if len(names) != X.shape[1] or len(set(names)) != len(names):
                names = [f"f{i}" for i in range(X.shape[1])]
            tdir = os.path.join(args.out, f"task_{bench}_{safe_name(name)}")
            os.makedirs(tdir, exist_ok=True)
            n_imp = 0
            nf = 0
            for k, (tr, te) in enumerate(folds_for(ddir, y, task, args.folds, args.seed)):
                Xtr, Xte, n = impute(np.asarray(X[tr]), np.asarray(X[te]))
                n_imp += n
                write_split(os.path.join(tdir, f"repeat0_fold{k}_sample0_train.csv"),
                            Xtr, y[tr], names, "Class")
                write_split(os.path.join(tdir, f"repeat0_fold{k}_sample0_test.csv"),
                            Xte, y[te], names, "Class")
                nf += 1
            recs.append(dict(benchmark=bench, dataset=name, task=task, n_rows=len(y),
                             n_features=X.shape[1], n_folds=nf,
                             split="upstream" if os.path.isdir(os.path.join(ddir, "splits"))
                             else f"stratified{args.folds}" if task in BINARY_TASKS
                             else f"kfold{args.folds}",
                             imputed_cells=n_imp,
                             pct_imputed=100.0 * n_imp / (nf * X.size) if X.size else 0.0))
            print(f"[ok]   {bench}/{name}: {len(y):,}x{X.shape[1]} {nf} folds, "
                  f"{recs[-1]['pct_imputed']:.3f}% cells imputed -> {tdir}")
    if not recs:
        sys.exit("nothing written")
    mp = os.path.join(args.out, "folds_manifest.csv")
    with open(mp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader(); w.writerows(recs)
    print(f"wrote {mp} ({len(recs)} datasets)")


if __name__ == "__main__":
    main()
