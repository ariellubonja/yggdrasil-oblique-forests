#!/usr/bin/env python3
"""Emulate SPO normalization STANDARD_DEVIATION by pre-scaling fold CSVs (2026-11-21 probe).
Every feature column is divided by its training-fold std (ddof=0). The test fold reuses the train std.
Columns with std < 1e-6 stay unchanged. No centering. The label column is copied byte for byte.
Usage: scale_folds.py <folds_root> <out_root> <categoricals.csv> <ds>[,<ds>...]
Writes <out_root>/scaled/<ds>/fold<k>_{train,test}.csv + folds.json, <out_root>/orig/<ds> -> symlink, scale_stats.csv."""
import csv, json, os, shutil, sys

import numpy as np
import pandas as pd

src, out, cats_csv, dss = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
cats = {r["dataset"]: r for r in csv.DictReader(open(cats_csv))}
stats = []
for ds in dss:
    sd, od = os.path.join(src, ds), os.path.join(out, "scaled", ds)
    os.makedirs(od, exist_ok=True)
    os.makedirs(os.path.join(out, "orig"), exist_ok=True)
    link = os.path.join(out, "orig", ds)
    if not os.path.islink(link):
        os.symlink(sd, link)
    shutil.copy(os.path.join(sd, "folds.json"), od)
    nfold = len(json.load(open(os.path.join(sd, "folds.json")))["indices"])
    for k in range(nfold):
        tr = pd.read_csv(os.path.join(sd, f"fold{k}_train.csv"), dtype={"label": str})
        te = pd.read_csv(os.path.join(sd, f"fold{k}_test.csv"), dtype={"label": str})
        feats = [c for c in tr.columns if c != "label"]
        assert list(te.columns) == list(tr.columns), ds
        X = tr[feats].to_numpy(np.float64)
        assert not np.isnan(X).any() and not np.isnan(te[feats].to_numpy(np.float64)).any(), f"NaN in {ds} fold {k}"
        std = X.std(axis=0, ddof=0)
        keep = std < 1e-6
        div = np.where(keep, 1.0, std)
        ints = int(np.sum(np.all(X == np.round(X), axis=0)))
        for df, name in ((tr, "train"), (te, "test")):
            s = df.copy()
            for j, c in enumerate(feats):
                if not keep[j]:
                    s[c] = df[c].to_numpy(np.float64) / div[j]
            p = os.path.join(od, f"fold{k}_{name}.csv")
            s.to_csv(p, index=False)
            back = pd.read_csv(p, dtype={"label": str})  # round-trip check: unscale and compare
            assert (back["label"] == df["label"]).all()
            err = np.abs(back[feats].to_numpy(np.float64) * div - df[feats].to_numpy(np.float64))
            assert np.all(err <= 1e-9 * np.maximum(1.0, np.abs(df[feats].to_numpy(np.float64)))), f"{ds} {k} {name}"
        live = std[~keep]
        stats.append({"dataset": ds, "fold": k, "train_rows": len(tr), "test_rows": len(te), "cols": len(feats),
                      "n_ordinal_cols": cats[ds]["categorical"], "n_integer_valued_cols": ints,
                      "n_const_cols": int(keep.sum()), "std_min": float(live.min()), "std_max": float(live.max()),
                      "std_ratio_max_min": float(live.max() / live.min())})
        print(stats[-1], flush=True)
with open(os.path.join(out, "scale_stats.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(stats[0]))
    w.writeheader()
    w.writerows(stats)
