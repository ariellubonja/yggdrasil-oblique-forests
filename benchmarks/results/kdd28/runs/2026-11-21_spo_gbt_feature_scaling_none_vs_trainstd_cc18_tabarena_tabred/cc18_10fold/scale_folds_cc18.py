#!/usr/bin/env python3
"""CC18 port of the 2026-11-21 probe's scale_folds.py: emulate SPO normalization STANDARD_DEVIATION by pre-scaling folds.
Every feature column is divided by its training-fold std (ddof=0), test reuses it, std < 1e-6 unchanged, no centering.
Label column (meta.json label_col) copied untouched. meta*.json copied so the driver's cc18 table reads the scaled dir.
Usage: scale_folds_cc18.py <cc18_10fold root> <out_root>  -> <out>/scaled/<ds>/..., <out>/orig/<ds> symlink, scale_stats.csv"""
import csv, glob, json, os, shutil, sys

import numpy as np
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
dss = sorted(d for d in os.listdir(src) if os.path.isfile(os.path.join(src, d, "meta.json")))
stats = []
for ds in dss:
    sd, od = os.path.join(src, ds), os.path.join(out, "scaled", ds)
    os.makedirs(od, exist_ok=True)
    os.makedirs(os.path.join(out, "orig"), exist_ok=True)
    link = os.path.join(out, "orig", ds)
    if not os.path.islink(link):
        os.symlink(sd, link)
    for m in glob.glob(os.path.join(sd, "meta*.json")):
        shutil.copy(m, od)
    lbl = json.load(open(os.path.join(sd, "meta.json")))["label_col"]
    for k in range(10):
        meta = json.load(open(os.path.join(sd, "meta.json" if k == 0 else f"meta_fold{k}.json")))
        tr = pd.read_csv(os.path.join(sd, f"fold{k}_train.csv"), dtype={lbl: str}, keep_default_na=False)
        te = pd.read_csv(os.path.join(sd, f"fold{k}_test.csv"), dtype={lbl: str}, keep_default_na=False)
        feats = [c for c in tr.columns if c != lbl]
        assert list(te.columns) == list(tr.columns), ds
        X = tr[feats].to_numpy(np.float64)
        assert not np.isnan(X).any() and not np.isnan(te[feats].to_numpy(np.float64)).any(), f"NaN in {ds} fold {k}"
        std = X.std(axis=0, ddof=0)
        keep = std < 1e-6
        div = np.where(keep, 1.0, std)
        for df, name in ((tr, "train"), (te, "test")):
            s = df.copy()
            for j, c in enumerate(feats):
                if not keep[j]:
                    s[c] = df[c].to_numpy(np.float64) / div[j]
            p = os.path.join(od, f"fold{k}_{name}.csv")
            s.to_csv(p, index=False)
            back = pd.read_csv(p, dtype={lbl: str}, keep_default_na=False)  # round trip: unscale and compare
            assert (back[lbl] == df[lbl]).all()
            err = np.abs(back[feats].to_numpy(np.float64) * div - df[feats].to_numpy(np.float64))
            assert np.all(err <= 1e-9 * np.maximum(1.0, np.abs(df[feats].to_numpy(np.float64)))), f"{ds} {k} {name}"
        live = std[~keep]
        enc = meta["features_encoding"]
        stats.append({"dataset": ds, "fold": k, "train_rows": len(tr), "test_rows": len(te), "cols": len(feats),
                      "n_ordinal_cols": enc["n_categorical_encoded"], "n_bool_cols": enc["n_bool"],
                      "n_integer_valued_cols": int(np.sum(np.all(X == np.round(X), axis=0))),
                      "n_const_cols": int(keep.sum()), "std_min": float(live.min()), "std_max": float(live.max()),
                      "std_ratio_max_min": float(live.max() / live.min())})
        print(stats[-1], flush=True)
with open(os.path.join(out, "scale_stats.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(stats[0]))
    w.writeheader()
    w.writerows(stats)
