#!/usr/bin/env python3
"""Fully numeric copies of the CC18 binary datasets, fold 0 (dataset policy, CLAUDE.md).

Per dataset: string columns -> ordinal codes (train-sorted distinct values, test values
unseen in train -> code -1), NaN cells -> train-fold column mean (all-NaN column -> 0.0),
label -> 0/1 with the majority train class = 0 (tabular_suite_prep.encode_binary_label).
Writes <out>/<folder>/{train,test}.csv and <out>/prep_report.csv (what changed per dataset).
Usage: prep_cc18_fold0.py --repo ~/yggdrasil-oblique-forests --out <dir>
"""
import argparse, csv, json, os, sys

import numpy as np
import pandas as pd


def load(path, label):
    df = pd.read_csv(path, na_values=["?", ""], keep_default_na=True, low_memory=False)
    y = df[label]
    X = df.drop(columns=[label])
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.expanduser("~/yggdrasil-oblique-forests"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.repo, "benchmarks", "data"))
    from tabular_suite_prep import encode_binary_label  # noqa: E402

    data = os.path.join(a.repo, "benchmarks", "data", "cc18_binary_csv")
    entries = json.load(open(os.path.join(a.repo, "benchmarks", "data", "cc18_manifest.json")))["entries"]
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for e in entries:
        if e["status"] != "ok":
            continue
        folder = f"task_{e['task_id']}_{e['folder_name']}"
        label = e["label_column"]
        tr = os.path.join(data, folder, "repeat0_fold0_sample0_train.csv")
        te = os.path.join(data, folder, "repeat0_fold0_sample0_test.csv")
        if not (os.path.exists(tr) and os.path.exists(te)):
            print("MISSING", folder, file=sys.stderr)
            continue
        Xtr, ytr = load(tr, label)
        Xte, yte = load(te, label)
        assert list(Xtr.columns) == list(Xte.columns), folder
        enc_cols, nan_tr, nan_te, allnan = [], 0, 0, 0
        unseen = 0
        for c in Xtr.columns:
            col_tr, col_te = Xtr[c], Xte[c]
            if not pd.api.types.is_numeric_dtype(col_tr) or not pd.api.types.is_numeric_dtype(col_te):
                s_tr, s_te = col_tr.astype("string"), col_te.astype("string")
                codes = {v: float(i) for i, v in enumerate(sorted(s_tr.dropna().unique()))}
                col_tr = s_tr.map(codes).astype("float64")
                mapped = s_te.map(codes)
                unseen += int((mapped.isna() & s_te.notna()).sum())
                col_te = mapped.fillna(-1.0).where(s_te.notna(), np.nan).astype("float64")
                enc_cols.append(c)
            col_tr = col_tr.astype("float64")
            col_te = col_te.astype("float64")
            n_tr, n_te = int(col_tr.isna().sum()), int(col_te.isna().sum())
            if n_tr or n_te:
                mean = float(col_tr.mean()) if n_tr < len(col_tr) else 0.0
                if n_tr == len(col_tr):
                    allnan += 1
                col_tr = col_tr.fillna(mean)
                col_te = col_te.fillna(mean)
                nan_tr += n_tr
                nan_te += n_te
            Xtr[c] = col_tr.astype("float32")
            Xte[c] = col_te.astype("float32")
        # Label tokens from the TRAIN class order; the test fold reuses the mapping.
        tok_tr, lstats = encode_binary_label(ytr)
        cls = lstats["classes"]
        ytr_s = ytr.astype("string") if not pd.api.types.is_numeric_dtype(ytr) else ytr
        yte_s = yte.astype("string") if not pd.api.types.is_numeric_dtype(yte) else yte
        key = {str(v): i for i, v in enumerate(cls)}
        tok_te = yte_s.map(lambda v: key[str(v)]).astype("int8")
        assert (ytr_s.map(lambda v: key[str(v)]).to_numpy() == tok_tr.to_numpy()).all()
        od = os.path.join(a.out, folder)
        os.makedirs(od, exist_ok=True)
        for name, X, t in (("train", Xtr, tok_tr), ("test", Xte, tok_te)):
            X = X.copy()
            X["label"] = t.to_numpy()
            X.to_csv(os.path.join(od, f"{name}.csv"), index=False, float_format="%.9g")
        rows.append(dict(dataset=folder, rows_train=len(Xtr), rows_test=len(Xte), F=Xtr.shape[1],
                         cols_encoded=len(enc_cols), encoded_cols=";".join(enc_cols),
                         nan_imputed_train=nan_tr, nan_imputed_test=nan_te, all_nan_cols=allnan,
                         test_unseen_categories=unseen, class0=cls[0], class1=cls[1],
                         pos_rate_train=round(lstats["positive_rate"], 4)))
        print(f"{folder:48s} F={Xtr.shape[1]:5d} enc={len(enc_cols):3d} nan={nan_tr + nan_te:6d} "
              f"unseen={unseen} label0={cls[0]} label1={cls[1]}", flush=True)
    with open(os.path.join(a.out, "prep_report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} datasets -> {a.out}")


if __name__ == "__main__":
    main()
