#!/usr/bin/env python3
"""Axis-aligned GBDT / RF baselines (XGBoost, LightGBM, CatBoost, scikit-learn)
on the same fold CSVs the SPO harness consumes, emitting CSVs in the shape of
benchmarks/utils/parse_log_to_csv.py (dataset,algorithm,fold_1..N,avg,std) so
accuracy_stats.py / make_paper_accuracy_tables.py ingest them unchanged.

Two input modes:
  --data_dir DIR   every DIR/task_*/repeat0_fold{k}_sample0_{train,test}.csv
                   (CC18 layout; make_folds.py writes TabArena/TabReD the same way)
  --train_csv A --test_csv B --dataset NAME   one held-out pair (HIGGS/SUSY/Epsilon)

Label = last CSV column (any 2 distinct values). Numeric columns -> float32;
non-numeric columns (CC18 has raw string categories, which YDF splits natively)
become pandas categoricals with train-fold categories and go to each library's
native categorical path (XGBoost enable_categorical, LightGBM category dtype,
CatBoost cat_features, sklearn-HGB from_dtype; sklearn-RF gets the integer codes).
NaN is left to each library (make_folds.py already imputes TabArena/TabReD).

Outputs per algorithm, in --out_dir:
  gbdt_<suffix>_<algo>.csv           held-out accuracy
  gbdt_<suffix>_<algo>_auc.csv       ROC AUC of the positive class
  gbdt_<suffix>_<algo>_logloss.csv   log loss (natural log)
  gbdt_<suffix>_<algo>_time.csv      fit wall-clock seconds (only meaningful on the
                                     benchmark machine; Mac numbers are for smoke tests)

Presets (--preset):
  matched  300 trees, depth 6, lr 0.1, min 5 examples/leaf, no row/col subsampling:
           the SPO-GBT setting used in the rebuttal, for a like-for-like comparison.
  default  library defaults (only seed / threads set): "what a practitioner gets".
"""
import argparse
import glob
import os
import platform
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT = os.path.join(REPO_ROOT, "benchmarks", "results", "accuracy")
ALGOS = ["xgb", "lgbm", "cat", "sk_hgb", "sk_rf"]


def load_pair(train_path, test_path):
    """(Xtr, ytr, Xte, yte) as DataFrames; categoricals share train-fold categories."""
    tr, te = pd.read_csv(train_path), pd.read_csv(test_path)
    label = tr.columns[-1]
    classes = sorted(pd.unique(tr[label].dropna()), key=str)
    if len(classes) != 2:
        raise ValueError(f"{train_path}: label has {len(classes)} classes, need 2")
    cmap = {c: i for i, c in enumerate(classes)}
    ys = [d[label].map(cmap).to_numpy(dtype=np.int8) for d in (tr, te)]
    Xtr, Xte = tr.drop(columns=[label]), te.drop(columns=[label])
    for c in Xtr.columns:
        if pd.api.types.is_numeric_dtype(Xtr[c]) and pd.api.types.is_numeric_dtype(Xte[c]):
            Xtr[c] = Xtr[c].astype(np.float32); Xte[c] = Xte[c].astype(np.float32)
        else:
            cats = pd.Index(sorted(Xtr[c].dropna().astype(str).unique()))
            Xtr[c] = pd.Categorical(Xtr[c].astype(str).where(Xtr[c].notna()), categories=cats)
            Xte[c] = pd.Categorical(Xte[c].astype(str).where(Xte[c].notna()), categories=cats)
    return Xtr, ys[0], Xte, ys[1]


def cat_cols(X):
    return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]


def make_model(algo, preset, threads, seed):
    """Returns (model, label) for one algorithm under one preset."""
    m = preset == "matched"
    if algo == "xgb":
        import xgboost as xgb
        kw = dict(tree_method="hist", n_jobs=threads, random_state=seed,
                  enable_categorical=True)
        if m:
            kw.update(n_estimators=300, max_depth=6, learning_rate=0.1,
                      subsample=1.0, colsample_bytree=1.0)
        return xgb.XGBClassifier(**kw), f"XGBoost_{xgb.__version__}"
    if algo == "lgbm":
        import lightgbm as lgb
        kw = dict(n_jobs=threads, random_state=seed, verbose=-1)
        if m:
            kw.update(n_estimators=300, max_depth=6, num_leaves=63,
                      learning_rate=0.1, min_child_samples=5,
                      subsample=1.0, colsample_bytree=1.0)
        return lgb.LGBMClassifier(**kw), f"LightGBM_{lgb.__version__}"
    if algo == "cat":
        import catboost as cb
        kw = dict(thread_count=threads, random_seed=seed, verbose=0,
                  allow_writing_files=False)
        if m:
            kw.update(iterations=300, depth=6, learning_rate=0.1,
                      bootstrap_type="No", rsm=1.0)
        return cb.CatBoostClassifier(**kw), f"CatBoost_{cb.__version__}"
    if algo == "sk_hgb":
        import sklearn
        from sklearn.ensemble import HistGradientBoostingClassifier
        kw = dict(random_state=seed, categorical_features="from_dtype")
        if m:
            kw.update(max_iter=300, max_depth=6, learning_rate=0.1,
                      min_samples_leaf=5, early_stopping=False)
        return HistGradientBoostingClassifier(**kw), f"sklearn-HGB_{sklearn.__version__}"
    if algo == "sk_rf":
        import sklearn
        from sklearn.ensemble import RandomForestClassifier
        kw = dict(n_jobs=threads, random_state=seed)
        if m:
            kw.update(n_estimators=240)
        return RandomForestClassifier(**kw), f"sklearn-RF_{sklearn.__version__}"
    raise ValueError(algo)


def fit_eval(algo, preset, threads, seed, Xtr, ytr, Xte, yte):
    model, label = make_model(algo, preset, threads, seed)
    cats = cat_cols(Xtr)
    fit_kw = {}
    if algo == "cat" and cats:  # CatBoost wants string cats, no NaN
        Xtr, Xte = Xtr.copy(), Xte.copy()
        for c in cats:
            Xtr[c] = Xtr[c].astype(object).fillna("NA").astype(str)
            Xte[c] = Xte[c].astype(object).fillna("NA").astype(str)
        fit_kw["cat_features"] = cats
    elif algo == "sk_rf" and cats:  # no native categoricals: ordinal codes
        Xtr, Xte = Xtr.copy(), Xte.copy()
        for c in cats:
            Xtr[c] = Xtr[c].cat.codes.astype(np.float32)
            Xte[c] = Xte[c].cat.codes.astype(np.float32)
    t0 = time.perf_counter()
    model.fit(Xtr, ytr, **fit_kw)
    fit_s = time.perf_counter() - t0
    p1 = np.clip(model.predict_proba(Xte)[:, 1].astype(np.float64), 1e-15, 1 - 1e-15)
    return dict(acc=accuracy_score(yte, p1 >= 0.5), auc=roc_auc_score(yte, p1),
                logloss=log_loss(yte, p1, labels=[0, 1]), time=fit_s), label


def provenance(args):
    def sh(cmd):
        try:
            return subprocess.check_output(cmd, shell=True, cwd=REPO_ROOT,
                                           stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return "unknown"
    dirty = "" if sh("git diff --quiet && echo clean") == "clean" else "-dirty"
    return "\n".join([
        "==== PROVENANCE ====",
        f"date_utc: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"git_sha: {sh('git rev-parse --short HEAD')}{dirty}",
        f"git_branch: {sh('git branch --show-current')}",
        f"machine: {platform.processor() or platform.machine()} "
        f"(nproc={os.cpu_count()}) host={socket.gethostname()}",
        f"EXTRA_BAZEL_CONFIGS: <none>",
        f"EXTRA_TRAIN_ARGS: gbdt_baselines.py --preset {args.preset} "
        f"--threads {args.threads} --seed={args.seed}",
        f"NUM_FOLDS: {args.folds}  NUM_TREES: {'300/240' if args.preset == 'matched' else 'library default'}",
        "====================",
    ])


def write_csv(path, header, rows, nfolds):
    cols = ["dataset", "algorithm"] + [f"fold_{i + 1}" for i in range(nfolds)] + ["avg", "std"]
    with open(path, "w") as f:
        f.write(header + "\n" + ",".join(cols) + "\n")
        for ds, algo_label, vals in rows:
            v = [x for x in vals if x is not None]
            cells = ["" if x is None else f"{x:.6f}" for x in vals]
            cells += [""] * (nfolds - len(cells))
            avg = f"{np.mean(v):.6f}" if v else ""
            std = f"{np.std(v, ddof=1):.6f}" if len(v) > 1 else ""
            f.write(",".join([ds, algo_label] + cells + [avg, std]) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", help="dir of task_*/ fold CSVs (CC18 layout)")
    ap.add_argument("--train_csv"); ap.add_argument("--test_csv")
    ap.add_argument("--dataset", help="dataset name for --train_csv mode")
    ap.add_argument("--suffix", required=True, help="output file tag, e.g. cc18_m7i")
    ap.add_argument("--algos", default="xgb,lgbm,cat", help=f"subset of {ALGOS}")
    ap.add_argument("--preset", choices=["matched", "default"], default="matched")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--only", nargs="*", help="task dir name substrings to include")
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    args = ap.parse_args()
    algos = [a for a in args.algos.split(",") if a]
    bad = [a for a in algos if a not in ALGOS]
    if bad:
        sys.exit(f"unknown algos {bad}; choose from {ALGOS}")

    # (dataset, [(train, test), ...]) jobs.
    jobs = []
    if args.data_dir:
        for d in sorted(glob.glob(os.path.join(args.data_dir, "task_*"))):
            name = os.path.basename(d)
            if args.only and not any(s in name for s in args.only):
                continue
            pairs = []
            for k in range(args.folds):
                tr = os.path.join(d, f"repeat0_fold{k}_sample0_train.csv")
                te = os.path.join(d, f"repeat0_fold{k}_sample0_test.csv")
                pairs.append((tr, te) if os.path.exists(tr) and os.path.exists(te) else None)
            if any(pairs):
                jobs.append((name, pairs))
    elif args.train_csv and args.test_csv:
        jobs.append((args.dataset or os.path.basename(args.train_csv),
                     [(args.train_csv, args.test_csv)]))
        args.folds = 1
    if not jobs:
        sys.exit("no datasets found (need --data_dir or --train_csv/--test_csv)")

    os.makedirs(args.out_dir, exist_ok=True)
    header = provenance(args)
    results = {a: {m: [] for m in ("acc", "auc", "logloss", "time")} for a in algos}
    labels = {}
    for ds, pairs in jobs:
        cache = {}
        for algo in algos:
            vals = {m: [] for m in results[algo]}
            for pair in pairs:
                if pair is None:
                    for m in vals: vals[m].append(None)
                    continue
                if pair not in cache:
                    cache[pair] = load_pair(*pair)
                Xtr, ytr, Xte, yte = cache[pair]
                try:
                    r, labels[algo] = fit_eval(algo, args.preset, args.threads, args.seed,
                                               Xtr, ytr, Xte, yte)
                except Exception as e:  # keep the sweep going; cell stays empty
                    print(f"  FAIL {ds} {algo} {os.path.basename(pair[0])}: {e}", flush=True)
                    r = {m: None for m in vals}
                for m in vals: vals[m].append(r[m])
            for m in vals:
                results[algo][m].append((ds, f"{labels.get(algo, algo)}_{args.preset}", vals[m]))
            acc = [v for v in vals["acc"] if v is not None]
            t = [v for v in vals["time"] if v is not None]
            print(f"{ds:55s} {algo:6s} acc={np.mean(acc) if acc else float('nan'):.4f} "
                  f"fit={np.sum(t):.1f}s", flush=True)
        # Flush after every dataset so a killed sweep still leaves usable files.
        for algo in algos:
            base = os.path.join(args.out_dir, f"gbdt_{args.suffix}_{algo}")
            for m, sfx in (("acc", ""), ("auc", "_auc"), ("logloss", "_logloss"), ("time", "_time")):
                write_csv(f"{base}{sfx}.csv", header, results[algo][m], args.folds)
    print("wrote", os.path.join(args.out_dir, f"gbdt_{args.suffix}_<algo>[_auc|_logloss|_time].csv"))


if __name__ == "__main__":
    main()
