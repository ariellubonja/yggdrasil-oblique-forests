#!/usr/bin/env python3
"""Self-check of the kdd28_wide_real_candidates fold files before any training (2026-11-22).
  python check_wide_real_candidate_folds.py <out_dir> [--only a,b]
Per dataset: raw re-parsed from <out_dir>/raw, then every fold file is checked for: numeric only, no NaN/inf, header =
source features minus the listed ID columns + label, row coverage (folds.json indices), labels re-derived here
(not via encode_binary_label), ordinal codes = rank among sorted distinct strings, sampled raw numeric cells equal
float32(raw), NaN cells = training-fold mean, shapes = manifest. Exit 1 on any failure."""
import argparse, collections, json, os, sys

import numpy as np
import pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
import prep_wide_real_candidate_folds as prep  # noqa: E402  (raw parsing only)


def check(e, out, rng):
    name, fails, notes = e["name"], [], []
    fdir, raw = os.path.join(out, "folds", name), os.path.join(out, "raw", name)
    meta, fj = json.load(open(os.path.join(fdir, "meta.json"))), json.load(open(os.path.join(fdir, "folds.json")))
    X, y, n_train, _ = prep.LOADERS[e["source"]](e, raw)
    n, F = X.shape
    want_rows = e["expected_rows"] if isinstance(e["expected_rows"], int) else sum(e["expected_rows"].values())
    if (n, F) != (want_rows, e["expected_features"]) or (meta["rows"], meta["features"]) != (n, F):
        fails.append(f"shape {(n, F)} vs manifest {(want_rows, e['expected_features'])} vs meta {(meta['rows'], meta['features'])}")
    # labels, re-derived: Counter over str(value); 2 classes -> majority 0, else 1 iff majority (ties: smaller str)
    if e["task"] == "regression":
        want_lab = y.to_numpy(dtype="float64")
    else:
        sv = [str(v) for v in y.tolist()]
        cnt = collections.Counter(sv)
        top = sorted(cnt, key=lambda v: (-cnt[v], v))[0]
        want_lab = np.array([(v != top) if len(cnt) == 2 else (v == top) for v in sv], dtype=np.int64)
        notes.append(f"classes {len(cnt)}, majority '{top}' ({cnt[top]}), label1 {int(want_lab.sum())}")
    cats = [c for c in X.columns if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]))]
    nums = [c for c in X.columns if c not in cats]
    cat_s = list(rng.choice(cats, min(5, len(cats)), replace=False)) if cats else []
    num_s = list(rng.choice(nums, min(20, len(nums)), replace=False)) if nums else []
    codes = {c: {v: float(i) for i, v in enumerate(sorted(set(X[c].dropna().astype(str))))} for c in cat_s}
    nan_cols = [c for c in X.columns if X[c].isna().any()]
    raw32 = {c: X[c].to_numpy(dtype="float32") for c in set(num_s) | set(nan_cols) if c in nums}
    seen_test = []
    for k, rec in sorted(fj["indices"].items(), key=lambda kv: int(kv[0])):
        if "train_idx" in rec:
            tr, te = np.array(rec["train_idx"]), np.array(rec["test_idx"])
        else:
            tr, te = np.arange(*rec["train_range"]), np.arange(*rec["test_range"])
            if len(tr) != n_train:
                fails.append(f"fold {k}: holdout train rows {len(tr)} != shipped {n_train}")
        if len(np.intersect1d(tr, te)) or len(tr) + len(te) != n:
            fails.append(f"fold {k}: train/test overlap or rows lost")
        seen_test.append(te)
        for part, idx in (("train", tr), ("test", te)):
            df = pd.read_csv(os.path.join(fdir, f"fold{k}_{part}.csv"))
            if list(df.columns) != list(X.columns) + ["label"]:
                fails.append(f"fold {k} {part}: header != source features + label")
                continue
            vals = df.to_numpy()
            if any(df[c].dtype.kind not in "iuf" for c in df.columns) or not np.isfinite(vals.astype("float64")).all():
                fails.append(f"fold {k} {part}: non-numeric, NaN or inf cells")
            if len(df) != len(idx):
                fails.append(f"fold {k} {part}: {len(df)} rows != {len(idx)} indices")
                continue
            lab = df["label"].to_numpy()
            ok = np.allclose(lab, want_lab[idx], rtol=1e-8, atol=0) if e["task"] == "regression" else np.array_equal(lab, want_lab[idx])
            if not ok:
                fails.append(f"fold {k} {part}: label mismatch")
            for c in cat_s:
                r = X[c].iloc[idx]
                m = r.notna().to_numpy()
                if not np.array_equal(df[c].to_numpy()[m], r[m].astype(str).map(codes[c]).to_numpy(dtype="float64")):
                    fails.append(f"fold {k} {part}: ordinal codes of {c} != rank among sorted distinct values")
            for c in num_s:
                r = raw32[c][idx]
                m = ~np.isnan(r)
                if not np.array_equal(df[c].to_numpy(dtype="float64")[m].astype("float32"), r[m]):
                    fails.append(f"fold {k} {part}: raw values of {c} changed")
            for c in nan_cols:  # imputed = train-fold mean of the float32 values (written %.9g)
                if c not in raw32:
                    continue
                trv = raw32[c][tr].astype("float64")
                mu = float(np.nanmean(trv)) if (~np.isnan(trv)).any() else 0.0
                m = np.isnan(raw32[c][idx])
                if m.any() and not np.allclose(df[c].to_numpy()[m], mu, rtol=1e-7, atol=1e-12):
                    fails.append(f"fold {k} {part}: imputed cells of {c} != train-fold mean {mu}")
    if "train_idx" in next(iter(fj["indices"].values())):
        allte = np.sort(np.concatenate(seen_test))
        if not np.array_equal(allte, np.arange(n)):
            fails.append("CV test folds do not partition the rows")
    notes.append(f"folds {len(fj['indices'])}, cat sampled {len(cat_s)}/{len(cats)}, num sampled {len(num_s)}, NaN columns {len(nan_cols)}")
    return fails, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    only = set(filter(None, a.only.split(",")))
    rng, bad = np.random.default_rng(0), 0
    for e in json.load(open(prep.MANIFEST))["datasets"]:
        if only and e["name"] not in only:
            continue
        fails, notes = check(e, os.path.abspath(a.out_dir), rng)
        bad += bool(fails)
        print(f"{'FAIL' if fails else 'OK  '} {e['name']}: " + "; ".join(notes + fails), flush=True)
    print(f"self-check: {bad} dataset(s) failed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
