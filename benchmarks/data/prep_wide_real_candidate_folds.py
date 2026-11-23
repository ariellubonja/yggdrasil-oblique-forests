#!/usr/bin/env python3
"""Wide real-dataset candidates (kdd28_wide_real_candidates, 2026-11-22). Pins/rules: kdd28_wide_real_candidates_manifest.json.
Reuses encode_features_keep_nan, encode_binary_label and run_suite.make_folds unchanged. Pinned venv (numpy 2.5.2,
pandas 3.0.5, scikit-learn 1.9.0, pyarrow 25.0.1):
  python prep_wide_real_candidate_folds.py <out_dir> [--only a,b]
Writes <out_dir>/raw/<name>/, <out_dir>/src/<name>/train_nan.csv, <out_dir>/folds/<name>/{fold<k>_train,fold<k>_test}.csv,
folds.json, meta.json and <out_dir>/what_changed.csv. sha256 of raw + fold files checked against (or first recorded in)
kdd28_wide_real_candidates_manifest.sha256."""
import argparse, collections, csv, glob, gzip, hashlib, io, json, os, subprocess, sys, zipfile

import numpy as np
import pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(D))
sys.path.insert(0, D)
sys.path.insert(0, os.path.join(REPO, "benchmarks/src/spaa27_experiments"))
from tabular_suite_prep import encode_binary_label  # noqa: E402
from make_train_nan import encode_features_keep_nan  # noqa: E402
from run_suite import make_folds  # noqa: E402

MANIFEST = os.path.join(D, "kdd28_wide_real_candidates_manifest.json")
SHA = os.path.join(D, "kdd28_wide_real_candidates_manifest.sha256")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(url, out):
    if not os.path.isfile(out):
        subprocess.run(["curl", "-sSfL", "--retry", "5", "-o", out + ".part", url], check=True)
        os.replace(out + ".part", out)
    return out


def check_openml_pin(e, raw):
    meta_p = fetch(f"https://www.openml.org/api/v1/json/data/{e['openml_data_id']}", os.path.join(raw, "openml_data.json"))
    d = json.load(open(meta_p))["data_set_description"]
    got = {"name": d["name"], "version": int(d["version"]), "status": d["status"], "md5": d["md5_checksum"]}
    want = {"name": e["openml_name"], "version": e["openml_version"], "status": "active", "md5": e["openml_md5"]}
    if got != want:
        sys.exit(f"{e['name']}: OpenML pin drift, got {got}, want {want}")


# ---- loaders: (X, y, n_train or None, raw files). X = features only (ID columns dropped), y = raw label
def load_openml(e, raw):
    check_openml_pin(e, raw)
    p = fetch(e["parquet_url"], os.path.join(raw, f"dataset_{e['openml_data_id']}.pq"))
    df = pd.read_parquet(p)
    if list(df.shape) != e["openml_shape"]:
        sys.exit(f"{e['name']}: raw shape {list(df.shape)} != manifest {e['openml_shape']}")
    return df.drop(columns=[e["target"]] + e["drop_columns"]), df[e["target"]], None, [p]


def load_gisette(e, raw):
    z = zipfile.ZipFile(fetch(e["url"], os.path.join(raw, "gisette.zip")))
    read = lambda n: np.loadtxt(io.BytesIO(z.read(n)), dtype=np.int64, ndmin=2)  # noqa: E731
    Xtr, Xte = read("GISETTE/gisette_train.data"), read("GISETTE/gisette_valid.data")
    ytr, yte = read("GISETTE/gisette_train.labels").ravel(), read("gisette_valid.labels").ravel()
    cols = [f"f{j:04d}" for j in range(Xtr.shape[1])]
    X = pd.DataFrame(np.vstack([Xtr, Xte]), columns=cols)
    return X, pd.Series(np.concatenate([ytr, yte]), name="label"), len(Xtr), [z.filename]


def load_fashion(e, raw):
    files, arr = [], {}
    for k, url in e["urls"].items():
        p = fetch(url, os.path.join(raw, os.path.basename(url)))
        if hashlib.md5(open(p, "rb").read()).hexdigest() != e["md5_published"][k]:
            sys.exit(f"{e['name']}: {k} md5 != Zalando published md5")
        b = gzip.decompress(open(p, "rb").read())
        magic, n = int.from_bytes(b[:4], "big"), int.from_bytes(b[4:8], "big")
        arr[k] = (np.frombuffer(b, np.uint8, offset=16).reshape(n, 784) if magic == 2051 else np.frombuffer(b, np.uint8, offset=8))
        files.append(p)
    cols = [f"pixel{j + 1}" for j in range(784)]
    X = pd.DataFrame(np.vstack([arr["train_images"], arr["test_images"]]), columns=cols)
    y = pd.Series(np.concatenate([arr["train_labels"], arr["test_labels"]]).astype(np.int64), name="class")
    return X, y, len(arr["train_labels"]), files


def load_har(e, raw):
    outer = zipfile.ZipFile(fetch(e["url"], os.path.join(raw, "har.zip")))
    z = zipfile.ZipFile(io.BytesIO(outer.read("UCI HAR Dataset.zip")))
    read = lambda n: pd.read_csv(io.BytesIO(z.read(f"UCI HAR Dataset/{n}")), sep=r"\s+", header=None)  # noqa: E731
    Xtr, Xte = read("train/X_train.txt"), read("test/X_test.txt")
    y = pd.concat([read("train/y_train.txt")[0], read("test/y_test.txt")[0]], ignore_index=True).astype(np.int64)
    X = pd.concat([Xtr, Xte], ignore_index=True)
    X.columns = [f"f{j:03d}" for j in range(X.shape[1])]
    return X, y, len(Xtr), [outer.filename]


def load_isolet(e, raw):
    z = zipfile.ZipFile(fetch(e["url"], os.path.join(raw, "isolet.zip")))
    parts = []
    for n in ("isolet1+2+3+4.data.Z", "isolet5.data.Z"):  # Unix compress (.Z): gzip -dc reads it
        txt = subprocess.run(["gzip", "-dc"], input=z.read(n), capture_output=True, check=True).stdout
        parts.append(pd.read_csv(io.BytesIO(txt), header=None, skipinitialspace=True))
    df = pd.concat(parts, ignore_index=True)
    X = df.iloc[:, :-1].copy()
    X.columns = [f"f{j:03d}" for j in range(X.shape[1])]
    y = df.iloc[:, -1]
    if not np.all(y == np.round(y)):
        sys.exit(f"{e['name']}: non-integer class values")
    return X, y.astype(np.int64), len(parts[0]), [z.filename]


def load_ct(e, raw):
    z = zipfile.ZipFile(fetch(e["url"], os.path.join(raw, "ct_slices.zip")))
    df = pd.read_csv(io.BytesIO(z.read("slice_localization_data.csv")))
    return df.drop(columns=["reference"] + e["drop_columns"]), df["reference"], None, [z.filename]


def load_blog(e, raw):
    z = zipfile.ZipFile(fetch(e["url"], os.path.join(raw, "blogfeedback.zip")))
    read = lambda n: pd.read_csv(io.BytesIO(z.read(n)), header=None)  # noqa: E731
    tests = sorted(n for n in z.namelist() if n.startswith("blogData_test-"))
    if len(tests) != 60:
        sys.exit(f"{e['name']}: {len(tests)} test files, expected 60")
    tr = read("blogData_train.csv")
    df = pd.concat([tr] + [read(n) for n in tests], ignore_index=True)
    X = df.iloc[:, :-1].copy()
    X.columns = [f"f{j:03d}" for j in range(X.shape[1])]
    return X, df.iloc[:, -1].astype("float64"), len(tr), [z.filename]


LOADERS = {"openml": load_openml, "uci_gisette": load_gisette, "zalando_idx": load_fashion, "uci_har": load_har,
           "uci_isolet": load_isolet, "uci_ct_slices": load_ct, "uci_blogfeedback": load_blog}


def build(e, out):
    name = e["name"]
    raw, src, fdir = (os.path.join(out, d, name) for d in ("raw", "src", "folds"))
    for d in (raw, src):
        os.makedirs(d, exist_ok=True)
    X, y, n_train, raw_files = LOADERS[e["source"]](e, raw)
    want_rows = e["expected_rows"] if isinstance(e["expected_rows"], int) else sum(e["expected_rows"].values())
    if X.shape != (want_rows, e["expected_features"]):
        sys.exit(f"{name}: features {X.shape} != manifest ({want_rows}, {e['expected_features']})")
    if n_train is not None and n_train != e["expected_rows"]["train"]:
        sys.exit(f"{name}: {n_train} train rows != manifest {e['expected_rows']['train']}")
    if y.isna().any():
        sys.exit(f"{name}: label has {int(y.isna().sum())} NaN")
    # categorical/string columns that get ordinal codes (bool and datetime have their own rules, counted apart)
    is_cat = lambda c: not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])  # noqa: E731
                            or pd.api.types.is_datetime64_any_dtype(X[c]))
    ncat = sum(is_cat(c) for c in X.columns)
    nbool = sum(pd.api.types.is_bool_dtype(X[c]) for c in X.columns)
    ndt = sum(pd.api.types.is_datetime64_any_dtype(X[c]) for c in X.columns)
    feats = encode_features_keep_nan(X)
    reg = e["task"] == "regression"
    if reg:
        lab, lmeta = y.astype("float64"), {"task": "regression", "mean": float(y.mean()), "std": float(y.std())}
    else:
        lab, lmeta = encode_binary_label(y)
        if (len(lmeta["classes"]) == 2 and "multiclass" not in lmeta) != (e["task"] == "binary"):
            sys.exit(f"{name}: manifest task {e['task']} but label has classes {lmeta}")
    feats["label"] = lab.to_numpy()
    feats.to_csv(os.path.join(src, "train_nan.csv"), index=False, float_format="%.9g", na_rep="")
    n = len(feats)
    if n_train is None:
        make_folds(name, os.path.join(src, "train.csv"), "label", 5, 0, os.path.join(out, "folds"), regression=reg)
        protocol = "5 CV folds, " + ("KFold" if reg else "StratifiedKFold") + "(shuffle=True, random_state=0)"
    else:
        frac = (n - n_train) / n
        if int(round(n * (1 - frac))) != n_train:
            sys.exit(f"{name}: holdout_frac {frac} does not reproduce {n_train} train rows")
        make_folds(name, os.path.join(src, "train.csv"), "label", 5, 0, os.path.join(out, "folds"), chrono=True,
                   holdout_frac=frac, regression=reg)
        protocol = f"shipped split, one holdout fold k=0: {n_train} train / {n - n_train} test rows"
    fj = json.load(open(os.path.join(fdir, "folds.json")))
    num = feats.drop(columns="label")
    meta = {"dataset": name, "source": e["source"], "pin": {k: e[k] for k in e if k.startswith("openml_") or k in ("url", "urls", "md5_published")},
            "raw_sha256": {os.path.basename(p): sha256(p) for p in raw_files}, "task": e["task"], "rows": n,
            "features": num.shape[1], "dropped_columns": e["drop_columns"], "categorical_encoded": int(ncat), "bool_columns": int(nbool), "datetime_columns": int(ndt),
            "categorical_columns": [c for c in X.columns if is_cat(c)],
            "nan_cells": int(num.isna().sum().sum()),
            "nan_cells_imputed_per_fold": {k: v["n_nan_cells_imputed_train"] + v["n_nan_cells_imputed_test"] for k, v in fj["indices"].items()},
            "constant_columns": int((num.nunique(dropna=False) <= 1).sum()),
            "duplicate_columns": int(num.T.duplicated().sum()) if num.shape[1] * n < 2e8 else None,
            "label": lmeta, "folds": protocol, "n_train": n_train, "note": e.get("note", "")}
    json.dump(meta, open(os.path.join(fdir, "meta.json"), "w"), indent=1)
    print(f"built {name}: rows={n} feats={num.shape[1]} cat={ncat} nan={meta['nan_cells']} {protocol} label={lmeta.get('counts', '')}", flush=True)
    return meta, raw_files


def outputs(e, out, raw_files):
    name, fdir = e["name"], os.path.join(out, "folds", e["name"])
    k = 1 if e["folds"] == "shipped" else 5
    files = {f"raw/{name}/{os.path.basename(p)}": p for p in raw_files}
    files |= {f"folds/{name}/fold{i}_{s}.csv": os.path.join(fdir, f"fold{i}_{s}.csv") for i in range(k) for s in ("train", "test")}
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    out = os.path.abspath(a.out_dir)
    only = set(filter(None, a.only.split(",")))
    todo = [e for e in json.load(open(MANIFEST))["datasets"] if not only or e["name"] in only]
    have = {}
    if os.path.isfile(SHA):
        for line in open(SHA):
            h, k = line.split()
            have[k] = h
    rows, ok, bad, new = [], 0, 0, 0
    for e in todo:
        meta, raw_files = build(e, out)
        lab = meta["label"]
        rule = ("float label passed through" if e["task"] == "regression" else
                f"majority-vs-rest, positive = class {lab['multiclass']['positive_class']} of {lab['multiclass']['n_classes']}"
                if "multiclass" in lab else f"two classes, majority {lab['classes'][0]} = 0, {lab['classes'][1]} = 1")
        counts = (f"mean {lab['mean']:.4g} sd {lab['std']:.4g}" if e["task"] == "regression" else f"0: {lab['counts'][0]}, 1: {lab['counts'][1]}")
        pin = (f"OpenML {e['openml_data_id']} v{e['openml_version']} md5 {e['openml_md5']}" if e["source"] == "openml"
               else "; ".join(f"{k} sha256 {v}" for k, v in meta["raw_sha256"].items()))
        rows.append({"dataset": e["name"], "task": e["task"], "source_pin": pin, "rows": meta["rows"], "features": meta["features"],
                     "id_columns_dropped": " ".join(e["drop_columns"]) or "-", "categorical_encoded": meta["categorical_encoded"],
                     "nan_cells_imputed": meta["nan_cells"], "constant_columns": meta["constant_columns"],
                     "duplicate_columns": meta["duplicate_columns"], "label_rule": rule, "label_counts": counts, "fold_protocol": meta["folds"]})
        for k, p in outputs(e, out, raw_files).items():
            h = sha256(p)
            if k not in have:
                have[k], new = h, new + 1
            elif have[k] == h:
                ok += 1
            else:
                bad += 1
                print("MISMATCH", k)
    with open(os.path.join(out, "what_changed.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if new:
        with open(SHA, "w") as f:
            for k in sorted(have):
                f.write(f"{have[k]}  {k}\n")
    print(f"sha256 vs {os.path.relpath(SHA, REPO)}: {ok} match, {bad} mismatch, {new} new (recorded)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
