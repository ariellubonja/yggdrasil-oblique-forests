#!/usr/bin/env python3
"""Build NaN-preserving copies (train_nan.csv) of the suite CSVs so folds can be imputed per training fold.

The suite CSVs under benchmarks/data/{tabarena,tabred}_binary_csv/<name>/train.csv were produced by
tabular_suite_prep.py (pr356-replication branch), which imputed NaN cells with the FULL-dataset column mean before
any split. For accuracy claims, imputation must use the training fold only (PROTOCOL.md D3). This script rebuilds
the same matrix with the same encoding rules but leaves NaN cells empty, writing <name>/train_nan.csv next to
train.csv, and verifies that every non-NaN cell matches train.csv (float32) and that NaN counts match meta.json.

Sources:
  TabReD  : benchmarks/data/tabred_raw_mirror/<name>_train_full.csv (id, timestamp, num_*/bin_*/cat_*, target).
            Drop id/timestamp; every other column is a feature (categoricals ordinal-encoded as in the reference
            CSV); target -> label with the same majority-class->0 rule (must reproduce meta.json).
  TabArena: benchmarks/data/tabarena/<name>/{data.parquet,info.json} (OpenML re-download); features = all columns
            except the target; encoding per tabular_suite_prep.encode_features minus the imputation step.
Usage: make_train_nan.py [names...]   (default: every dataset whose meta.json has nan_cells_imputed > 0)
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SUITES = {"tabarena": os.path.join(REPO, "benchmarks/data/tabarena_binary_csv"),
          "tabred": os.path.join(REPO, "benchmarks/data/tabred_binary_csv")}


def encode_features_keep_nan(X: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for name in X.columns:
        col = X[name]
        if pd.api.types.is_bool_dtype(col):
            vals = col.astype("float32")
        elif pd.api.types.is_numeric_dtype(col):
            vals = col.astype("float32")
        elif pd.api.types.is_datetime64_any_dtype(col):
            vals = col.astype("int64").astype("float32") / 1e9
        else:
            s = col.astype("string")
            uniq = pd.Index(sorted(s.dropna().unique()))
            vals = s.map({v: float(i) for i, v in enumerate(uniq)}).astype("float32")
        out[name] = vals.astype("float32")
    return pd.DataFrame(out, index=X.index)


def encode_label(y: pd.Series) -> pd.Series:
    s = y.astype("string") if not pd.api.types.is_numeric_dtype(y) else y
    counts = s.value_counts(dropna=True)
    assert len(counts) == 2, counts
    order = list(counts.index)
    return s.map({order[0]: 0, order[1]: 1}).astype("int8")


def load_raw(suite: str, name: str) -> tuple[pd.DataFrame, pd.Series]:
    if suite == "tabred":
        raw = pd.read_csv(os.path.join(REPO, "benchmarks/data/tabred_raw_mirror", f"{name}_train_full.csv"))
        X = raw.drop(columns=[c for c in ("id", "timestamp", "target") if c in raw.columns])
        return X, raw["target"]
    ddir = os.path.join(REPO, "benchmarks/data/tabarena", name)
    info = json.load(open(os.path.join(ddir, "info.json")))
    df = pd.read_parquet(os.path.join(ddir, "data.parquet"))
    target = info["target_feature"]
    keep = [c for c in df.columns if c != target]
    return df[keep], df[target]


def build(suite: str, name: str) -> None:
    ddir = os.path.join(SUITES[suite], name)
    meta = json.load(open(os.path.join(ddir, "meta.json")))
    ref = pd.read_csv(os.path.join(ddir, "train.csv"), dtype="float32")
    X, y = load_raw(suite, name)
    feats = encode_features_keep_nan(X)
    lab = encode_label(y)
    # Column order must match the encoded reference (same source column order).
    if list(feats.columns) != list(ref.columns[:-1]):
        # tabarena parquet may carry the columns in the same order; if names differ, fall back to positional.
        if len(feats.columns) != len(ref.columns) - 1:
            raise SystemExit(f"{name}: column count mismatch {len(feats.columns)} vs {len(ref.columns)-1}")
        feats.columns = ref.columns[:-1]
    ref_X = ref.iloc[:, :-1].to_numpy(np.float32)
    new_X = feats.to_numpy(np.float32)
    nan = np.isnan(new_X)
    n_nan = int(nan.sum())
    if n_nan != int(meta["features_encoding"]["nan_cells_imputed"]):
        raise SystemExit(f"{name}: NaN count {n_nan} != meta {meta['features_encoding']['nan_cells_imputed']}")
    diff = np.abs(np.where(nan, 0, new_X) - np.where(nan, 0, ref_X))
    tol = 1e-6 * (1 + np.abs(ref_X))
    bad = int((diff > tol).sum())
    if bad:
        raise SystemExit(f"{name}: {bad} non-NaN cells differ from train.csv")
    if not np.array_equal(lab.to_numpy(np.int8), ref.iloc[:, -1].to_numpy(np.int8)):
        raise SystemExit(f"{name}: label mismatch")
    # Imputed cells must equal the full-column mean of the non-NaN cells (sanity: reproduces the old encoding).
    col_means = np.nanmean(new_X, axis=0)
    imputed = ref_X[nan]
    exp = np.broadcast_to(col_means, new_X.shape)[nan]
    if not np.allclose(imputed, exp, rtol=1e-5, atol=1e-6):
        print(f"  note: {name}: imputed cells differ from column means by up to {np.max(np.abs(imputed-exp)):.3g} (float32 mean rounding)")
    feats[meta["label_col"]] = lab.to_numpy()
    out = os.path.join(ddir, "train_nan.csv")
    feats.to_csv(out, index=False, float_format="%.9g", na_rep="")
    print(f"{suite}/{name}: rows={len(feats)} feats={feats.shape[1]-1} nan_cells={n_nan} -> {out}")


def main() -> int:
    want = set(sys.argv[1:])
    done = 0
    for suite, d in SUITES.items():
        for name in sorted(os.listdir(d)):
            mp = os.path.join(d, name, "meta.json")
            if not os.path.exists(mp):
                continue
            meta = json.load(open(mp))
            if meta["features_encoding"]["nan_cells_imputed"] == 0:
                continue
            if want and name not in want:
                continue
            src_ok = (suite == "tabred") or os.path.exists(os.path.join(REPO, "benchmarks/data/tabarena", name, "data.parquet"))
            if not src_ok:
                print(f"SKIP {suite}/{name}: raw source not on disk yet")
                continue
            build(suite, name); done += 1
    print(f"built {done} train_nan.csv files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
