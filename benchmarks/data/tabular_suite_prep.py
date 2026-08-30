#!/usr/bin/env python3
"""Shared preprocessing for external tabular suites (TabArena, TabReD).

The oblique harness' fast CSV loader (`single_pass_csv::Load` in
examples/train_oblique_forest.cc) reads every column as NUMERICAL in one pass and
only afterwards converts the label to a binary CATEGORICAL. A single non-numeric
cell aborts that path and silently falls back to the two-read loader, which
changes what the timing measures. These suites ship categorical, boolean and
datetime columns, so we materialise an all-numeric CSV once, here.

Preprocessing (identical for every arm of an A/B, so timing deltas stay
attributable to the compiled kernel, never to the data):

  features  numeric        -> float32 as-is
            bool           -> 0.0 / 1.0
            datetime       -> epoch seconds (float)
            categorical /
            object/string  -> ordinal codes over the *sorted* unique values
                              (sorted, not first-seen, so the encoding is
                              reproducible across pandas versions and row order)
  missing   NaN            -> column mean (0.0 for an all-NaN column)
                              The project treats the hot path as missing-free
                              (OBLIQUE_CONTEXT.md §1); imputing here keeps
                              projected values finite instead of relying on the
                              splitter's na_replacement.
  label     two classes    -> integer tokens 0/1, most frequent class = 0.
                              FormatLabelKey renders these as "0"/"1"; string
                              labels would drop the fast loader.

Ordinal-encoding a categorical is not a statistically meaningful encoding — it is
a *shape* choice: it preserves the suite's column count and row count so the
kernel sees a realistic workload. These CSVs are for runtime measurement, not for
publishing accuracy numbers on these suites.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd


def encode_features(X: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return an all-numeric float32 copy of `X` plus a summary of what changed."""
    out = {}
    stats = {
        "n_numeric": 0,
        "n_bool": 0,
        "n_datetime": 0,
        "n_categorical_encoded": 0,
        "n_all_nan": 0,
        "nan_cells_imputed": 0,
    }
    for name in X.columns:
        col = X[name]
        if pd.api.types.is_bool_dtype(col):
            vals = col.astype("float32")
            stats["n_bool"] += 1
        elif pd.api.types.is_numeric_dtype(col):
            vals = col.astype("float32")
            stats["n_numeric"] += 1
        elif pd.api.types.is_datetime64_any_dtype(col):
            vals = col.astype("int64").astype("float32") / 1e9
            stats["n_datetime"] += 1
        else:
            # Categorical / object / string -> ordinal codes over sorted uniques.
            s = col.astype("string")
            uniq = pd.Index(sorted(s.dropna().unique()))
            vals = s.map({v: float(i) for i, v in enumerate(uniq)}).astype("float32")
            stats["n_categorical_encoded"] += 1

        n_nan = int(np.isnan(vals.to_numpy(dtype="float32")).sum())
        if n_nan:
            stats["nan_cells_imputed"] += n_nan
            mean = float(np.nanmean(vals.to_numpy(dtype="float32")))
            if not np.isfinite(mean):
                mean = 0.0
                stats["n_all_nan"] += 1
            vals = vals.fillna(mean)
        out[name] = vals.astype("float32")

    return pd.DataFrame(out, index=X.index), stats


def encode_binary_label(y: pd.Series) -> tuple[pd.Series, dict]:
    """Map a two-class label to integer tokens 0/1 (most frequent class -> 0)."""
    s = y.astype("string") if not pd.api.types.is_numeric_dtype(y) else y
    counts = s.value_counts(dropna=True)
    if len(counts) != 2:
        raise ValueError(f"expected exactly 2 label classes, got {len(counts)}: "
                         f"{list(counts.index)[:5]}")
    order = list(counts.index)  # frequency-descending
    mapping = {order[0]: 0, order[1]: 1}
    tokens = s.map(mapping)
    if tokens.isna().any():
        raise ValueError("label contains missing values")
    return tokens.astype("int8"), {
        "classes": [str(order[0]), str(order[1])],
        "counts": [int(counts.iloc[0]), int(counts.iloc[1])],
        "positive_rate": float(counts.iloc[1] / counts.sum()),
    }


def write_dataset(out_dir: str, name: str, X: pd.DataFrame, y: pd.Series,
                  label_col: str = "label", extra_meta: dict | None = None) -> dict:
    """Encode, write `<out_dir>/<name>/train.csv`, and return its meta dict."""
    ds_dir = os.path.join(out_dir, name)
    os.makedirs(ds_dir, exist_ok=True)

    feats, fstats = encode_features(X)
    tokens, lstats = encode_binary_label(y)
    feats[label_col] = tokens.to_numpy()

    csv_path = os.path.join(ds_dir, "train.csv")
    feats.to_csv(csv_path, index=False, float_format="%.9g")

    meta = {
        "dataset": name,
        "rows": int(len(feats)),
        "features": int(feats.shape[1] - 1),
        "label_col": label_col,
        "label": lstats,
        "features_encoding": fstats,
        "csv_bytes": os.path.getsize(csv_path),
    }
    if extra_meta:
        meta.update(extra_meta)
    with open(os.path.join(ds_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta
