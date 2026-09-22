#!/usr/bin/env python3
"""Download the Numerai "Atlas" v5.0 tournament data and convert it to all-numeric
binary-classification CSVs for the oblique-RF harness.

Source: https://numer.ai/data  (Numerai public tournament data, free to download; no API
key needed).  Files are fetched with `numerapi` (installed in the repo venv):

    NumerAPI().download_dataset("v5.0/train.parquet",      benchmarks/data/numerai/train.parquet)
    NumerAPI().download_dataset("v5.0/validation.parquet", benchmarks/data/numerai/validation.parquet)
    NumerAPI().download_dataset("v5.0/features.json",      benchmarks/data/numerai/features.json)

Raw data (as of 2026-09-21):
  * train.parquet       2,746,268 rows, eras 0001-0574 (one era = one week, rows stored in
                        era order), 2,416 columns.
  * validation.parquet  4,148,785 rows, eras 0575-1237 -> the LATER period, i.e. the test split.
  * columns: `id` (string, row hash), `era` (string), `data_type` (string), 2,376 int8 feature
    columns with values 0..4 and NULLs (~0.4 % of cells), and 37 float32 target columns
    (`target` = the main one, values in {0, .25, .5, .75, 1}; 36 auxiliary targets).
  * features.json: `feature_sets["all"]` = the 2,376 feature names (used here, in that order),
    `targets` = the 37 target column names.

Conversion (repo dataset policy, CLAUDE.md 2026-09-21):
  * Label: `target` has 5 distinct values -> MAJORITY-vs-rest.  The most frequent value in
    TRAIN is 0.5 (1,373,107 / 2,746,268 = 50.0 %), so class = 1 iff target == 0.5, else 0.
    The same positive value is applied to the test split.
  * Rows with a NULL `target` are dropped (train: 0; validation: 34,713 rows of the most
    recent eras, whose targets are not yet resolved).  No other row is dropped.
  * Features: exactly feature_sets["all"], in that order, cast to float32.  NULL feature cells
    are imputed with the TRAIN column mean (test reuses the train means); a hypothetical
    all-NULL column would get 0.0.
  * Dropped columns: `id`, `era`, `data_type` (strings, not features) and all 37 target
    columns (`target` becomes `class`; the 36 auxiliary targets are alternative labels for the
    same rows and would leak).  The full list is in the meta JSON.
  * Column names are the feature names from features.json; one of them contains an apostrophe
    (feature_complemental_ok'd_prelateship) and is sanitised to an identifier (-> ..._ok_d_...).
    The rename is recorded in the meta JSON.

The data is wide (2.75M x 2,376 = 6.5e9 train cells), so both passes stream the parquet in
row-group-aligned batches with pyarrow (first pass: per-column sum/null-count in float64 for
the train means; second pass: cast + fill_null + CSV format).  CSV formatting runs in a small
thread pool (pyarrow releases the GIL) and the formatted chunks are written in order to a
single output file; peak RSS stays around 10-15 GB.

    benchmarks/data/download_numerai.py                 # full run, ~20 min, 33 GB of CSV
    benchmarks/data/download_numerai.py --force         # redo the CSVs even if they exist
    benchmarks/data/download_numerai.py --max_rows 200000 --out_suffix _smoke   # smoke test

Outputs (all in benchmarks/data/numerai/, gitignored):
  numerai_train.csv   2,746,268 x 2,377   class + 2,376 features, chronological (era) order
  numerai_test.csv    4,114,072 x 2,377   same columns, later eras
  numerai_meta.json   row/col counts, positive rates, target rule + value counts, dropped
                      columns, per-column imputation means and null counts, source files
  train.parquet / validation.parquet / features.json   the raw downloads (kept)
"""
import argparse
import json
import os
import re
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "numerai")
DATASETS = {
    "train": "v5.0/train.parquet",
    "validation": "v5.0/validation.parquet",
    "features": "v5.0/features.json",
}
LOCAL = {
    "train": os.path.join(OUT_DIR, "train.parquet"),
    "validation": os.path.join(OUT_DIR, "validation.parquet"),
    "features": os.path.join(OUT_DIR, "features.json"),
}


def log(msg):
    print("[%7.1fs] %s" % (time.time() - T0, msg), flush=True)


# ------------------------------------------------------------------ download (idempotent)
def download():
    missing = [k for k, p in LOCAL.items() if not (os.path.exists(p) and os.path.getsize(p) > 0)]
    if not missing:
        log("raw files already present, skipping download")
        return
    from numerapi import NumerAPI  # imported lazily: not needed when the files are there

    api = NumerAPI()
    for key in missing:
        log("downloading %s -> %s" % (DATASETS[key], LOCAL[key]))
        api.download_dataset(DATASETS[key], LOCAL[key])


def sanitize(name):
    """Feature name -> simple identifier (letters, digits, underscore)."""
    return re.sub(r"[^0-9A-Za-z_]", "_", name)


# ------------------------------------------------------------------ pass 1: train means
def train_means(features, batch_size, max_rows):
    """Per-column (sum, non-null count) over the kept train rows -> float64 means."""
    pf = pq.ParquetFile(LOCAL["train"])
    nfeat = len(features)
    sums = np.zeros(nfeat, dtype=np.float64)
    counts = np.zeros(nfeat, dtype=np.int64)
    nulls = np.zeros(nfeat, dtype=np.int64)
    seen = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=features):
        for i, col in enumerate(batch.columns):
            s = pc.sum(col).as_py()
            sums[i] += 0.0 if s is None else float(s)
            nulls[i] += col.null_count
            counts[i] += len(col) - col.null_count
        seen += batch.num_rows
        log("  means pass: %d rows" % seen)
        if max_rows and seen >= max_rows:
            break
    means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    return means, nulls, seen


# ------------------------------------------------------------------ pass 2: encode + write
def format_chunk(arrays, class_arr, means_f32):
    """int8 feature arrays (+ class) -> CSV bytes, no header. Runs in a worker thread."""
    cols = [class_arr]
    for i, col in enumerate(arrays):
        col = pc.cast(col, pa.float32())
        if col.null_count:
            col = pc.fill_null(col, pa.scalar(float(means_f32[i]), pa.float32()))
        cols.append(col)
    table = pa.Table.from_arrays(cols, names=["c%d" % i for i in range(len(cols))])
    sink = pa.BufferOutputStream()
    pacsv.write_csv(table, sink, pacsv.WriteOptions(include_header=False))
    return sink.getvalue()


def convert(split, parquet_path, features, out_names, means, positive_value, out_csv,
            batch_size, threads, max_rows):
    """Stream one parquet file into one CSV; returns (rows, positives, dropped_null_label)."""
    means_f32 = means.astype(np.float32)
    pf = pq.ParquetFile(parquet_path)
    rows = positives = dropped = 0
    pending = deque()
    header = ("class," + ",".join(out_names) + "\n").encode()
    with open(out_csv, "wb") as fh, ThreadPoolExecutor(max_workers=threads) as pool:
        fh.write(header)

        def drain(limit):
            nonlocal rows
            while len(pending) > limit:
                fh.write(pending.popleft().result())

        for batch in pf.iter_batches(batch_size=batch_size, columns=["target"] + features):
            target = batch.column(0)
            arrays = list(batch.columns[1:])
            null_label = target.null_count
            if null_label:
                mask = pc.is_valid(target)
                target = pc.filter(target, mask)
                arrays = [pc.filter(a, mask) for a in arrays]
                dropped += null_label
            n = len(target)
            if n == 0:
                continue
            cls = pc.cast(pc.equal(target, pa.scalar(positive_value, pa.float32())), pa.int8())
            positives += pc.sum(cls).as_py() or 0
            rows += n
            drain(threads)  # backpressure: at most `threads` chunks in flight
            pending.append(pool.submit(format_chunk, arrays, cls, means_f32))
            log("  %s: %d rows encoded" % (split, rows))
            if max_rows and rows >= max_rows:
                break
        drain(0)
    return rows, positives, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch_size", type=int, default=100000)
    ap.add_argument("--threads", type=int, default=6, help="CSV formatting threads")
    ap.add_argument("--max_rows", type=int, default=0, help="debug: stop after N rows per split")
    ap.add_argument("--out_suffix", default="", help="debug: suffix for the output file names")
    ap.add_argument("--force", action="store_true", help="rewrite CSVs even if they exist")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    download()

    spec = json.load(open(LOCAL["features"]))
    features = list(spec["feature_sets"]["all"])
    targets = list(spec["targets"])
    out_names = [sanitize(f) for f in features]
    renamed = {f: o for f, o in zip(features, out_names) if f != o}
    assert len(set(out_names)) == len(out_names), "feature name collision after sanitising"

    train_csv = os.path.join(OUT_DIR, "numerai_train%s.csv" % args.out_suffix)
    test_csv = os.path.join(OUT_DIR, "numerai_test%s.csv" % args.out_suffix)
    meta_path = os.path.join(OUT_DIR, "numerai_meta%s.json" % args.out_suffix)
    if not args.force and all(os.path.exists(p) and os.path.getsize(p) > 0
                              for p in (train_csv, test_csv, meta_path)):
        log("CSVs already present, nothing to do (use --force to rewrite)")
        return

    # Label rule: majority value of the TRAIN `target` column vs the rest.
    log("reading train targets")
    tcol = pq.read_table(LOCAL["train"], columns=["target"]).column("target")
    vc = pc.value_counts(tcol).to_pylist()
    counts = {("null" if d["values"] is None else float(d["values"])): d["counts"] for d in vc}
    positive_value = max((v for v in counts if v != "null"), key=lambda v: (counts[v], -v))
    log("train target counts %s -> positive value %s" % (counts, positive_value))

    log("pass 1: train column means")
    means, nulls, train_seen = train_means(features, args.batch_size, args.max_rows)

    log("pass 2: writing %s" % train_csv)
    tr_rows, tr_pos, tr_drop = convert("train", LOCAL["train"], features, out_names, means,
                                       positive_value, train_csv, args.batch_size,
                                       args.threads, args.max_rows)
    log("pass 2: writing %s" % test_csv)
    te_rows, te_pos, te_drop = convert("test", LOCAL["validation"], features, out_names, means,
                                       positive_value, test_csv, args.batch_size,
                                       args.threads, args.max_rows)

    order = np.argsort(-nulls)
    meta = {
        "name": "numerai",
        "source": "https://numer.ai/data  (Numerai tournament data v5.0 'Atlas')",
        "source_files": {k: os.path.basename(v) for k, v in LOCAL.items()},
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train": {
            "csv": os.path.basename(train_csv), "rows": tr_rows, "cols": len(out_names) + 1,
            "positives": tr_pos, "positive_rate": tr_pos / max(tr_rows, 1),
            "bytes": os.path.getsize(train_csv), "eras": "0001-0574 (chronological order)",
            "dropped_null_label_rows": tr_drop,
        },
        "test": {
            "csv": os.path.basename(test_csv), "rows": te_rows, "cols": len(out_names) + 1,
            "positives": te_pos, "positive_rate": te_pos / max(te_rows, 1),
            "bytes": os.path.getsize(test_csv), "eras": "0575-1237 (chronological order)",
            "dropped_null_label_rows": te_drop,
        },
        "target_rule": ("multi-class majority-vs-rest: class = 1 iff `target` == %s, the most "
                        "frequent of the 5 train target values" % positive_value),
        "target_value_counts_train": {str(k): v for k, v in counts.items()},
        "dropped_columns": {
            "id": "string row hash, not a feature",
            "era": "string week id, not a feature (defines the train/test time split)",
            "data_type": "string split tag ('train'/'validation'), not a feature",
            **{t: ("main target -> `class`" if t == "target"
                   else "auxiliary target (alternative label for the same row; leaks)")
               for t in targets},
        },
        "encoding": {
            "features": "feature_sets['all'] (2,376 int8 columns, values 0-4) cast to float32",
            "categoricals": "none (no string feature columns)",
            "nan_imputation": ("NULL feature cells -> TRAIN column mean (test reuses the train "
                               "means); all-NULL column -> 0.0"),
            "renamed_columns": renamed,
        },
        "train_null_cells_total": int(nulls.sum()),
        "train_null_cell_fraction": float(nulls.sum()) / max(train_seen * len(features), 1),
        "top_null_columns": [{"column": out_names[i], "null_cells": int(nulls[i]),
                              "train_mean": float(means[i])} for i in order[:10]],
        "imputation_means": {out_names[i]: float(means[i]) for i in range(len(out_names))},
        "train_null_counts": {out_names[i]: int(nulls[i]) for i in range(len(out_names))
                              if nulls[i]},
    }
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=1)
    log("wrote %s" % meta_path)
    log("train %d x %d (%.4f pos), test %d x %d (%.4f pos)" %
        (tr_rows, len(out_names) + 1, meta["train"]["positive_rate"],
         te_rows, len(out_names) + 1, meta["test"]["positive_rate"]))


if __name__ == "__main__":
    T0 = time.time()
    main()
