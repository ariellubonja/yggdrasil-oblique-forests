"""Shared helpers for the large-dataset download/massage scripts (2026-09-24)."""
import json, os, time
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_csv_unquoted(table: pa.Table, path: str):
    """Arrow always quotes header names; the harness wants a bare header."""
    with pa.OSFile(path, "wb") as sink:
        sink.write((",".join(table.column_names) + "\n").encode())
        pacsv.write_csv(table, sink, write_options=pacsv.WriteOptions(include_header=False, quoting_style="none"))


def ordinal_encode(col: pa.ChunkedArray):
    """Distinct string values sorted alphabetically -> 0,1,2,...; nulls stay null. Returns (float32 array, map)."""
    vals = sorted(v for v in pc.unique(col).to_pylist() if v is not None)
    idx = {v: float(i) for i, v in enumerate(vals)}
    out = pc.index_in(col, value_set=pa.array(vals, type=col.type))  # null for null
    return pc.cast(out, pa.float32()), idx


def impute_and_binarize(X: pa.Table, target: np.ndarray, means: dict, threshold: float, label_name="class"):
    """Mean-impute every feature column with the given (train) means, prepend class = target > threshold."""
    cols, names = [], []
    cls = pa.array((target > threshold).astype(np.int8))
    cols.append(cls); names.append(label_name)
    for n in X.column_names:
        c = pc.cast(X.column(n), pa.float32(), safe=False)
        if c.null_count > 0:
            c = pc.fill_null(c, pa.scalar(float(means[n]), pa.float32()))
        cols.append(c); names.append(n)
    return pa.table(cols, names=names)


def column_means(X: pa.Table):
    return {n: float(pc.mean(pc.cast(X.column(n), pa.float64())).as_py() or 0.0) for n in X.column_names}


def nan_counts(X: pa.Table):
    return {n: int(X.column(n).null_count) for n in X.column_names if X.column(n).null_count}


def write_meta(path, meta):
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=1, default=str)
