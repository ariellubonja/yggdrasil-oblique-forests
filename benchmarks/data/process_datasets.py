#!/usr/bin/env python3
"""Processing rule for the TabArena / TabReD downloads (as specified by the user):

    1. Keep only *binary-classification* and *regression* datasets
       (TabArena problem_type in {binary, regression}; TabReD task.type in
       {binclass, regression}).  Multiclass datasets are dropped entirely.
    2. Drop every non-numeric column.
    3. Missing values, by --missing (default keep):
         keep       drop columns with >= --col-miss-threshold % NaN (default 50),
                    then keep every row; NaN cells stay in X.npy and are imputed
                    with the training-fold column mean by make_folds.py (YDF's own
                    oblique na_replacement rule).  Only rows with a missing
                    target are dropped.
         drop-rows  the original rule: drop every row with any missing item.
       Either way the dataset is dropped if the surviving numeric block has
       miss% >= --miss-threshold (default 10).

Details of how each step is applied
-----------------------------------
Step 2 (non-numeric columns):
  TabArena -- a column is numeric iff OpenML's is_categorical mask is False AND
    the pandas dtype kind is in "iuf".  Everything else (categoricals, object,
    bool) is dropped.
  TabReD  -- the preprocessed format already splits features into x_num / x_bin
    / x_cat, so only x_num is kept.  x_bin (0/1) and x_cat are dropped as
    non-numeric; pass --keep-binary to keep x_bin as numeric columns too.

Step 3 (miss%):
  miss% is measured on the surviving numeric feature block only (columns are
  dropped first, per the ordering above), as
      100 * (# NaN feature cells) / (rows * numeric columns).
  A dataset whose missingness lived entirely in the columns removed by step 2
  therefore reaches step 3 at 0% and loses no rows.
  Rows with a missing target are always dropped, independent of the threshold.

Outputs (benchmarks/data/processed/<benchmark>/<dataset>/):
  X.npy       float32, (rows, numeric columns); NaN kept under --missing keep
  y.npy       int8 0/1 for binary, float32 for regression
  info.json   provenance, feature names, class mapping, and the full before/after
              row & column accounting for this dataset
  splits/     TabReD only: the upstream splits with row indices remapped onto
              the surviving rows (dropped rows removed from each split)

Nothing is written for a dataset that step 1 or step 3 rejects; the manifest CSV
records it with a "kept" flag and a reason.

WARNING: under --missing drop-rows several datasets collapse (APSFailure 76k->756,
cooking-time 320k->2.9k, maps-routing and sberbank-housing -> 0 rows) because
nearly every row carries at least one NaN; that is why keep is the default.
The manifest reports rows_kept / pct_rows_kept and n_cols_dropped_missing.

Usage:
  .venv/bin/python benchmarks/data/process_datasets.py \
      [--out DIR] [--csv manifest.csv] [--keep-binary] [--min-rows-frac 0.0] \
      [--missing keep|drop-rows] [--col-miss-threshold 50.0]
      [--miss-threshold 10.0] [--only NAME ...] [--dry-run]
"""
import argparse
import csv
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SRC = os.path.join(REPO_ROOT, "benchmarks", "data")
DEFAULT_OUT = os.path.join(REPO_ROOT, "benchmarks", "data", "processed")

NUMERIC_KINDS = set("iuf")
# step 1: the only problem classes we keep, per benchmark's own vocabulary.
KEEP_TASKS = {"binary", "regression", "binclass"}
BINARY_TASKS = {"binary", "binclass"}

CHUNK_ROWS = 65536  # row block for the memmapped TabReD NaN scan


def encode_target(y, task):
    """binary -> int8 {0,1} (+ class mapping); regression -> float32."""
    if task in BINARY_TASKS:
        s = pd.Series(y)
        classes = sorted(pd.unique(s.dropna()), key=str)
        if len(classes) != 2:
            raise ValueError(f"binary target with {len(classes)} classes: {classes}")
        mapping = {c: i for i, c in enumerate(classes)}
        return s.map(mapping).to_numpy(dtype=np.int8), {str(k): v for k, v in mapping.items()}
    return np.asarray(y, dtype=np.float32), None


def write_dataset(odir, X, y, info, dry_run):
    if dry_run:
        return
    os.makedirs(odir, exist_ok=True)
    np.save(os.path.join(odir, "X.npy"), X)
    np.save(os.path.join(odir, "y.npy"), y)
    with open(os.path.join(odir, "info.json"), "w") as f:
        json.dump(info, f, indent=1)


def process_tabarena(args):
    rows = []
    TABARENA = os.path.join(args.src, "tabarena")
    if not os.path.isdir(TABARENA):
        return rows
    for name in sorted(os.listdir(TABARENA), key=str.lower):
        ddir = os.path.join(TABARENA, name)
        info_p = os.path.join(ddir, "info.json")
        if name.startswith("_") or not os.path.exists(info_p):
            continue
        if args.only and name not in args.only:
            continue
        info = json.load(open(info_p))
        task = info["problem_type"]
        rec = dict(benchmark="TabArena", dataset=name, task=task, kept=0, reason="")

        # --- step 1: problem class ---
        if task not in KEEP_TASKS:
            rec["reason"] = f"task={task} (not binary/regression)"
            rows.append(rec)
            print(f"[skip] TabArena/{name}: {rec['reason']}")
            continue

        df = pd.read_parquet(os.path.join(ddir, "data.parquet"))
        target = info["target_feature"]
        feats = [c for c in df.columns if c != target]
        cols = info["columns"]

        # --- step 2: drop non-numeric columns ---
        num_feats = [c for c in feats
                     if not cols.get(c, {}).get("is_categorical", False)
                     and df[c].dtype.kind in NUMERIC_KINDS]
        rec.update(n_rows_in=len(df), n_features_in=len(feats),
                   n_numeric_cols=len(num_feats),
                   n_cols_dropped=len(feats) - len(num_feats))
        if not num_feats:
            rec["reason"] = "no numeric columns"
            rows.append(rec)
            print(f"[skip] TabArena/{name}: {rec['reason']}")
            continue

        # --- step 3a (keep): drop columns that are mostly NaN ---
        n_hi = 0
        if args.missing == "keep":
            col_pct = 100.0 * df[num_feats].isna().mean()
            hi = [c for c in num_feats if col_pct[c] >= args.col_miss_threshold]
            n_hi = len(hi)
            num_feats = [c for c in num_feats if c not in hi]
        rec.update(n_cols_dropped_missing=n_hi, missing_policy=args.missing)
        if not num_feats:
            rec["reason"] = "no numeric columns after missing-column drop"
            rows.append(rec)
            print(f"[skip] TabArena/{name}: {rec['reason']}")
            continue
        sub = df[num_feats]
        na = sub.isna().to_numpy()
        cells = na.size
        miss = int(na.sum())
        pct = 100.0 * miss / cells if cells else 0.0
        rec.update(n_missing=miss, pct_missing=pct)

        # --- step 3b: miss% gate, then row policy ---
        if pct >= args.miss_threshold:
            rec["reason"] = f"miss%={pct:.2f} >= {args.miss_threshold}"
            rows.append(rec)
            print(f"[skip] TabArena/{name}: {rec['reason']}")
            continue
        keep = ~df[target].isna().to_numpy()
        if args.missing == "drop-rows":
            keep &= ~na.any(axis=1)
        n_keep = int(keep.sum())
        rec.update(rows_kept=n_keep, rows_dropped=len(df) - n_keep,
                   pct_rows_kept=100.0 * n_keep / len(df) if len(df) else 0.0)
        if n_keep < args.min_rows_frac * len(df):
            rec["reason"] = f"only {rec['pct_rows_kept']:.1f}% of rows survive"
            rows.append(rec)
            print(f"[skip] TabArena/{name}: {rec['reason']}")
            continue

        X = sub.to_numpy(dtype=np.float32)[keep]
        y, mapping = encode_target(df[target].to_numpy()[keep], task)
        out_info = dict(name=name, benchmark="TabArena", task=task,
                        source_openml_dataset_id=info.get("openml_dataset_id"),
                        source_openml_task_id=info.get("openml_task_id"),
                        target_feature=target, class_mapping=mapping,
                        feature_names=num_feats,
                        n_rows=int(X.shape[0]), n_features=int(X.shape[1]),
                        original_n_rows=len(df), original_n_features=len(feats),
                        pct_missing_numeric_block=pct, missing_policy=args.missing,
                        n_cols_dropped_missing=n_hi)
        write_dataset(os.path.join(args.out, "tabarena", name), X, y, out_info, args.dry_run)
        rec.update(kept=1)
        rows.append(rec)
        print(f"[ok]   TabArena/{name}: {X.shape[0]:,}x{X.shape[1]} "
              f"(was {len(df):,}x{len(feats)}, miss%={pct:.3f})")
    return rows


def _load(ddir, fn, mmap=True):
    p = os.path.join(ddir, fn)
    if not os.path.exists(p):
        return None
    return np.load(p, mmap_mode="r" if mmap else None)


def _nan_scan(blocks, n_rows, keepcols=None):
    """Chunked NaN scan over float memmaps -> (total NaN, per-row any-NaN mask,
    per-block per-column NaN counts). keepcols: per-block column masks."""
    miss = 0
    row_mask = np.zeros(n_rows, dtype=bool)
    col_counts = [np.zeros(_ncols(a), dtype=np.int64) for a in blocks]
    for start in range(0, n_rows, CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, n_rows)
        for bi, a in enumerate(blocks):
            if a is None or a.dtype.kind != "f":
                continue
            m = np.isnan(np.asarray(a[start:stop]))
            if m.ndim == 1:
                m = m[:, None]
            if keepcols is not None:
                m = m[:, keepcols[bi]]
            miss += int(m.sum())
            row_mask[start:stop] |= m.any(axis=1)
            if keepcols is None:
                col_counts[bi] += m.sum(axis=0)
    return miss, row_mask, col_counts


def _ncols(a):
    return 0 if a is None else (1 if a.ndim == 1 else a.shape[1])


def process_tabred(args):
    rows = []
    TABRED = os.path.join(args.src, "tabred")
    if not os.path.isdir(TABRED):
        return rows
    for name in sorted(os.listdir(TABRED), key=str.lower):
        ddir = os.path.join(TABRED, name)
        info_p = os.path.join(ddir, "info.json")
        if name.startswith("_") or not os.path.exists(info_p):
            continue
        if args.only and name not in args.only:
            continue
        info = json.load(open(info_p))
        task = info.get("task", {}).get("type", "?")
        rec = dict(benchmark="TabReD", dataset=name, task=task, kept=0, reason="")

        # --- step 1: problem class ---
        if task not in KEEP_TASKS:
            rec["reason"] = f"task={task} (not binary/regression)"
            rows.append(rec)
            print(f"[skip] TabReD/{name}: {rec['reason']}")
            continue

        x_num, x_bin, x_cat = (_load(ddir, "x_num.npy"), _load(ddir, "x_bin.npy"),
                               _load(ddir, "x_cat.npy"))
        y_raw = _load(ddir, "y.npy", mmap=False)
        n_rows = len(y_raw)

        ncols = _ncols

        # --- step 2: keep x_num (and x_bin under --keep-binary); drop x_cat ---
        blocks = [x_num] + ([x_bin] if args.keep_binary else [])
        blocks = [b for b in blocks if b is not None]
        n_in = ncols(x_num) + ncols(x_bin) + ncols(x_cat)
        n_num = sum(ncols(b) for b in blocks)
        rec.update(n_rows_in=n_rows, n_features_in=n_in, n_numeric_cols=n_num,
                   n_cols_dropped=n_in - n_num)
        if n_num == 0:
            rec["reason"] = "no numeric columns"
            rows.append(rec)
            print(f"[skip] TabReD/{name}: {rec['reason']}")
            continue

        # --- step 3a (keep): drop columns that are mostly NaN ---
        keepcols = [np.ones(ncols(b), dtype=bool) for b in blocks]
        n_hi = 0
        if args.missing == "keep":
            _, _, col_counts = _nan_scan(blocks, n_rows)
            keepcols = [100.0 * c / max(n_rows, 1) < args.col_miss_threshold
                        for c in col_counts]
            n_hi = sum(int((~k).sum()) for k in keepcols)
            n_num -= n_hi
        rec.update(n_cols_dropped_missing=n_hi, missing_policy=args.missing,
                   n_numeric_cols=n_num, n_cols_dropped=n_in - n_num)
        if n_num == 0:
            rec["reason"] = "no numeric columns after missing-column drop"
            rows.append(rec)
            print(f"[skip] TabReD/{name}: {rec['reason']}")
            continue
        miss, row_mask, _ = _nan_scan(blocks, n_rows, keepcols)
        cells = n_rows * n_num
        pct = 100.0 * miss / cells if cells else 0.0
        rec.update(n_missing=miss, pct_missing=pct)

        # --- step 3b: miss% gate, then row policy ---
        if pct >= args.miss_threshold:
            rec["reason"] = f"miss%={pct:.2f} >= {args.miss_threshold}"
            rows.append(rec)
            print(f"[skip] TabReD/{name}: {rec['reason']}")
            continue
        keep = np.ones(n_rows, dtype=bool)
        if args.missing == "drop-rows":
            keep &= ~row_mask
        if y_raw.dtype.kind == "f":
            keep &= ~np.isnan(y_raw)
        n_keep = int(keep.sum())
        rec.update(rows_kept=n_keep, rows_dropped=n_rows - n_keep,
                   pct_rows_kept=100.0 * n_keep / n_rows if n_rows else 0.0)
        if n_keep < args.min_rows_frac * n_rows:
            rec["reason"] = f"only {rec['pct_rows_kept']:.1f}% of rows survive"
            rows.append(rec)
            print(f"[skip] TabReD/{name}: {rec['reason']}")
            continue

        idx = np.flatnonzero(keep)
        X = np.empty((n_keep, n_num), dtype=np.float32)
        off = 0
        names = []
        for bi, b in enumerate(blocks):
            col = np.asarray(b[idx]).astype(np.float32, copy=False)
            col = col.reshape(n_keep, ncols(b))[:, keepcols[bi]]
            w = col.shape[1]
            X[:, off:off + w] = col
            off += w
            pfx = "num" if b is x_num else "bin"
            names += [f"{pfx}_{i}" for i in np.flatnonzero(keepcols[bi])]
        y, mapping = encode_target(y_raw[idx], task)
        out_info = dict(name=name, benchmark="TabReD", task=task,
                        score=info.get("task", {}).get("score"),
                        target_feature="y", class_mapping=mapping,
                        feature_names=names,
                        n_rows=int(X.shape[0]), n_features=int(X.shape[1]),
                        original_n_rows=n_rows, original_n_features=n_in,
                        pct_missing_numeric_block=pct, missing_policy=args.missing,
                        n_cols_dropped_missing=n_hi)
        odir = os.path.join(args.out, "tabred", name)
        write_dataset(odir, X, y, out_info, args.dry_run)

        # Row indices shifted; remap the upstream splits onto surviving rows.
        sdir = os.path.join(ddir, "splits")
        if os.path.isdir(sdir) and not args.dry_run:
            remap = np.full(n_rows, -1, dtype=np.int64)
            remap[idx] = np.arange(n_keep)
            osdir = os.path.join(odir, "splits")
            shutil.rmtree(osdir, ignore_errors=True)
            for split in sorted(os.listdir(sdir)):
                src = os.path.join(sdir, split)
                if not os.path.isdir(src):
                    continue
                dst = os.path.join(osdir, split)
                os.makedirs(dst, exist_ok=True)
                for part in sorted(os.listdir(src)):
                    if not part.endswith(".npy"):
                        continue
                    old = np.load(os.path.join(src, part))
                    np.save(os.path.join(dst, part), remap[old][remap[old] >= 0])

        rec.update(kept=1)
        rows.append(rec)
        print(f"[ok]   TabReD/{name}: {X.shape[0]:,}x{X.shape[1]} "
              f"(was {n_rows:,}x{n_in}, miss%={pct:.3f})")
    return rows


FIELDS = ["benchmark", "dataset", "task", "kept", "reason", "n_rows_in",
          "n_features_in", "n_numeric_cols", "n_cols_dropped", "n_missing",
          "pct_missing", "rows_kept", "rows_dropped", "pct_rows_kept",
          "missing_policy", "n_cols_dropped_missing"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=DEFAULT_SRC, help="dir holding tabarena/ and tabred/")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--missing", choices=["keep", "drop-rows"], default="keep",
                    help="keep NaN for fold-time mean imputation, or drop rows")
    ap.add_argument("--col-miss-threshold", type=float, default=50.0,
                    help="--missing keep: drop columns with >= this %% NaN first")
    ap.add_argument("--csv", default=os.path.join(
        REPO_ROOT, "benchmarks", "results", "dataset_properties", "processed_manifest.csv"))
    ap.add_argument("--miss-threshold", type=float, default=10.0,
                    help="drop the dataset if its numeric-block miss%% is >= this")
    ap.add_argument("--keep-binary", action="store_true",
                    help="TabReD: keep the x_bin 0/1 block as numeric columns")
    ap.add_argument("--min-rows-frac", type=float, default=0.0,
                    help="also reject a dataset if fewer than this fraction of "
                         "its rows survive the row drop (default 0 = off)")
    ap.add_argument("--only", nargs="*", default=None, help="dataset names to process")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    rows = process_tabarena(args) + process_tabred(args)
    if not rows:
        sys.exit("no datasets found")

    kept = [r for r in rows if r["kept"]]
    print(f"\nkept {len(kept)}/{len(rows)} datasets "
          f"({sum(1 for r in kept if r['benchmark']=='TabArena')} TabArena, "
          f"{sum(1 for r in kept if r['benchmark']=='TabReD')} TabReD)")
    thin = [r for r in kept if r.get("pct_rows_kept", 100.0) < 50.0]
    if thin:
        print("WARNING: lost >50% of rows to the row drop:")
        for r in sorted(thin, key=lambda r: r["pct_rows_kept"]):
            print(f"  {r['benchmark']}/{r['dataset']}: {r['rows_kept']:,} of "
                  f"{r['n_rows_in']:,} rows ({r['pct_rows_kept']:.2f}%)")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
