#!/usr/bin/env python3
"""Download the Criteo 1TB click logs and convert them to all-numeric
binary-classification CSVs for the oblique-RF harness.

Source: HuggingFace dataset `criteo/CriteoClickLogs` (parquet mirror of the
  Criteo Terabyte Click Logs, https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/,
  Creative Commons Attribution-NonCommercial-ShareAlike 4.0).
  * 6,029 snappy-parquet part files, ~276 GB, hive-partitioned by day:
    `data/day=2015-02-15/` ... `data/day=2015-03-10/` (24 days, ~196 M rows/day,
    ~0.8 M rows per part file).
  * Schema (40 columns, no id/timestamp):
      label                     int32   1 = click (~3.3 % of rows)
      integer_feature_1..13     int32   count-like features, nulls allowed
      categorical_feature_1..26 string  8-char lowercase-hex hashes, nulls allowed

This script downloads only the days it needs (`--train_days` days starting at
2015-02-15, plus `--test_parts` part files of the following day), then converts
them with a two-pass, multiprocess pipeline:
  pass 1  per part file: value counts of every categorical column, sums/null
          counts of every integer column  ->  global ordinal maps + train means
  pass 2  per part file: encode + impute, write a headerless CSV to
          `csv_parts/`, then concatenate the parts in file-name order.
The 8-char hex hashes are decoded to uint32 with a numpy nibble LUT (fixed-width
lowercase hex sorts identically as text and as uint32), which makes the unique /
sort / lookup work ~50x cheaper than string hashing; the resulting ordinal codes
are exactly the ones the repo rule prescribes (distinct values as strings,
sorted alphabetically -> 0, 1, 2, ...).

Target: `class` = `label` unchanged. The majority class is 0 (no click, ~96.7 %),
so the repo's "majority -> 0" convention needs no flip. Two classes only, so no
majority-vs-rest reduction applies.

Encoding / imputation (CLAUDE.md dataset policy, user directives 2026-09-21):
  * 13 integer features -> float32; null -> TRAIN column mean (test reuses it).
  * 26 categorical features -> ordinal codes over the union of the train and test
    values, distinct hashes sorted alphabetically -> 0..K-1; null -> the TRAIN
    mean OF THE CODES (i.e. imputation happens after encoding).
  * No rows and no columns are dropped. Rows with a null label would be dropped
    (there are none; the count is reported in the meta JSON).
  * Everything is stored as float32, so integer features above 2^24 and ordinal
    codes above 2^24 lose precision; the meta JSON records both maxima.

  benchmarks/data/download_criteo.py                       # 1 train day + 4 test parts
  benchmarks/data/download_criteo.py --train_days 2 --test_parts 8
  benchmarks/data/download_criteo.py --train_parts 50      # smoke test (first 50 parts)

Outputs (all in benchmarks/data/criteo/, gitignored):
  criteo_train.csv   class,int_1..int_13,cat_1..cat_26   (~196 M rows for 1 day)
  criteo_test.csv    same columns, the next day's first `--test_parts` parts
  criteo_meta.json   row/col counts, positive rates, target rule, per-column
                     cardinalities, imputation means and null counts, source files
  parquet/           the downloaded snappy-parquet part files (kept)
  csv_parts/         per-part temp CSVs (deleted after a successful concat unless
                     --keep_parts); makes a re-run resume instead of restart
"""
import argparse
import datetime
import glob
import json
import multiprocessing as mp
import os
import shutil
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "criteo")
PARQUET_DIR = os.path.join(OUT_DIR, "parquet")
PARTS_DIR = os.path.join(OUT_DIR, "csv_parts")
REPO_ID = "criteo/CriteoClickLogs"
FIRST_DAY = datetime.date(2015, 2, 15)
LAST_DAY = datetime.date(2015, 3, 10)

N_INT, N_CAT = 13, 26
INT_SRC = ["integer_feature_%d" % (i + 1) for i in range(N_INT)]
CAT_SRC = ["categorical_feature_%d" % (i + 1) for i in range(N_CAT)]
INT_OUT = ["int_%d" % (i + 1) for i in range(N_INT)]
CAT_OUT = ["cat_%d" % (i + 1) for i in range(N_CAT)]
OUT_COLS = ["class"] + INT_OUT + CAT_OUT

# 8 lowercase-hex chars -> uint32. Invalid bytes decode to 255 and trip the check.
_HEX_LUT = np.full(256, 255, np.uint8)
for _i, _ch in enumerate(b"0123456789abcdef"):
    _HEX_LUT[_ch] = _i


def day_dir(day):
    return os.path.join(PARQUET_DIR, "data", "day=%s" % day.isoformat())


def parts_of(day):
    return sorted(glob.glob(os.path.join(day_dir(day), "*.parquet")))


# ----------------------------------------------------------------------- download
def download(train_days, test_parts):
    """Fetch the train days in full and the first `test_parts` parts of the next day."""
    from huggingface_hub import hf_hub_download, snapshot_download

    days = [FIRST_DAY + datetime.timedelta(days=i) for i in range(train_days)]
    test_day = FIRST_DAY + datetime.timedelta(days=train_days)
    if test_day > LAST_DAY:
        sys.exit("--train_days %d leaves no day for the test split" % train_days)

    for day in days:
        print("[download] train day %s (full, idempotent)" % day, flush=True)
        snapshot_download(
            REPO_ID, repo_type="dataset",
            allow_patterns=["data/day=%s/*" % day.isoformat(), "README.md"],
            local_dir=PARQUET_DIR, max_workers=16)

    from huggingface_hub import HfApi
    remote = sorted(f for f in HfApi().list_repo_files(REPO_ID, repo_type="dataset")
                    if f.startswith("data/day=%s/" % test_day.isoformat()))
    if len(remote) < test_parts:
        sys.exit("test day %s has only %d parts" % (test_day, len(remote)))
    print("[download] test day %s: %d part(s)" % (test_day, test_parts), flush=True)
    for name in remote[:test_parts]:
        hf_hub_download(REPO_ID, name, repo_type="dataset", local_dir=PARQUET_DIR)

    train_files = [f for day in days for f in parts_of(day)]
    test_files = [os.path.join(PARQUET_DIR, n) for n in remote[:test_parts]]
    missing = [f for f in train_files + test_files if not os.path.exists(f)]
    if missing:
        sys.exit("missing after download: %s" % missing[:3])
    return days, test_day, train_files, test_files


# ------------------------------------------------------------------ hex fast path
def hex8_to_u32(arr):
    """Decode a string column of 8-char lowercase-hex values.

    Returns (valid_mask[bool, n], values_u32[n_valid]) with the values in row order.
    """
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.chunk(0) if arr.num_chunks == 1 else pa.concat_arrays(arr.chunks)
    n = len(arr)
    if n == 0:
        return np.zeros(0, bool), np.zeros(0, np.uint32)
    if arr.offset != 0:
        arr = pa.concat_arrays([arr])  # materialise a zero-offset copy
    bufs = arr.buffers()
    off = np.frombuffer(bufs[1], dtype=np.int32, count=n + 1)
    if bufs[0] is None:
        valid = np.ones(n, bool)
    else:
        valid = np.unpackbits(np.frombuffer(bufs[0], np.uint8), count=n,
                              bitorder="little").astype(bool)
    n_valid = int(valid.sum())
    widths = np.diff(off)
    if not np.array_equal(widths[valid], np.full(n_valid, 8, np.int32)):
        raise RuntimeError("categorical value is not 8 bytes wide")
    raw = np.frombuffer(bufs[2], np.uint8)[off[0]:off[n]].reshape(-1, 8)
    digits = _HEX_LUT[raw]
    if n_valid and digits.max() > 15:
        raise RuntimeError("categorical value is not lowercase hex")
    out = np.zeros(raw.shape[0], np.uint32)
    for k in range(8):
        out = (out << np.uint32(4)) | digits[:, k].astype(np.uint32)
    return valid, out


# -------------------------------------------------------------------------- pass 1
def scan_part(path):
    """Per-file statistics: categorical value counts, integer sums, label counts."""
    table = pq.read_table(path, columns=["label"] + INT_SRC + CAT_SRC,
                          use_threads=False)
    n = table.num_rows
    label = table["label"].combine_chunks()
    res = {
        "n_rows": n,
        "n_pos": int(pc.sum(pc.equal(label, 1)).as_py() or 0),
        "label_nulls": label.null_count,
        "int_sum": [], "int_nonnull": [], "int_max": [],
        "cat_vals": [], "cat_cnts": [], "cat_nulls": [],
    }
    for name in INT_SRC:
        col = table[name]
        s = pc.sum(col).as_py()
        res["int_sum"].append(0 if s is None else int(s))
        res["int_nonnull"].append(n - col.null_count)
        mx = pc.max(col).as_py()
        res["int_max"].append(0 if mx is None else int(mx))
    for name in CAT_SRC:
        valid, vals = hex8_to_u32(table[name])
        uniq, cnts = np.unique(vals, return_counts=True)
        res["cat_vals"].append(uniq)
        res["cat_cnts"].append(cnts.astype(np.int32))
        res["cat_nulls"].append(int(n - valid.sum()))
    return res


def _merge(vals_list, cnts_list):
    """Collapse a list of (values, counts) into one sorted unique pair."""
    vals = np.concatenate(vals_list)
    cnts = np.concatenate(cnts_list).astype(np.int64)
    uniq, inv = np.unique(vals, return_inverse=True)
    return uniq, np.bincount(inv, weights=cnts, minlength=len(uniq)).astype(np.int64)


MERGE_THRESHOLD = 12_000_000  # values buffered per column before a merge


def pass1(train_files, test_files, workers):
    """Global ordinal maps (train u test) + train means; counts are train-only."""
    files = [(f, True) for f in train_files] + [(f, False) for f in test_files]
    acc_v = [[np.zeros(0, np.uint32)] for _ in range(N_CAT)]
    acc_c = [[np.zeros(0, np.int64)] for _ in range(N_CAT)]
    buffered = [0] * N_CAT
    stats = {
        "train": {"rows": 0, "pos": 0, "label_nulls": 0,
                  "int_sum": np.zeros(N_INT, np.int64),
                  "int_nonnull": np.zeros(N_INT, np.int64),
                  "int_max": np.zeros(N_INT, np.int64),
                  "cat_nulls": np.zeros(N_CAT, np.int64)},
        "test": {"rows": 0, "pos": 0, "label_nulls": 0,
                 "int_sum": np.zeros(N_INT, np.int64),
                 "int_nonnull": np.zeros(N_INT, np.int64),
                 "int_max": np.zeros(N_INT, np.int64),
                 "cat_nulls": np.zeros(N_CAT, np.int64)},
    }
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, (res, (path, is_train)) in enumerate(
                zip(pool.imap(scan_part, [f for f, _ in files], chunksize=1), files)):
            s = stats["train" if is_train else "test"]
            s["rows"] += res["n_rows"]
            s["pos"] += res["n_pos"]
            s["label_nulls"] += res["label_nulls"]
            s["int_sum"] += np.asarray(res["int_sum"], np.int64)
            s["int_nonnull"] += np.asarray(res["int_nonnull"], np.int64)
            s["int_max"] = np.maximum(s["int_max"], np.asarray(res["int_max"], np.int64))
            s["cat_nulls"] += np.asarray(res["cat_nulls"], np.int64)
            for c in range(N_CAT):
                acc_v[c].append(res["cat_vals"][c])
                # test values join the map but never the mean.
                acc_c[c].append(res["cat_cnts"][c].astype(np.int64) if is_train
                                else np.zeros(len(res["cat_vals"][c]), np.int64))
                buffered[c] += len(res["cat_vals"][c])
                if buffered[c] > MERGE_THRESHOLD:
                    u, k = _merge(acc_v[c], acc_c[c])
                    acc_v[c], acc_c[c], buffered[c] = [u], [k], len(u)
            if (i + 1) % 10 == 0 or i + 1 == len(files):
                print("[pass1] %d/%d files  %.0fs  rows=%d"
                      % (i + 1, len(files), time.time() - t0,
                         stats["train"]["rows"] + stats["test"]["rows"]), flush=True)
    maps, code_means, card = [], [], []
    for c in range(N_CAT):
        u, k = _merge(acc_v[c], acc_c[c])
        maps.append(u)
        card.append(len(u))
        total = k.sum()
        code_means.append(float(np.dot(k.astype(np.float64), np.arange(len(u))) / total)
                          if total else 0.0)
    int_means = []
    for j in range(N_INT):
        nn = stats["train"]["int_nonnull"][j]
        int_means.append(float(stats["train"]["int_sum"][j] / nn) if nn else 0.0)
    return maps, np.asarray(code_means), np.asarray(int_means), card, stats


# -------------------------------------------------------------------------- pass 2
_G = {}


def _init_worker(maps, code_means, int_means):
    _G["maps"] = maps
    _G["code_means"] = code_means
    _G["int_means"] = int_means


def convert_part(job):
    """Encode one part file into a headerless CSV; returns (out_path, n_rows)."""
    path, out_path = job
    if os.path.exists(out_path):
        return out_path, -1  # already converted (resume)
    table = pq.read_table(path, columns=["label"] + INT_SRC + CAT_SRC,
                          use_threads=False)
    n = table.num_rows
    label = table["label"].combine_chunks()
    if label.null_count:  # the only rows we are allowed to drop
        keep = np.flatnonzero(label.is_valid().to_numpy(zero_copy_only=False))
        table = table.take(pa.array(keep))
        label = table["label"].combine_chunks()
        n = table.num_rows
    cols = {"class": pc.cast(label, pa.int8())}
    for j, (src, dst) in enumerate(zip(INT_SRC, INT_OUT)):
        # safe=False: a few integer features exceed 2^24 and cannot be represented
        # exactly in fp32; the harness is fp32 anyway, so rounding is accepted.
        col = pc.cast(table[src], pa.float32(), safe=False)
        cols[dst] = pc.fill_null(col, pa.scalar(np.float32(_G["int_means"][j]),
                                                pa.float32()))
    for c, (src, dst) in enumerate(zip(CAT_SRC, CAT_OUT)):
        valid, vals = hex8_to_u32(table[src])
        out = np.full(n, np.float32(_G["code_means"][c]), np.float32)
        if len(vals):
            out[valid] = np.searchsorted(_G["maps"][c], vals).astype(np.float32)
        cols[dst] = pa.array(out)
    out_table = pa.table({k: cols[k] for k in OUT_COLS})
    tmp = out_path + ".tmp"
    pacsv.write_csv(out_table, tmp, pacsv.WriteOptions(include_header=False))
    os.replace(tmp, out_path)
    return out_path, n


def pass2(files, split, maps, code_means, int_means, workers):
    os.makedirs(os.path.join(PARTS_DIR, split), exist_ok=True)
    jobs = [(f, os.path.join(PARTS_DIR, split, "%05d.csv" % i))
            for i, f in enumerate(files)]
    t0 = time.time()
    done = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(maps, code_means, int_means)) as pool:
        for i, (out_path, n) in enumerate(pool.imap(convert_part, jobs, chunksize=1)):
            done.append(out_path)
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                print("[pass2:%s] %d/%d parts  %.0fs" % (split, i + 1, len(jobs),
                                                         time.time() - t0), flush=True)
    return done


def prepare_parts_dir(fingerprint):
    """Reuse csv_parts/ only if it was produced by the same files and the same maps."""
    manifest = os.path.join(PARTS_DIR, "manifest.json")
    if os.path.exists(manifest):
        with open(manifest) as f:
            if json.load(f) == fingerprint:
                print("[pass2] resuming from existing csv_parts/", flush=True)
                return
        print("[pass2] csv_parts/ was built from different inputs -> rebuilding",
              flush=True)
        shutil.rmtree(PARTS_DIR, ignore_errors=True)
    os.makedirs(PARTS_DIR, exist_ok=True)
    with open(manifest, "w") as f:
        json.dump(fingerprint, f)


def concat(parts, out_csv):
    """Header + the part CSVs in order, streamed."""
    tmp = out_csv + ".tmp"
    t0 = time.time()
    with open(tmp, "wb") as dst:
        dst.write((",".join(OUT_COLS) + "\n").encode())
        for i, p in enumerate(parts):
            with open(p, "rb") as src:
                shutil.copyfileobj(src, dst, 1 << 24)
            if (i + 1) % 25 == 0 or i + 1 == len(parts):
                print("[concat] %d/%d  %.0fs  %.1f GB"
                      % (i + 1, len(parts), time.time() - t0,
                         dst.tell() / 1e9), flush=True)
    os.replace(tmp, out_csv)


def count_lines(path):
    n, buf_size = 0, 1 << 24
    with open(path, "rb") as f:
        while True:
            b = f.read(buf_size)
            if not b:
                return n
            n += b.count(b"\n")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_days", type=int, default=1,
                    help="days from 2015-02-15 used for the train CSV (default 1)")
    ap.add_argument("--test_parts", type=int, default=4,
                    help="part files of the day AFTER the train days (default 4)")
    ap.add_argument("--train_parts", type=int, default=0,
                    help="debug: keep only the first N train part files (0 = all)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--keep_parts", action="store_true",
                    help="keep csv_parts/ after the concat")
    ap.add_argument("--force", action="store_true", help="rebuild existing CSVs")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    train_csv = os.path.join(OUT_DIR, "criteo_train.csv")
    test_csv = os.path.join(OUT_DIR, "criteo_test.csv")
    meta_path = os.path.join(OUT_DIR, "criteo_meta.json")
    if not args.force and all(os.path.exists(p) for p in (train_csv, test_csv, meta_path)):
        print("already built: %s, %s (use --force to rebuild)" % (train_csv, test_csv))
        return

    t_start = time.time()
    days, test_day, train_files, test_files = download(args.train_days, args.test_parts)
    if args.train_parts:
        train_files = train_files[:args.train_parts]
    print("[files] train %d parts (%s), test %d parts (%s)"
          % (len(train_files), ", ".join(d.isoformat() for d in days),
             len(test_files), test_day), flush=True)

    maps, code_means, int_means, card, stats = pass1(train_files, test_files,
                                                     args.workers)
    print("[pass1] cardinalities: %s" % card, flush=True)
    print("[pass1] train rows=%d pos=%.4f%%  test rows=%d pos=%.4f%%"
          % (stats["train"]["rows"], 100.0 * stats["train"]["pos"] / max(stats["train"]["rows"], 1),
             stats["test"]["rows"], 100.0 * stats["test"]["pos"] / max(stats["test"]["rows"], 1)),
          flush=True)

    prepare_parts_dir({
        "train_files": [os.path.basename(f) for f in train_files],
        "test_files": [os.path.basename(f) for f in test_files],
        "cardinalities": [int(c) for c in card],
        "code_means": [round(float(m), 6) for m in code_means],
        "int_means": [round(float(m), 6) for m in int_means],
    })
    train_parts = pass2(train_files, "train", maps, code_means, int_means, args.workers)
    test_parts_out = pass2(test_files, "test", maps, code_means, int_means, args.workers)
    concat(train_parts, train_csv)
    concat(test_parts_out, test_csv)

    train_lines = count_lines(train_csv) - 1
    test_lines = count_lines(test_csv) - 1
    assert train_lines == stats["train"]["rows"] - stats["train"]["label_nulls"], \
        (train_lines, stats["train"]["rows"])
    assert test_lines == stats["test"]["rows"] - stats["test"]["label_nulls"], \
        (test_lines, stats["test"]["rows"])

    meta = {
        "name": "criteo",
        "source": {
            "repo": "https://huggingface.co/datasets/%s" % REPO_ID,
            "origin": "https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/",
            "licence": "CC BY-NC-SA 4.0",
            "train_days": [d.isoformat() for d in days],
            "test_day": test_day.isoformat(),
            "train_part_files": [os.path.basename(f) for f in train_files],
            "test_part_files": [os.path.basename(f) for f in test_files],
        },
        "target_rule": ("class = label (1 = click). Two classes; the majority class "
                        "is 0 (no click), so the majority->0 convention needs no flip."),
        "row_order": "part files sorted by file name, row order preserved inside a part",
        "train": {"rows": train_lines, "cols": len(OUT_COLS),
                  "positive_rate": stats["train"]["pos"] / max(stats["train"]["rows"], 1),
                  "bytes": os.path.getsize(train_csv)},
        "test": {"rows": test_lines, "cols": len(OUT_COLS),
                 "positive_rate": stats["test"]["pos"] / max(stats["test"]["rows"], 1),
                 "bytes": os.path.getsize(test_csv)},
        "dropped_columns": [],
        "dropped_rows": {"null_label_train": int(stats["train"]["label_nulls"]),
                         "null_label_test": int(stats["test"]["label_nulls"])},
        "dtype": "float32 for every feature; class is 0/1",
        "integer_features": {
            name: {"source": src,
                   "train_mean_imputed": float(int_means[j]),
                   "null_cells_train": int(stats["train"]["rows"]
                                           - stats["train"]["int_nonnull"][j]),
                   "null_cells_test": int(stats["test"]["rows"] - stats["test"]["int_nonnull"][j]),
                   "max_value_train": int(stats["train"]["int_max"][j]),
                   "fp32_exact": bool(stats["train"]["int_max"][j] <= 2 ** 24)}
            for j, (name, src) in enumerate(zip(INT_OUT, INT_SRC))},
        "categorical_features": {
            name: {"source": src, "cardinality": int(card[c]),
                   "code_rule": "distinct 8-hex values over train u test, sorted "
                                "alphabetically -> 0..K-1",
                   "train_mean_code_imputed": float(code_means[c]),
                   "null_cells_train": int(stats["train"]["cat_nulls"][c]),
                   "null_cells_test": int(stats["test"]["cat_nulls"][c]),
                   "codes_above_2pow24": max(int(card[c]) - 2 ** 24, 0)}
            for c, (name, src) in enumerate(zip(CAT_OUT, CAT_SRC))},
        "notes": [
            "Nulls are imputed with the TRAIN column mean; for categoricals the mean "
            "is taken over the ordinal codes, after encoding.",
            "Ordinal maps are built over train u test so one map serves both splits.",
            "All features are stored as float32: integers above 2^24 (16777216) are "
            "rounded to the nearest representable value. max ordinal code = %d; "
            "columns whose cardinality exceeds 2^24 (%s) therefore have colliding "
            "codes in their high range -- unavoidable for an fp32 feature store."
            % (max(card) - 1,
               ", ".join("%s:%d" % (CAT_OUT[c], card[c])
                         for c in range(N_CAT) if card[c] > 2 ** 24) or "none"),
        ],
        "columns": OUT_COLS,
        "build_seconds": round(time.time() - t_start, 1),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    if not args.keep_parts:
        shutil.rmtree(PARTS_DIR, ignore_errors=True)
    print("[done] %s (%d rows, %.1f GB), %s (%d rows, %.1f GB), %.0fs"
          % (train_csv, train_lines, os.path.getsize(train_csv) / 1e9,
             test_csv, test_lines, os.path.getsize(test_csv) / 1e9,
             time.time() - t_start), flush=True)


if __name__ == "__main__":
    main()
