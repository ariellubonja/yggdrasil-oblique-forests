#!/usr/bin/env python3
"""Download the NYC TLC *yellow taxi* trip records and convert them to all-numeric
binary-classification CSVs for the oblique-RF harness.

Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
  (NYC Taxi & Limousine Commission trip record data, public domain / NYC Open Data
  terms of use).  One parquet file per month, served from
  https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet
  (~50 MB, ~3.3 M rows each).  Default span 2022-01 .. 2024-12 = 36 files,
  119,136,044 rows.

Raw schema (19 columns, identical in every month of the default span; the only
inter-month difference is the spelling `airport_fee` (2022-01 .. 2023-01) vs
`Airport_fee` (2023-02 .. 2024-12) and the integer widths / `string` vs
`large_string`, all normalised here):
    VendorID, tpep_pickup_datetime, tpep_dropoff_datetime, passenger_count,
    trip_distance, RatecodeID, store_and_fwd_flag, PULocationID, DOLocationID,
    payment_type, fare_amount, extra, mta_tax, tip_amount, tolls_amount,
    improvement_surcharge, total_amount, congestion_surcharge, airport_fee
No month in the default span carries extra columns (no `cbd_congestion_fee`), so
nothing had to be dropped for being absent from some months.  If a future month does,
this script keeps only the columns present in EVERY selected month and lists the
dropped ones in the meta JSON.

Target (the user's spec is the regression target `trip_duration_s`, binarised):
    trip_duration_s = tpep_dropoff_datetime - tpep_pickup_datetime   (integer seconds)
    class = 1 iff trip_duration_s > median(trip_duration_s over the TRAIN split)
The train median is computed once and the *same* threshold is applied to the test
split; it is recorded in the meta JSON.  All rows are kept, including garbage
durations (negative, zero, > 1 day) — the median split is robust to them.  The only
rows dropped are those whose pickup or dropoff timestamp is null (no label); they are
counted in the meta JSON.

Features (18): `tpep_dropoff_datetime` is dropped (it defines the target).
    vendor_id, pickup_epoch_s, passenger_count, trip_distance, ratecode_id,
    store_and_fwd_flag, pu_location_id, do_location_id, payment_type, fare_amount,
    extra, mta_tax, tip_amount, tolls_amount, improvement_surcharge, total_amount,
    congestion_surcharge, airport_fee
  * `pickup_epoch_s` = tpep_pickup_datetime as Unix epoch seconds.  Stored fp32 like
    every other feature, so the timestamp is rounded to a 128 s grid (fp32 has a
    24-bit mantissa, epoch ~1.7e9) — a deliberate, documented precision loss.
  * `store_and_fwd_flag` is the only string column: ordinal codes over the distinct
    values sorted alphabetically (N -> 0, Y -> 1); nulls become NaN and are then
    mean-imputed like any other NaN.
  * every other column -> float32; nulls -> the TRAIN column mean (the test split
    reuses the train mean; an all-NaN column would get 0.0).

Row order is chronological by FILE: month by month, and within a month exactly the
parquet row order (never re-sorted), so `head -n K` of the train CSV is an
earliest-K prefix at month granularity.

  benchmarks/data/download_nyc_taxi.py                      # full default run
  benchmarks/data/download_nyc_taxi.py --train_months 2022-01:2022-03 \
      --test_months 2022-04:2022-04 --out_prefix nyc_taxi_smoke   # smoke test

Outputs (all in benchmarks/data/nyc_taxi/, gitignored):
  nyc_taxi_train.csv   class,<18 features>   2022-01 .. 2024-06  (~100 M rows)
  nyc_taxi_test.csv    same columns          2024-07 .. 2024-12  (~21 M rows)
  nyc_taxi_meta.json   row/col counts, positive rates, target rule, encodings,
                       imputation means, NaN counts per column, source files
  parquet/             the downloaded monthly files (never deleted)
  cache/               per-month float32 feature/duration .npy intermediates; removed
                       on success unless --keep_cache (rebuilding them takes ~10 s)

Everything is idempotent: finished downloads, finished per-month caches and finished
CSV parts are skipped on a re-run (parts are invalidated if the train median or the
imputation means change).
"""
import argparse
import json
import os
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "nyc_taxi")
PARQUET_DIR = os.path.join(OUT_DIR, "parquet")
CACHE_DIR = os.path.join(OUT_DIR, "cache")
PARTS_DIR = os.path.join(OUT_DIR, "csv_parts")
URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_%s.parquet"

# raw parquet name (case-insensitive match) -> output feature name, in output order.
# tpep_dropoff_datetime is absent on purpose: it defines the target.
FEATURES = [
    ("VendorID", "vendor_id"),
    ("tpep_pickup_datetime", "pickup_epoch_s"),
    ("passenger_count", "passenger_count"),
    ("trip_distance", "trip_distance"),
    ("RatecodeID", "ratecode_id"),
    ("store_and_fwd_flag", "store_and_fwd_flag"),
    ("PULocationID", "pu_location_id"),
    ("DOLocationID", "do_location_id"),
    ("payment_type", "payment_type"),
    ("fare_amount", "fare_amount"),
    ("extra", "extra"),
    ("mta_tax", "mta_tax"),
    ("tip_amount", "tip_amount"),
    ("tolls_amount", "tolls_amount"),
    ("improvement_surcharge", "improvement_surcharge"),
    ("total_amount", "total_amount"),
    ("congestion_surcharge", "congestion_surcharge"),
    ("airport_fee", "airport_fee"),
]
PICKUP, DROPOFF = "tpep_pickup_datetime", "tpep_dropoff_datetime"
FLAG_COL = "store_and_fwd_flag"
FLAG_MAP = {"N": 0.0, "Y": 1.0}          # distinct values sorted alphabetically


# ----------------------------------------------------------------------------- months
def month_range(spec):
    """'2022-01:2024-06' (or a single '2022-01') -> ['2022-01', ..., '2024-06']."""
    lo, _, hi = spec.partition(":")
    hi = hi or lo
    (y0, m0), (y1, m1) = (tuple(map(int, s.split("-"))) for s in (lo, hi))
    if (y1, m1) < (y0, m0):
        raise SystemExit("bad month range %r" % spec)
    out, y, m = [], y0, m0
    while (y, m) <= (y1, m1):
        out.append("%04d-%02d" % (y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


# ----------------------------------------------------------------------------- download
def download_month(month, retries=5):
    path = os.path.join(PARQUET_DIR, "yellow_tripdata_%s.parquet" % month)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path, 0
    part = path + ".part"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(URL % month, timeout=120) as r, open(part, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            if os.path.getsize(part) == 0:
                raise IOError("empty download for %s" % month)
            pq.ParquetFile(part).metadata            # cheap integrity check
            os.replace(part, path)
            return path, os.path.getsize(path)
        except Exception:                            # noqa: BLE001
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError


def download_all(months, jobs):
    os.makedirs(PARQUET_DIR, exist_ok=True)
    t0, new_bytes = time.time(), 0
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for fut in as_completed([ex.submit(download_month, m) for m in months]):
            new_bytes += fut.result()[1]
    print("downloads ok: %d months, %.2f GB newly fetched, %.0f s"
          % (len(months), new_bytes / 1e9, time.time() - t0), flush=True)


# ----------------------------------------------------------------------------- stage 1
def _norm(name):
    return name.lower()


def common_columns(months):
    """Column names (lower-cased) present in EVERY month, plus what was dropped."""
    per_month = []
    for m in months:
        path = os.path.join(PARQUET_DIR, "yellow_tripdata_%s.parquet" % m)
        per_month.append(set(_norm(n) for n in pq.ParquetFile(path).schema_arrow.names))
    common = set.intersection(*per_month)
    union = set.union(*per_month)
    return common, sorted(union - common)


def prepare_month(args):
    """Parquet -> per-month float32 feature matrix (NaNs kept) + int64 durations.

    Returns stats only; the arrays go to CACHE_DIR/<month>_{x,d}.npy.
    """
    month, keep = args
    x_path = os.path.join(CACHE_DIR, "%s_x.npy" % month)
    d_path = os.path.join(CACHE_DIR, "%s_d.npy" % month)
    s_path = os.path.join(CACHE_DIR, "%s_stats.json" % month)
    if os.path.exists(s_path) and os.path.exists(x_path) and os.path.exists(d_path):
        with open(s_path) as f:
            return json.load(f)

    path = os.path.join(PARQUET_DIR, "yellow_tripdata_%s.parquet" % month)
    pf = pq.ParquetFile(path)
    raw_by_norm = {_norm(n): n for n in pf.schema_arrow.names}
    cols = [raw_by_norm[_norm(src)] for src, _ in FEATURES if _norm(src) in keep]
    table = pf.read(columns=cols + [raw_by_norm[DROPOFF]])
    n_raw = table.num_rows

    pick = table.column(raw_by_norm[PICKUP]).combine_chunks()
    drop = table.column(raw_by_norm[DROPOFF]).combine_chunks()
    valid = np.asarray(pick.is_valid().to_numpy(zero_copy_only=False), dtype=bool) \
        & np.asarray(drop.is_valid().to_numpy(zero_copy_only=False), dtype=bool)
    n_drop = int(n_raw - valid.sum())
    pick_us = pick.fill_null(0).cast(pa.int64()).to_numpy(zero_copy_only=False).astype(np.int64)
    drop_us = drop.fill_null(0).cast(pa.int64()).to_numpy(zero_copy_only=False).astype(np.int64)

    # floor-divide microseconds -> integer seconds (floor also for negative durations)
    dur = np.floor_divide(drop_us - pick_us, 1_000_000)
    if n_drop:
        dur = dur[valid]
    np.save(d_path, dur)

    out_names = [dst for src, dst in FEATURES if _norm(src) in keep]
    n = dur.size
    x = np.empty((n, len(out_names)), dtype=np.float32)
    nan_counts, sums, flag_values = {}, {}, []
    for j, (src, dst) in enumerate([(s, d) for s, d in FEATURES if _norm(s) in keep]):
        arr = table.column(raw_by_norm[_norm(src)]).combine_chunks()
        if dst == "pickup_epoch_s":
            col = arr.fill_null(0).cast(pa.int64()).to_numpy(
                zero_copy_only=False).astype(np.float64) / 1e6
        elif dst == FLAG_COL:
            vals = arr.cast(pa.string()).to_pylist()
            flag_values = sorted({v for v in vals if v is not None})
            col = np.array([FLAG_MAP.get(v, np.nan) if v is not None else np.nan
                            for v in vals], dtype=np.float64)
        else:
            col = arr.cast(pa.float64()).to_numpy(zero_copy_only=False)
        col = np.asarray(col, dtype=np.float64)
        if n_drop:
            col = col[valid]
        x[:, j] = col.astype(np.float32)
        bad = ~np.isfinite(x[:, j])
        nan_counts[dst] = int(bad.sum())
        sums[dst] = float(x[:, j][~bad].astype(np.float64).sum())
    np.save(x_path, x)

    stats = {"month": month, "n_raw": n_raw, "n_rows": int(n), "n_dropped_null_ts": n_drop,
             "nan_counts": nan_counts, "sums": sums, "columns": out_names,
             "flag_values": flag_values}
    with open(s_path, "w") as f:
        json.dump(stats, f)
    print("  prepared %s: %d rows (%d dropped, %d NaN cells)"
          % (month, n, n_drop, sum(nan_counts.values())), flush=True)
    return stats


# ----------------------------------------------------------------------------- stage 2/3
def write_part(args):
    """Impute NaNs with the train means, derive class from the median, write one part."""
    month, means, median, columns = args
    part = os.path.join(PARTS_DIR, "%s.csv" % month)
    info = os.path.join(PARTS_DIR, "%s_part.json" % month)
    if os.path.exists(part) and os.path.getsize(part) > 0 and os.path.exists(info):
        with open(info) as f:
            return month, json.load(f)
    x = np.load(os.path.join(CACHE_DIR, "%s_x.npy" % month))
    dur = np.load(os.path.join(CACHE_DIR, "%s_d.npy" % month))
    cls = (dur > median).astype(np.int8)
    cols = {"class": pa.array(cls)}
    for j, name in enumerate(columns):
        v = x[:, j]
        bad = ~np.isfinite(v)
        if bad.any():
            v = np.where(bad, np.float32(means[name]), v)
        if not np.isfinite(v).all():
            raise ValueError("%s: non-finite values left in %s" % (month, name))
        cols[name] = pa.array(v)
    table = pa.table(cols)
    tmp = part + ".part"
    opts = pcsv.WriteOptions(include_header=False, quoting_style="none", batch_size=1 << 16)
    with open(tmp, "wb") as f:
        pcsv.write_csv(table, f, opts)
    os.replace(tmp, part)
    out = {"rows": int(cls.size), "positives": int(cls.sum()), "bytes": os.path.getsize(part)}
    with open(info, "w") as f:
        json.dump(out, f)
    return month, out


def concat_parts(months, columns, out_path):
    header = ("class," + ",".join(columns) + "\n").encode()
    tmp = out_path + ".part"
    with open(tmp, "wb") as fo:
        fo.write(header)
        for m in months:
            with open(os.path.join(PARTS_DIR, "%s.csv" % m), "rb") as fi:
                shutil.copyfileobj(fi, fo, 1 << 22)
    os.replace(tmp, out_path)
    return os.path.getsize(out_path)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_months", default="2022-01:2024-06")
    ap.add_argument("--test_months", default="2024-07:2024-12")
    ap.add_argument("--out_prefix", default="nyc_taxi")
    ap.add_argument("--jobs", type=int, default=8, help="parallel downloads")
    ap.add_argument("--workers", type=int, default=8, help="parallel month converters")
    ap.add_argument("--keep_parts", action="store_true",
                    help="keep csv_parts/ after the final concatenation")
    ap.add_argument("--keep_cache", action="store_true",
                    help="keep cache/ (per-month .npy intermediates) after a successful run")
    ap.add_argument("--skip_convert", action="store_true", help="download only")
    args = ap.parse_args()

    train_months, test_months = month_range(args.train_months), month_range(args.test_months)
    months = train_months + test_months
    if len(set(months)) != len(months):
        raise SystemExit("train and test month ranges overlap")
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    print("== downloading %d months (%s .. %s)" % (len(months), months[0], months[-1]), flush=True)
    download_all(months, args.jobs)
    if args.skip_convert:
        return 0

    keep, dropped_cols = common_columns(months)
    columns = [dst for src, dst in FEATURES if _norm(src) in keep]
    missing = [dst for src, dst in FEATURES if _norm(src) not in keep]
    if missing:
        print("WARNING: features absent from some months, excluded: %s" % missing, flush=True)

    os.makedirs(CACHE_DIR, exist_ok=True)
    print("== stage 1: parquet -> per-month arrays", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        stats = list(ex.map(prepare_month, [(m, keep) for m in months]))
    stats = {s["month"]: s for s in stats}
    print("   stage 1 done in %.0f s" % (time.time() - t0), flush=True)
    seen_flags = sorted({v for s in stats.values() for v in s["flag_values"]})
    if not set(seen_flags) <= set(FLAG_MAP):
        raise SystemExit("unexpected %s values %s; extend FLAG_MAP" % (FLAG_COL, seen_flags))

    # train median of trip_duration_s and train column means (non-NaN only)
    t0 = time.time()
    dur = np.concatenate([np.load(os.path.join(CACHE_DIR, "%s_d.npy" % m)) for m in train_months])
    median = float(np.median(dur))
    n_train_rows = int(dur.size)
    del dur
    means, train_nan = {}, {}
    for name in columns:
        s = sum(stats[m]["sums"][name] for m in train_months)
        nan = sum(stats[m]["nan_counts"][name] for m in train_months)
        train_nan[name] = nan
        cnt = n_train_rows - nan
        means[name] = (s / cnt) if cnt > 0 else 0.0
    print("== train median trip_duration_s = %.1f s (over %d rows), means in %.0f s"
          % (median, n_train_rows, time.time() - t0), flush=True)

    # invalidate cached parts if the parameters changed
    os.makedirs(PARTS_DIR, exist_ok=True)
    params = {"median": median, "means": means, "columns": columns,
              "train_months": train_months, "test_months": test_months}
    p_path = os.path.join(PARTS_DIR, "params.json")
    if os.path.exists(p_path):
        with open(p_path) as f:
            if json.load(f) != params:
                print("   parameters changed -> regenerating CSV parts", flush=True)
                for f_ in os.listdir(PARTS_DIR):
                    os.remove(os.path.join(PARTS_DIR, f_))
    with open(p_path, "w") as f:
        json.dump(params, f)

    print("== stage 2: per-month CSV parts", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = dict(ex.map(write_part, [(m, means, median, columns) for m in months]))
    print("   stage 2 done in %.0f s" % (time.time() - t0), flush=True)

    meta = {"source": "NYC TLC yellow taxi trip records, "
                      "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page",
            "url_template": URL,
            "target_rule": "trip_duration_s = tpep_dropoff_datetime - tpep_pickup_datetime "
                           "(integer seconds); class = 1 iff trip_duration_s > "
                           "median(train trip_duration_s); the same threshold is applied to test",
            "train_median_trip_duration_s": median,
            "dropped_columns": {
                "tpep_dropoff_datetime": "defines the target",
                **{c: "absent from at least one selected month" for c in dropped_cols},
                **{c: "absent from at least one selected month" for c in missing}},
            "encodings": {
                "store_and_fwd_flag": {"map": FLAG_MAP, "cardinality": len(FLAG_MAP),
                                       "rule": "distinct values sorted alphabetically -> 0,1; "
                                               "null -> NaN -> train mean of the codes"},
                "pickup_epoch_s": "tpep_pickup_datetime -> Unix epoch seconds, stored fp32 "
                                  "(24-bit mantissa at ~1.7e9 => 128 s quantisation)"},
            "imputation": {"rule": "NaN -> TRAIN column mean (test reuses it); "
                                   "all-NaN column -> 0.0",
                           "train_means": means,
                           "train_nan_cells": train_nan},
            "columns": ["class"] + columns,
            "splits": {}}

    print("== stage 3: concatenating", flush=True)
    for split, ms in (("train", train_months), ("test", test_months)):
        out = os.path.join(OUT_DIR, "%s_%s.csv" % (args.out_prefix, split))
        t0 = time.time()
        size = concat_parts(ms, columns, out)
        rows = sum(stats[m]["n_rows"] for m in ms)
        pos = sum(results[m]["positives"] for m in ms)
        meta["splits"][split] = {
            "file": os.path.basename(out), "months": [ms[0], ms[-1]], "n_months": len(ms),
            "rows": rows, "features": len(columns), "csv_bytes": size,
            "parquet_rows": sum(stats[m]["n_raw"] for m in ms),
            "dropped_rows_null_timestamp": sum(stats[m]["n_dropped_null_ts"] for m in ms),
            "positive_rate": pos / rows,
            "nan_cells_imputed": {c: sum(stats[m]["nan_counts"][c] for m in ms) for c in columns},
            "source_files": ["parquet/yellow_tripdata_%s.parquet" % m for m in ms]}
        print("   %s: %d rows x %d features, %.2f GB, %.0f s"
              % (out, rows, len(columns), size / 1e9, time.time() - t0), flush=True)

    meta_path = os.path.join(OUT_DIR, "%s_meta.json" % args.out_prefix)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print("wrote %s" % meta_path, flush=True)

    if not args.keep_parts:
        shutil.rmtree(PARTS_DIR, ignore_errors=True)
    if not args.keep_cache:
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
    print("total %.0f s" % (time.time() - t_start), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
