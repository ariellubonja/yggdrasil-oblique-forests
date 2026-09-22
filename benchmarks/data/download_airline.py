#!/usr/bin/env python3
"""Download the BTS "Reporting Carrier On-Time Performance (1987-present)" data and
convert it to all-numeric binary-classification CSVs for the oblique-RF harness.

Source: US DOT / Bureau of Transportation Statistics, TranStats
  https://transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ   (US Government work, public domain)
  One zip per month:
    https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_<YYYY>_<M>.zip
  (month without zero padding, ~27-35 MB each).  Each zip holds
  `On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_<YYYY>_<M>.csv`
  (~540k rows) plus a readme.html.  The CSV has 109 named columns and a trailing
  comma on every line, so the parser sees a 110th, unnamed, always-empty column.

Binary task: the dataset's own `ArrDel15` flag, i.e.
    class = 1  iff the flight arrived 15+ minutes late (ArrDelayMinutes >= 15)
    class = 0  otherwise
`ArrDel15` is null for cancelled and diverted flights (~2-3 % of rows); those rows
have no label and are the only rows this script drops (counted per split in the meta
JSON).  The majority class is 0 (~82 %), so the repo's "majority -> 0" convention is
already satisfied and no flip is applied.

Leakage: every column that is derived from the arrival, from the delay itself, or
from a diversion is DROPPED (listed in the meta JSON under "dropped_columns"):
  ArrTime, ArrDelay, ArrDelayMinutes, ArrDel15 (the target), ArrivalDelayGroups,
  ActualElapsedTime, AirTime, TaxiIn, WheelsOn, Cancelled, CancellationCode,
  Diverted, CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, LateAircraftDelay,
  FirstDepTime, TotalAddGTime, LongestAddGTime, DivAirportLandings, DivReachedDest,
  DivActualElapsedTime, DivArrDelay, DivDistance, all 40 Div1..Div5 columns,
  Flights (constant 1) and the unnamed trailing column.
Everything known at (or before) wheels-off is KEPT: the schedule fields plus the
departure-side fields DepTime, DepDelay, DepDelayMinutes, DepDel15,
DepartureDelayGroups, TaxiOut, WheelsOff.  That leaves 43 features.

Encoding (repo dataset policy, CLAUDE.md 2026-09-21):
  * FlightDate            -> epoch seconds (UTC midnight; a multiple of 86400, hence
                             exactly representable in fp32 -- no precision loss for
                             the dates in range, though the fp32 ULP there is 128 s).
  * 13 string columns     -> ordinal codes, distinct values as strings sorted
    (Reporting_Airline, IATA_CODE_Reporting_Airline, Tail_Number, Origin,
     OriginCityName, OriginState, OriginStateName, Dest, DestCityName, DestState,
     DestStateName, DepTimeBlk, ArrTimeBlk)
                             alphabetically -> 0,1,2,...  The map is built over
                             train UNION test so one map serves both splits.
  * time-of-day fields (CRSDepTime, DepTime, WheelsOff, CRSArrTime) are hhmm
                             integers in the source and are kept as plain numbers.
  * any remaining null cell -> the TRAIN column mean (for a categorical column, the
                             train mean of its codes, computed after encoding).
  * Column names are the original names, lower-cased.

Two passes over the months are needed: pass 1 collects the string uniques and the
train sums/counts, pass 2 encodes and writes one CSV chunk per month; the chunks are
then concatenated in chronological order.  Both passes are cached on disk, so a
re-run only does the work that is missing.

  benchmarks/data/download_airline.py
  benchmarks/data/download_airline.py --train_months 2022-01:2023-12 --test_months 2024-12

Outputs (all in benchmarks/data/airline/, gitignored):
  raw/On_Time_<YYYY>_<M>.zip   the downloaded monthly zips (kept; never deleted)
  chunks/<split>_<YYYY>_<M>.csv  per-month encoded chunks (headerless, cache)
  stats/<YYYY>_<M>.json          per-month pass-1 statistics (cache)
  airline_train.csv              class + 43 features, chronological (month order)
  airline_test.csv               same columns, later months
  airline_meta.json              shapes, positive rates, target rule, drops, encodings
"""
import argparse
import io
import json
import os
import shutil
import sys
import time
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "airline")
RAW_DIR = os.path.join(OUT_DIR, "raw")
CHUNK_DIR = os.path.join(OUT_DIR, "chunks")
STAT_DIR = os.path.join(OUT_DIR, "stats")
URL = ("https://transtats.bts.gov/PREZIP/"
       "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{y}_{m}.zip")

TARGET = "ArrDel15"

# --- columns ---------------------------------------------------------------
DATE_COLS = ["FlightDate"]
CAT_COLS = [
    "Reporting_Airline", "IATA_CODE_Reporting_Airline", "Tail_Number",
    "Origin", "OriginCityName", "OriginState", "OriginStateName",
    "Dest", "DestCityName", "DestState", "DestStateName",
    "DepTimeBlk", "ArrTimeBlk",
]
NUM_COLS = [
    "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek",
    "DOT_ID_Reporting_Airline", "Flight_Number_Reporting_Airline",
    "OriginAirportID", "OriginAirportSeqID", "OriginCityMarketID",
    "OriginStateFips", "OriginWac",
    "DestAirportID", "DestAirportSeqID", "DestCityMarketID",
    "DestStateFips", "DestWac",
    "CRSDepTime", "DepTime", "DepDelay", "DepDelayMinutes", "DepDel15",
    "DepartureDelayGroups", "TaxiOut", "WheelsOff",
    "CRSArrTime", "CRSElapsedTime", "Distance", "DistanceGroup",
]
# Kept in source order (= the order of the output columns after `class`).
FEATURES = [
    "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek", "FlightDate",
    "Reporting_Airline", "DOT_ID_Reporting_Airline", "IATA_CODE_Reporting_Airline",
    "Tail_Number", "Flight_Number_Reporting_Airline",
    "OriginAirportID", "OriginAirportSeqID", "OriginCityMarketID", "Origin",
    "OriginCityName", "OriginState", "OriginStateFips", "OriginStateName", "OriginWac",
    "DestAirportID", "DestAirportSeqID", "DestCityMarketID", "Dest",
    "DestCityName", "DestState", "DestStateFips", "DestStateName", "DestWac",
    "CRSDepTime", "DepTime", "DepDelay", "DepDelayMinutes", "DepDel15",
    "DepartureDelayGroups", "DepTimeBlk", "TaxiOut", "WheelsOff",
    "CRSArrTime", "ArrTimeBlk", "CRSElapsedTime", "Distance", "DistanceGroup",
]
DROPPED = {
    "ArrTime": "post-arrival", "ArrDelay": "target-derived",
    "ArrDelayMinutes": "target-derived", "ArrDel15": "the target itself",
    "ArrivalDelayGroups": "target-derived", "ActualElapsedTime": "post-arrival",
    "AirTime": "post-arrival", "TaxiIn": "post-arrival", "WheelsOn": "post-arrival",
    "Cancelled": "post-departure outcome", "CancellationCode": "post-departure outcome",
    "Diverted": "post-departure outcome", "CarrierDelay": "delay attribution",
    "WeatherDelay": "delay attribution", "NASDelay": "delay attribution",
    "SecurityDelay": "delay attribution", "LateAircraftDelay": "delay attribution",
    "FirstDepTime": "gate-return / post-departure", "TotalAddGTime": "gate-return / post-departure",
    "LongestAddGTime": "gate-return / post-departure",
    "DivAirportLandings": "diversion", "DivReachedDest": "diversion",
    "DivActualElapsedTime": "diversion", "DivArrDelay": "diversion",
    "DivDistance": "diversion",
    "Flights": "constant 1", "": "unnamed trailing column (trailing comma in source)",
}
for _i in range(1, 6):
    for _s in ("Airport", "AirportID", "AirportSeqID", "WheelsOn", "TotalGTime",
               "LongestGTime", "WheelsOff", "TailNum"):
        DROPPED["Div%d%s" % (_i, _s)] = "diversion"

READ_COLS = FEATURES + [TARGET]
COL_TYPES = {c: pa.float64() for c in NUM_COLS}
COL_TYPES.update({c: pa.string() for c in CAT_COLS})
COL_TYPES["FlightDate"] = pa.date32()
COL_TYPES[TARGET] = pa.float64()


# --- helpers ---------------------------------------------------------------
def parse_months(spec):
    """'2018-01:2023-12' or '2024-12' -> [(2018,1), ...] inclusive."""
    def one(s):
        y, m = s.split("-")
        return int(y), int(m)
    if ":" in spec:
        a, b = spec.split(":")
        (y0, m0), (y1, m1) = one(a), one(b)
    else:
        (y0, m0) = (y1, m1) = one(spec)
    out, y, m = [], y0, m0
    while (y, m) <= (y1, m1):
        out.append((y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def raw_path(y, m):
    return os.path.join(RAW_DIR, "On_Time_%d_%d.zip" % (y, m))


def zip_ok(path):
    if not os.path.exists(path) or os.path.getsize(path) < 1_000_000:
        return False
    try:
        z = zipfile.ZipFile(path)
        return any(n.filename.endswith(".csv") for n in z.infolist())
    except Exception:
        return False


def download_month(y, m, retries=3):
    path = raw_path(y, m)
    if zip_ok(path):
        return (y, m, "cached", os.path.getsize(path))
    import urllib.request
    url = URL.format(y=y, m=m)
    tmp = path + ".part"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            os.replace(tmp, path)
            if zip_ok(path):
                return (y, m, "downloaded", os.path.getsize(path))
            os.remove(path)
        except Exception as e:
            err = e
            if os.path.exists(tmp):
                os.remove(tmp)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("could not download %s: %s" % (url, locals().get("err", "corrupt zip")))


def date_seconds(col):
    """date32[day] -> int64 epoch seconds (nulls preserved)."""
    return pc.multiply(pc.cast(pc.cast(col, pa.int32()), pa.int64()), 86400)


def read_month(y, m):
    """Parse one monthly zip -> pyarrow Table of READ_COLS, label-null rows dropped."""
    z = zipfile.ZipFile(raw_path(y, m))
    name = [i.filename for i in z.infolist() if i.filename.endswith(".csv")][0]
    buf = io.BytesIO(z.read(name))
    tb = pcsv.read_csv(
        buf,
        read_options=pcsv.ReadOptions(use_threads=False, block_size=1 << 24),
        parse_options=pcsv.ParseOptions(newlines_in_values=False),
        convert_options=pcsv.ConvertOptions(
            include_columns=READ_COLS, column_types=COL_TYPES,
            strings_can_be_null=True, null_values=["", "NA", "NULL"]),
    )
    n_raw = tb.num_rows
    mask = pc.is_valid(tb[TARGET])
    tb = tb.filter(mask)
    return tb, n_raw


# --- pass 1: per-month statistics -----------------------------------------
def pass1(args):
    y, m = args
    out = os.path.join(STAT_DIR, "%d_%02d.json" % (y, m))
    if os.path.exists(out):
        try:
            with open(out) as f:
                return json.load(f)
        except Exception:
            pass
    tb, n_raw = read_month(y, m)
    st = {"year": y, "month": m, "n_raw": n_raw, "n_rows": tb.num_rows,
          "n_dropped_no_label": n_raw - tb.num_rows,
          "n_pos": int(pc.sum(tb[TARGET]).as_py() or 0),
          "num": {}, "cat": {}, "date": {}}
    for c in NUM_COLS:
        col = tb[c]
        nn = col.length() - col.null_count
        st["num"][c] = [float(pc.sum(col).as_py() or 0.0), int(nn), int(col.null_count)]
    for c in CAT_COLS:
        col = tb[c]
        vc = col.value_counts()
        d = {}
        for entry in vc.to_pylist():
            v = entry["values"]
            if v is not None:
                d[v] = entry["counts"]
        st["cat"][c] = [d, int(col.null_count)]
    for c in DATE_COLS:
        secs = date_seconds(tb[c])
        nn = secs.length() - secs.null_count
        st["date"][c] = [float(pc.sum(secs).as_py() or 0.0), int(nn), int(secs.null_count)]
    with open(out + ".tmp", "w") as f:
        json.dump(st, f)
    os.replace(out + ".tmp", out)
    return st


# --- pass 2: encode + write one chunk per month ---------------------------
_ENC = {}


def _init_enc(enc):
    _ENC.update(enc)


def pass2(args):
    y, m, split = args
    out = os.path.join(CHUNK_DIR, "%s_%d_%02d.csv" % (split, y, m))
    done = out + ".done"
    if os.path.exists(done) and os.path.exists(out):
        with open(done) as f:
            info = json.load(f)
        # the chunk is only valid for the encoding it was written with
        if info.get("enc_key") == _ENC["key"]:
            return info
    maps = _ENC["maps"]
    means = _ENC["means"]
    tb, _ = read_month(y, m)
    tb = tb.combine_chunks()
    n = tb.num_rows
    n_pos = int(pc.sum(tb[TARGET]).as_py() or 0)
    arrays = [pc.cast(tb[TARGET], pa.int8())]
    names = ["class"]
    for c in FEATURES:
        if c in CAT_COLS:
            # dictionary-encode once, then map the (small) dictionary to codes.
            enc = tb[c].chunk(0).dictionary_encode() if tb[c].num_chunks else \
                pa.array([], type=pa.string()).dictionary_encode()
            mp = maps[c]
            lut = np.array([mp.get(v, -1) for v in enc.dictionary.to_pylist()],
                           dtype=np.float32)
            ind = pc.fill_null(enc.indices, 0).to_numpy(zero_copy_only=False).astype(np.int64)
            codes = lut[ind] if len(lut) else np.zeros(n, dtype=np.float32)
            miss = ~(enc.indices.is_valid().to_numpy(zero_copy_only=False).astype(bool))
            bad = miss | (codes < 0)
            if bad.any():
                codes = codes.copy()
                codes[bad] = np.float32(means[c])
            a = pa.array(codes)
        elif c in DATE_COLS:
            secs = date_seconds(tb[c])
            a = pc.cast(pc.fill_null(secs, int(round(means[c]))), pa.float32(), safe=False)
        else:
            a = pc.cast(pc.fill_null(tb[c], means[c]), pa.float32(), safe=False)
        if a.null_count:
            raise RuntimeError("nulls left in %s (%d-%d)" % (c, y, m))
        arrays.append(a)
        names.append(c.lower())
    tbl = pa.Table.from_arrays(arrays, names=names)
    pcsv.write_csv(tbl, out + ".tmp",
                   pcsv.WriteOptions(include_header=False, batch_size=64000))
    os.replace(out + ".tmp", out)
    info = {"year": y, "month": m, "split": split, "n_rows": n, "n_pos": n_pos,
            "bytes": os.path.getsize(out), "path": out, "enc_key": _ENC["key"]}
    with open(done, "w") as f:
        json.dump(info, f)
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_months", default="2018-01:2023-12")
    ap.add_argument("--test_months", default="2024-10:2024-12")
    ap.add_argument("--download_workers", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip_download", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    for d in (OUT_DIR, RAW_DIR, CHUNK_DIR, STAT_DIR):
        os.makedirs(d, exist_ok=True)
    train_months = parse_months(args.train_months)
    test_months = parse_months(args.test_months)
    all_months = train_months + test_months
    assert len(set(all_months)) == len(all_months), "train and test months overlap"
    print("[airline] %d train months, %d test months" % (len(train_months), len(test_months)),
          flush=True)

    # 0. download -----------------------------------------------------------
    if not args.skip_download:
        n_dl = 0
        with ThreadPoolExecutor(max_workers=min(6, args.download_workers)) as ex:
            futs = {ex.submit(download_month, y, m): (y, m) for y, m in all_months}
            for fut in as_completed(futs):
                y, m, how, sz = fut.result()
                if how == "downloaded":
                    n_dl += 1
                    print("[airline] downloaded %d-%02d (%.1f MB)" % (y, m, sz / 1e6), flush=True)
        print("[airline] downloads ready (%d new) %.0fs" % (n_dl, time.time() - t0), flush=True)

    # 1. pass 1 -------------------------------------------------------------
    stats = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for st in ex.map(pass1, all_months):
            stats[(st["year"], st["month"])] = st
    print("[airline] pass 1 done %.0fs" % (time.time() - t0), flush=True)

    # ordinal maps over train UNION test, alphabetical
    maps, cardinalities = {}, {}
    for c in CAT_COLS:
        vals = set()
        for k in all_months:
            vals.update(stats[k]["cat"][c][0].keys())
        mp = {v: i for i, v in enumerate(sorted(vals))}
        maps[c] = mp
        cardinalities[c] = len(mp)

    # train means (numeric / date raw; categorical = mean of codes over train)
    means, null_counts = {}, {}
    for c in NUM_COLS:
        s = sum(stats[k]["num"][c][0] for k in train_months)
        n = sum(stats[k]["num"][c][1] for k in train_months)
        means[c] = (s / n) if n else 0.0
        null_counts[c] = sum(stats[k]["num"][c][2] for k in all_months)
    for c in DATE_COLS:
        s = sum(stats[k]["date"][c][0] for k in train_months)
        n = sum(stats[k]["date"][c][1] for k in train_months)
        means[c] = (s / n) if n else 0.0
        null_counts[c] = sum(stats[k]["date"][c][2] for k in all_months)
    for c in CAT_COLS:
        tot = cnt = 0.0
        for k in train_months:
            for v, n in stats[k]["cat"][c][0].items():
                tot += maps[c][v] * n
                cnt += n
        means[c] = (tot / cnt) if cnt else 0.0
        null_counts[c] = sum(stats[k]["cat"][c][1] for k in all_months)

    # 2. pass 2 -------------------------------------------------------------
    import hashlib
    enc_key = hashlib.sha1(json.dumps(
        [cardinalities, {k: round(v, 6) for k, v in sorted(means.items())},
         FEATURES], sort_keys=True).encode()).hexdigest()[:16]
    jobs = ([(y, m, "train") for y, m in train_months] +
            [(y, m, "test") for y, m in test_months])
    infos = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_enc,
                             initargs=({"maps": maps, "means": means,
                                        "key": enc_key},)) as ex:
        for info in ex.map(pass2, jobs):
            infos[(info["split"], info["year"], info["month"])] = info
    print("[airline] pass 2 done %.0fs" % (time.time() - t0), flush=True)

    # 3. concatenate in chronological order ---------------------------------
    header = ",".join(["class"] + [c.lower() for c in FEATURES]) + "\n"
    summary = {}
    for split, months in (("train", train_months), ("test", test_months)):
        dst = os.path.join(OUT_DIR, "airline_%s.csv" % split)
        with open(dst + ".tmp", "wb") as out:
            out.write(header.encode())
            for y, m in months:
                with open(infos[(split, y, m)]["path"], "rb") as f:
                    shutil.copyfileobj(f, out, 1 << 22)
        os.replace(dst + ".tmp", dst)
        rows = sum(infos[(split, y, m)]["n_rows"] for y, m in months)
        pos = sum(infos[(split, y, m)]["n_pos"] for y, m in months)
        summary[split] = {
            "path": dst, "rows": rows, "cols": 1 + len(FEATURES),
            "n_features": len(FEATURES), "positive_rate": pos / rows if rows else 0.0,
            "n_positive": pos,
            "rows_dropped_null_label": sum(stats[(y, m)]["n_dropped_no_label"] for y, m in months),
            "rows_before_label_drop": sum(stats[(y, m)]["n_raw"] for y, m in months),
            "bytes": os.path.getsize(dst),
            "months": ["%d-%02d" % (y, m) for y, m in months],
        }
        print("[airline] %s: %d x %d, pos=%.4f, %.2f GB" %
              (split, rows, 1 + len(FEATURES), summary[split]["positive_rate"],
               summary[split]["bytes"] / 1e9), flush=True)

    # 4. meta ---------------------------------------------------------------
    meta = {
        "name": "airline",
        "source": "US DOT/BTS TranStats, Reporting Carrier On-Time Performance (1987-present)",
        "source_url": "https://transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ",
        "licence": "US Government work, public domain",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": "benchmarks/data/download_airline.py",
        "rerun_command": ("benchmarks/data/download_airline.py --train_months %s "
                          "--test_months %s" % (args.train_months, args.test_months)),
        "target_rule": ("class = ArrDel15 (1 iff ArrDelayMinutes >= 15, the dataset's own "
                        "flag). Majority class is 0, so no flip. Rows with a null ArrDel15 "
                        "(cancelled / diverted flights) have no label and are dropped."),
        "splits": summary,
        "columns": ["class"] + [c.lower() for c in FEATURES],
        "dropped_columns": DROPPED,
        "encoding": {
            "date_epoch_seconds": {c: {"train_mean_used_for_nulls": means[c]} for c in DATE_COLS},
            "ordinal_maps": {c: {"cardinality": cardinalities[c],
                                 "train_mean_of_codes_used_for_nulls": means[c]}
                             for c in CAT_COLS},
            "numeric_train_means_used_for_nulls": {c: means[c] for c in NUM_COLS},
            "null_cells_imputed_per_column": {c: null_counts[c] for c in FEATURES},
            "notes": [
                "ordinal maps are built over train UNION test, distinct values as strings "
                "sorted alphabetically -> 0,1,2,...",
                "a null categorical cell is imputed with the TRAIN mean of the codes "
                "(i.e. after encoding)",
                "FlightDate -> epoch seconds at UTC midnight; those are multiples of 86400 "
                "and exactly representable in fp32 (fp32 ULP in that range is 128 s)",
                "time-of-day columns (CRSDepTime, DepTime, WheelsOff, CRSArrTime) are hhmm "
                "integers in the source and are kept as plain numbers",
                "all values are written as fp32-rounded decimals",
            ],
        },
        "source_files": ["benchmarks/data/airline/raw/On_Time_%d_%d.zip" % (y, m)
                         for y, m in all_months],
        "per_month_rows": {"%d-%02d" % (y, m): stats[(y, m)]["n_rows"] for y, m in all_months},
    }
    mp = os.path.join(OUT_DIR, "airline_meta.json")
    with open(mp, "w") as f:
        json.dump(meta, f, indent=1)
    print("[airline] wrote %s   total %.0fs" % (mp, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
