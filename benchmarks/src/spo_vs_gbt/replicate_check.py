#!/usr/bin/env python3
"""Spot-check replicability of the SPO-vs-GBT study (REPLICABILITY.md, section 6).

Samples N recorded rows from the study's result CSVs, re-runs each one through
the SAME code path the original run used (run_suite.run_ydf / run_suite.run_python
/ run_speedup_map.run_cell), and compares:

  * accuracy fields (test_acc, test_auc, test_logloss): must match EXACTLY
    (|delta| <= --acc-tol, default 0). YDF is deterministic at a fixed seed and
    thread count; xgboost/lightgbm/catboost are deterministic at a fixed
    random_state + n_jobs on the same library versions.
  * train_s: must be within --time-tol relative (default 0.15 = 15 %).
  * the harness command line (YDF arms): the freshly built argv must equal the
    recorded `cmd` column verbatim (catches a changed arm definition, binary
    path, flag default, or data path).

Usage (from the repo checkout, never while run_all.sh holds its lock):

  /home/ubuntu/gbt_venv/bin/python replicate_check.py --n 6 --sample-seed 0
  /home/ubuntu/gbt_venv/bin/python replicate_check.py --n 3 --sources large --exclude-slow
  /home/ubuntu/gbt_venv/bin/python replicate_check.py --rows "diabetes,0,spo_rf_dyn_vec" \
        --rows "HIGGS,0,xgboost"          # explicit rows: dataset,fold,method[,rep[,seed]]

Every re-run is logged to --log-dir (default logs/replication/) and a CSV of
recorded-vs-new values with PASS/FAIL per field is written to --out.
Exit code 0 = every sampled row passed, 1 = at least one failure, 2 = refused
(chain lock held / missing inputs).
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import random
import shlex
import sys
import time

import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
import run_suite  # noqa: E402
import run_speedup_map  # noqa: E402
from arms import ARMS  # noqa: E402

WORK_DIR = "/home/ubuntu/spo_vs_gbt"
RESULTS_DIR = os.path.join(WORK_DIR, "results")
FOLDS_DIR = os.path.join(WORK_DIR, "folds")
BIN_DIR = os.path.join(WORK_DIR, "bin")
LOCK_FILE = os.path.join(WORK_DIR, "run_all.lock")
REPO = "/home/ubuntu/yggdrasil-oblique-forests"

# Mirrors run_all.sh's LARGE_DATASETS (NAME -> train, test, label).
LARGE = {
    "HIGGS": (f"{REPO}/benchmarks/data/HIGGS_train_10500k.csv",
              f"{REPO}/benchmarks/data/HIGGS_test_500k.csv", "class"),
    "SUSY": (f"{REPO}/benchmarks/data/SUSY_train_4500k.csv",
             f"{REPO}/benchmarks/data/SUSY_test_500k.csv", "class"),
    "EPSILON": (f"{REPO}/benchmarks/data/epsilon_normalized_train.csv",
                f"{WORK_DIR}/data/epsilon_test_100k.csv", "label"),
}

SOURCES = {
    "suite": os.path.join(RESULTS_DIR, "suite_results.csv"),
    "large": os.path.join(RESULTS_DIR, "large_results.csv"),
    "speedup": os.path.join(RESULTS_DIR, "speedup_map.csv"),
}
ACC_FIELDS = ["test_acc", "test_auc", "test_logloss"]


def refuse_if_chain_running() -> None:
    if not os.path.exists(LOCK_FILE):
        return
    fd = os.open(LOCK_FILE, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
    except BlockingIOError:
        print("REFUSED: run_all.sh holds", LOCK_FILE,
              "-- never overlap two 48-thread trainings (timings inflate ~1.7x)")
        sys.exit(2)
    finally:
        os.close(fd)


def fold_paths(dataset: str, fold: int) -> tuple[str, str, str]:
    if dataset in LARGE:
        return LARGE[dataset]
    ds_dir = os.path.join(FOLDS_DIR, dataset)
    with open(os.path.join(ds_dir, "folds.json")) as f:
        label_col = json.load(f)["label_col"]
    return (os.path.join(ds_dir, f"fold{fold}_train.csv"),
            os.path.join(ds_dir, f"fold{fold}_test.csv"), label_col)


def sample_rows(args: argparse.Namespace) -> list[dict]:
    rng = random.Random(args.sample_seed)
    picked: list[dict] = []
    if args.rows:
        for spec in args.rows:
            parts = spec.split(",")
            ds, fold, method = parts[0], int(parts[1]), parts[2]
            rep = int(parts[3]) if len(parts) > 3 else None
            seed = int(parts[4]) if len(parts) > 4 else None
            src = "large" if ds in LARGE else "suite"
            if method not in ARMS:  # speedup-map rows are keyed by arm + rows/depth
                src = "speedup"
            df = pd.read_csv(SOURCES[src])
            key = "arm" if src == "speedup" else "method"
            m = (df.dataset == ds) & (df[key] == method)
            if src != "speedup":
                m &= df.fold == fold
                m &= df.rep == (rep if rep is not None else 0)
                m &= df.seed == (seed if seed is not None else 1)
            sub = df[m]
            if sub.empty:
                print("no recorded row for", spec)
                sys.exit(2)
            for _, r in sub.iterrows():
                picked.append({"source": src, **r.to_dict()})
        return picked

    for src in args.sources.split(","):
        df = pd.read_csv(SOURCES[src])
        df = df[df.status.isin(["OK", "SLOWPATH"])]
        if args.exclude_slow:
            df = df[pd.to_numeric(df.train_s, errors="coerce").fillna(0) <= args.max_train_s]
        pool = list(df.index)
        rng.shuffle(pool)
        for idx in pool[: args.n]:
            picked.append({"source": src, **df.loc[idx].to_dict()})
    return picked


def rerun(row: dict, args: argparse.Namespace) -> dict:
    src = row["source"]
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if src == "speedup":
        argv = shlex.split(row["cmd"])
        log_path = os.path.join(args.log_dir, f"{row['dataset']}_{row['arm']}_{ts}.log")
        res = run_speedup_map.run_cell(argv, args.timeout, log_path)
        res["cmd_new"] = shlex.join(argv)
        return res
    method = row["method"]
    arm = ARMS[method]
    train, test, label = fold_paths(row["dataset"], int(row["fold"]))
    seed = int(row["seed"]) if not pd.isna(row["seed"]) else 1
    log_path = os.path.join(args.log_dir, f"{row['dataset']}_{row['fold']}_{method}_{ts}.log")
    if arm["engine"] == "ydf_fork":
        res = run_suite.run_ydf(arm, train, test, label, int(row["threads"]), seed,
                                BIN_DIR, args.timeout, log_path)
    else:
        res = run_suite.run_python(method, arm, train, test, label, int(row["threads"]),
                                   seed, log_path, args.timeout)
    res["cmd_new"] = res.get("cmd", "")
    return res


def compare(row: dict, res: dict, args: argparse.Namespace) -> dict:
    out = {
        "source": row["source"], "dataset": row["dataset"],
        "fold": row.get("fold", ""), "rep": row.get("rep", ""), "seed": row.get("seed", ""),
        "method": row.get("method", row.get("arm", "")),
        "status_recorded": row["status"], "status_new": res["status"],
    }
    ok = res["status"] in ("OK", "SLOWPATH")
    fails = []
    if not ok:
        fails.append("status")
    # command line identity (YDF rows only; python rows record a ctor repr)
    if row["source"] == "speedup" or ARMS.get(out["method"], {}).get("engine") == "ydf_fork":
        out["cmd_match"] = str(row["cmd"]) == str(res.get("cmd_new", ""))
        if not out["cmd_match"]:
            fails.append("cmd")
    else:
        out["cmd_match"] = ""
    # accuracy
    for f in ACC_FIELDS:
        if f not in row:
            continue
        a = pd.to_numeric(row.get(f), errors="coerce")
        b = pd.to_numeric(res.get(f), errors="coerce")
        out[f"{f}_recorded"], out[f"{f}_new"] = a, b
        if pd.isna(a) and pd.isna(b):
            continue
        d = abs(float(a) - float(b)) if not (pd.isna(a) or pd.isna(b)) else float("inf")
        out[f"{f}_delta"] = d
        if d > args.acc_tol:
            fails.append(f)
    # time
    a = pd.to_numeric(row.get("train_s"), errors="coerce")
    b = pd.to_numeric(res.get("train_s"), errors="coerce")
    out["train_s_recorded"], out["train_s_new"] = a, b
    if not (pd.isna(a) or pd.isna(b)) and float(a) > 0:
        rel = abs(float(b) - float(a)) / float(a)
        out["train_s_rel_delta"] = round(rel, 4)
        if rel > args.time_tol and float(a) >= args.time_floor_s:
            fails.append("train_s")
    out["engine_version_recorded"] = row.get("engine_version", "")
    out["engine_version_new"] = res.get("engine_version", "")
    out["result"] = "PASS" if not fails else "FAIL:" + "+".join(fails)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5, help="rows per source to sample")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--sources", default="suite,large,speedup")
    ap.add_argument("--rows", action="append", default=[],
                    help="explicit 'dataset,fold,method[,rep[,seed]]' (repeatable)")
    ap.add_argument("--exclude-slow", action="store_true",
                    help="only sample rows whose recorded train_s <= --max-train-s")
    ap.add_argument("--max-train-s", type=float, default=600.0)
    ap.add_argument("--acc-tol", type=float, default=0.0)
    ap.add_argument("--time-tol", type=float, default=0.15)
    ap.add_argument("--time-floor-s", type=float, default=2.0,
                    help="ignore the time check for runs shorter than this (noise)")
    ap.add_argument("--timeout", type=int, default=4 * 3600)
    ap.add_argument("--log-dir", default=os.path.join(WORK_DIR, "logs", "replication"))
    ap.add_argument("--out", default="")
    ap.add_argument("--dry-run", action="store_true", help="list the sample, run nothing")
    ap.add_argument("--allow-concurrent", action="store_true",
                    help="skip the run_all.lock guard (ONLY for sub-second rows; timings of "
                         "anything longer are invalid while the chain runs)")
    args = ap.parse_args()

    if not (args.dry_run or args.allow_concurrent):
        refuse_if_chain_running()
    for k, p in SOURCES.items():
        if k in args.sources.split(",") and not os.path.exists(p):
            print("missing", p); return 2
    os.makedirs(args.log_dir, exist_ok=True)
    if not args.out:
        args.out = os.path.join(RESULTS_DIR, "replication",
                                time.strftime("replication_%Y%m%dT%H%M%SZ.csv", time.gmtime()))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = sample_rows(args)
    print(f"sampled {len(rows)} rows (sample_seed={args.sample_seed}):")
    for r in rows:
        print(f"  {r['source']:8s} {r['dataset']:40s} fold={r.get('fold','')} "
              f"{r.get('method', r.get('arm',''))}  train_s={r.get('train_s')}  "
              f"auc={r.get('test_auc','')}")
    if args.dry_run:
        return 0

    results = []
    for i, r in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] re-running {r['dataset']} {r.get('method', r.get('arm',''))} ...",
              flush=True)
        res = rerun(r, args)
        cmp_ = compare(r, res, args)
        results.append(cmp_)
        print(f"    -> {cmp_['result']}  train_s {cmp_['train_s_recorded']} -> "
              f"{cmp_['train_s_new']}  auc {cmp_.get('test_auc_recorded','')} -> "
              f"{cmp_.get('test_auc_new','')}", flush=True)

    keys: list[str] = []
    for r in results:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    n_fail = sum(1 for r in results if r["result"] != "PASS")
    print(f"\n{len(results) - n_fail}/{len(results)} PASS  -> {args.out}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
