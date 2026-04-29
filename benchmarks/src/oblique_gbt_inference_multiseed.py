#!/usr/bin/env python3
"""Re-run a chosen set of (dataset, split, depth) configs with multiple
seeds to estimate variance.

Reads the same flags as oblique_gbt_inference_sweep.py but takes
explicit "configs" (each "dataset:split:depth") and seeds.
"""

import argparse
import csv
import sys
import tempfile
import time
from pathlib import Path

# Reuse helpers from sibling module via path injection.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oblique_gbt_inference_sweep as base  # noqa: E402

REPO = base.REPO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--configs", required=True,
        help="Comma-separated 'dataset:split:depth' triples, e.g. "
        "'SUSY:Oblique:4,SUSY:AxisAligned:6,HIGGS:Oblique:5'.")
    ap.add_argument("--seeds", default="1234,5678,9999")
    ap.add_argument("--num_trees", type=int, default=100)
    ap.add_argument("--shrinkage", type=float, default=0.1)
    ap.add_argument("--num_proj_exp", type=float, default=1.0)
    ap.add_argument("--max_num_proj", type=int, default=1000)
    ap.add_argument("--density", type=float, default=2.0)
    ap.add_argument("--num_split_type", default="Exact")
    ap.add_argument("--num_threads", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--num_runs", type=int, default=20)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--log_dir", default=None)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    triples = []
    for c in args.configs.split(","):
        ds, split, d = c.split(":")
        split = "Axis Aligned" if split == "AxisAligned" else split
        triples.append((ds, split, int(d)))

    log_dir = Path(args.log_dir) if args.log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not out_path.exists() or out_path.stat().st_size == 0
    fields = [
        "dataset", "split_type", "max_depth", "seed",
        "num_trees_requested", "num_trees_final", "shrinkage",
        "num_proj_exp", "density", "max_num_proj", "num_split_type",
        "train_seconds", "accuracy", "log_loss",
        "best_engine", "best_us_per_ex",
        "generic_us_per_ex", "slow_us_per_ex",
        "all_engines",
    ]
    with open(out_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if is_new:
            w.writeheader()
            fh.flush()
        plan = [(ds, sp, d, s) for (ds, sp, d) in triples for s in seeds]
        for i, (ds, split, depth, seed) in enumerate(plan):
            train_csv = REPO / "benchmarks/data/splits" / f"{ds}_train_500k.csv"
            test_csv = REPO / "benchmarks/data/splits" / f"{ds}_test_100k.csv"
            tag = (f"{ds}_{split.replace(' ', '')}_d{depth}_s{seed}")
            print(f"\n=== [{i+1}/{len(plan)}] {tag} ===", flush=True)
            with tempfile.TemporaryDirectory() as tmp:
                mdir = Path(tmp) / "model"
                t0 = time.perf_counter()
                tsec, final_trees, train_log = base.train_one(
                    train_csv, "class", split, depth, args.num_trees,
                    args.shrinkage, args.num_proj_exp, args.max_num_proj,
                    args.density, args.num_threads, mdir,
                    args.num_split_type, "continuous", seed)
                if tsec is None:
                    continue
                bench_rows, bench_log = base.bench_one(
                    mdir, test_csv, args.batch_size, args.num_runs)
                if bench_rows is None:
                    continue
                acc, ll, eval_log = base.eval_one(mdir, test_csv)
                bench_sorted = sorted(bench_rows, key=lambda r: r[0])
                best_us, best_name = bench_sorted[0]
                generic_us = next(
                    (us for us, n in bench_rows
                     if "Generic" in n and "slow" not in n.lower()),
                    None)
                slow_us = next(
                    (us for us, n in bench_rows if "slow" in n.lower()),
                    None)
                row = {
                    "dataset": ds,
                    "split_type": split,
                    "max_depth": depth,
                    "seed": seed,
                    "num_trees_requested": args.num_trees,
                    "num_trees_final": final_trees if final_trees else "",
                    "shrinkage": args.shrinkage,
                    "num_proj_exp": args.num_proj_exp,
                    "density": args.density,
                    "max_num_proj": args.max_num_proj,
                    "num_split_type": args.num_split_type,
                    "train_seconds": f"{tsec:.3f}",
                    "accuracy": acc if acc is not None else "",
                    "log_loss": ll if ll is not None else "",
                    "best_engine": best_name,
                    "best_us_per_ex": f"{best_us:.4f}",
                    "generic_us_per_ex": (f"{generic_us:.4f}"
                                          if generic_us else ""),
                    "slow_us_per_ex": (f"{slow_us:.4f}" if slow_us else ""),
                    "all_engines": ";".join(
                        f"{n}={us:.4f}" for us, n in bench_sorted),
                }
                w.writerow(row)
                fh.flush()
                wall = time.perf_counter() - t0
                print(f"  seed={seed} acc={acc} train={tsec:.1f}s "
                      f"best={best_name}@{best_us:.4f}us  wall={wall:.1f}s",
                      flush=True)
                if log_dir:
                    (log_dir / f"{tag}_train.log").write_text(train_log)
                    (log_dir / f"{tag}_bench.log").write_text(bench_log)
                    if eval_log:
                        (log_dir / f"{tag}_eval.log").write_text(eval_log)


if __name__ == "__main__":
    main()
