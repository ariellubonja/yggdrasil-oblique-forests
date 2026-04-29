#!/usr/bin/env python3
"""Oblique vs axis-aligned GBT: accuracy x inference-latency sweep.

Trains a GBT for each (dataset, split_type, max_depth) combination,
evaluates accuracy on a held-out test CSV, and benchmarks inference
latency with the cli/benchmark_inference binary.

Outputs a CSV row per config:
  dataset, split_type, max_depth, num_trees, train_secs,
  accuracy, log_loss,
  best_engine_name, best_engine_us_per_ex,
  generic_engine_us_per_ex, slow_engine_us_per_ex
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRAIN_BIN = REPO / "bazel-bin" / "examples" / "train_oblique_forest"
BENCH_BIN = REPO / "bazel-bin" / "yggdrasil_decision_forests" / "cli" / "benchmark_inference"
EVAL_BIN = REPO / "bazel-bin" / "yggdrasil_decision_forests" / "cli" / "evaluate"


def run(cmd, **kw):
    """Run command, return (stdout+stderr, returncode)."""
    proc = subprocess.run(cmd, capture_output=True, text=True, **kw)
    out = (proc.stdout or "") + (proc.stderr or "")
    return out, proc.returncode


def train_one(dataset_csv, label_col, split_type, max_depth, num_trees,
              shrinkage, num_proj_exp, max_num_proj, density,
              num_threads, model_dir, num_split_type, weights_kind,
              seed):
    """Train one model and return (train_seconds, full_log)."""
    if model_dir.exists():
        shutil.rmtree(model_dir)
    cmd = [
        str(TRAIN_BIN),
        "--input_mode=csv",
        f"--train_csv={dataset_csv}",
        f"--label_col={label_col}",
        f"--feature_split_type={split_type}",
        "--ensemble_method=Boosting",
        f"--num_trees={num_trees}",
        f"--tree_depth={max_depth}",
        f"--num_threads={num_threads}",
        f"--shrinkage={shrinkage}",
        f"--num_projections_exponent={num_proj_exp}",
        f"--max_num_projections={max_num_proj}",
        f"--projection_density_factor={density}",
        f"--numerical_split_type={num_split_type}",
        f"--seed={seed}",
        f"--model_out_dir={model_dir}",
    ]
    t0 = time.perf_counter()
    out, rc = run(cmd)
    train_sec = time.perf_counter() - t0
    if rc != 0:
        print("[ERR] training failed:")
        print(out[-3000:])
        return None, out
    # Extract reported wall-time if present.
    m = re.search(r"wall-time.*?Training \+ Post-Processing: ([0-9.]+)s", out)
    if m:
        train_sec = float(m.group(1))
    # Note: weights_kind currently ignored — defaults to whatever
    # the GBT learner picks (continuous when NumericalSplit::EXACT).
    _ = weights_kind  # silence linter
    # Parse final tree count after early-stopping.
    m = re.search(r"Truncates the model to (\d+) tree", out)
    final_trees = int(m.group(1)) if m else None
    return train_sec, final_trees, out


# Engines that benchmark_inference reports.  Names taken from
# register_engines.cc / quick_scorer_extended.cc.
ENGINE_PAT = re.compile(
    r"\s*([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+(.+?)\s*$"
)


def parse_bench(out):
    """Returns list of (us_per_ex, name) sorted slowest-to-fastest in
    the parsed order they appear (we don't reorder)."""
    rows = []
    in_table = False
    for line in out.splitlines():
        line = line.rstrip()
        if line.startswith("time/example"):
            in_table = True
            continue
        if in_table:
            if set(line.strip()) == {"-"}:
                continue
            if not line.strip():
                continue
            m = ENGINE_PAT.match(line)
            if m:
                try:
                    us = float(m.group(1))
                    name = m.group(3).strip()
                    rows.append((us, name))
                except ValueError:
                    pass
    return rows


def bench_one(model_dir, dataset_csv, batch_size, num_runs):
    cmd = [
        str(BENCH_BIN),
        f"--model={model_dir}",
        f"--dataset=csv:{dataset_csv}",
        f"--batch_size={batch_size}",
        f"--num_runs={num_runs}",
        "--warmup_runs=1",
    ]
    out, rc = run(cmd)
    if rc != 0:
        print("[ERR] benchmark failed:")
        print(out[-3000:])
        return None, out
    return parse_bench(out), out


ACC_PAT = re.compile(r"^Accuracy:\s+([0-9.]+)")
LOG_PAT = re.compile(r"^LogLoss:\s*:\s*([0-9.]+)")


def eval_one(model_dir, dataset_csv):
    cmd = [
        str(EVAL_BIN),
        f"--model={model_dir}",
        f"--dataset=csv:{dataset_csv}",
    ]
    out, rc = run(cmd)
    if rc != 0:
        print("[ERR] evaluate failed:")
        print(out[-3000:])
        return None, None, out
    acc = log_loss = None
    for line in out.splitlines():
        m = ACC_PAT.match(line.strip())
        if m and acc is None:
            acc = float(m.group(1))
        m = LOG_PAT.match(line.strip())
        if m and log_loss is None:
            log_loss = float(m.group(1))
    return acc, log_loss, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="HIGGS,SUSY",
                    help="Comma-separated entries. Each entry is either a "
                    "stem (=> splits/<stem>_train_500k.csv + "
                    "splits/<stem>_test_100k.csv with label 'class') or "
                    "name:train_csv:test_csv[:label_col].")
    ap.add_argument("--axis_depths", default="3,4,5,6,8,10,12")
    ap.add_argument("--oblique_depths", default="2,3,4,5,6,8")
    ap.add_argument("--num_trees", type=int, default=100)
    ap.add_argument("--shrinkage", type=float, default=0.1)
    ap.add_argument("--num_proj_exp", type=float, default=1.0)
    ap.add_argument("--max_num_proj", type=int, default=1000)
    ap.add_argument("--density", type=float, default=2.0)
    ap.add_argument("--num_split_type", default="Exact")
    ap.add_argument("--weights_kind", default="continuous",
                    choices=["continuous", "binary"])
    ap.add_argument("--num_threads", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--num_runs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--log_dir", default=None,
                    help="If set, save per-config logs.")
    ap.add_argument("--include_axis", action="store_true", default=True)
    ap.add_argument("--include_oblique", action="store_true", default=True)
    ap.add_argument("--skip_axis", dest="include_axis", action="store_false")
    ap.add_argument("--skip_oblique", dest="include_oblique",
                    action="store_false")
    args = ap.parse_args()

    log_dir = Path(args.log_dir) if args.log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    # Build the (dataset, split, depth) plan.
    datasets = []
    for entry in args.datasets.split(","):
        parts = entry.split(":")
        if len(parts) == 1:
            stem = parts[0]
            train = REPO / "benchmarks/data/splits" / f"{stem}_train_500k.csv"
            test = REPO / "benchmarks/data/splits" / f"{stem}_test_100k.csv"
            label = "class"
        elif len(parts) in (3, 4):
            stem = parts[0]
            train = Path(parts[1]) if Path(parts[1]).is_absolute() \
                else REPO / parts[1]
            test = Path(parts[2]) if Path(parts[2]).is_absolute() \
                else REPO / parts[2]
            label = parts[3] if len(parts) == 4 else "class"
        else:
            print(f"[ERR] malformed --datasets entry: {entry}")
            sys.exit(2)
        if not train.exists() or not test.exists():
            print(f"[ERR] missing splits for {stem}: {train} / {test}")
            sys.exit(2)
        datasets.append((stem, train, test, label))

    plan = []
    for ds, train_csv, test_csv, label in datasets:
        if args.include_axis:
            for d in args.axis_depths.split(","):
                plan.append((ds, train_csv, test_csv, label,
                             "Axis Aligned", int(d)))
        if args.include_oblique:
            for d in args.oblique_depths.split(","):
                plan.append((ds, train_csv, test_csv, label,
                             "Oblique", int(d)))

    # Open output CSV with append (safe for resuming).
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not out_path.exists() or out_path.stat().st_size == 0
    fields = [
        "dataset", "split_type", "max_depth",
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

        for i, (ds, train_csv, test_csv, label, split, depth) in enumerate(plan):
            tag = f"{ds}_{split.replace(' ', '')}_d{depth}"
            print(f"\n=== [{i+1}/{len(plan)}] {tag} ===", flush=True)
            with tempfile.TemporaryDirectory() as tmp:
                mdir = Path(tmp) / "model"
                t0 = time.perf_counter()
                tsec, final_trees, train_log = train_one(
                    train_csv, label, split, depth, args.num_trees,
                    args.shrinkage, args.num_proj_exp, args.max_num_proj,
                    args.density, args.num_threads, mdir,
                    args.num_split_type, args.weights_kind, args.seed)
                if tsec is None:
                    continue
                bench_rows, bench_log = bench_one(
                    mdir, test_csv, args.batch_size, args.num_runs)
                if bench_rows is None:
                    continue
                acc, ll, eval_log = eval_one(mdir, test_csv)

                # The benchmark_inference output puts entries in
                # parsed order; pick fastest as best.
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
                print(f"  acc={acc} ll={ll} train={tsec:.1f}s "
                      f"best={best_name}@{best_us:.4f}us  "
                      f"generic={generic_us}us  wall={wall:.1f}s",
                      flush=True)
                if log_dir:
                    (log_dir / f"{tag}_train.log").write_text(train_log)
                    (log_dir / f"{tag}_bench.log").write_text(bench_log)
                    if eval_log:
                        (log_dir / f"{tag}_eval.log").write_text(eval_log)


if __name__ == "__main__":
    main()
