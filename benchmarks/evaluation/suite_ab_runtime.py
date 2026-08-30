#!/usr/bin/env python3
"""A/B two prebuilt train_oblique_forest binaries over a directory of CSV datasets.

Built for the PR#356 replication on TabArena / TabReD, where dataset sizes span
four orders of magnitude (748 rows to 150k) and a fixed tree count would put half
the suite in the sub-10-ms noise floor. Two protocol choices follow from that:

* **Calibrated tree count.** Each dataset gets a short probe run; the tree count
  is then scaled so a measured run takes ~`--target-seconds`, clamped to
  [--min-trees, --max-trees]. Both arms of a dataset always use the *same* count,
  so the ratio stays meaningful; the count is recorded per dataset.
* **Interleaved arms (A B B A ...).** Arms alternate within a dataset rather than
  running as two blocks, so slow machine drift hits both arms equally instead of
  landing entirely on whichever ran second. (`benchmarks/results/E2E...` and the
  e2e-session-offset note document ~5-9% cross-session offsets on this box.)

Single-threaded by default: the change under test is a per-node scalar loop, and
--num_threads=1 removes pool scheduling and memory-contention noise. That is a
different protocol from runtime.sh (which uses all cores) — comparable *ratios*,
not comparable absolute seconds.

  python3 benchmarks/evaluation/suite_ab_runtime.py \
      --datasets-dir benchmarks/data/tabarena_binary_csv \
      --arm control=/path/to/bin_a --arm pr356=/path/to/bin_b \
      --out benchmarks/results/pr356_tabarena_oblique.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time

TIME_RE = re.compile(r"Training block took:\s*([0-9.eE+-]+)")


def run_once(binary: str, csv_path: str, label_col: str, trees: int,
             extra_args: list[str], threads: int, timeout: int) -> float | None:
    cmd = [binary, "--input_mode", "csv", "--train_csv", csv_path,
           "--label_col", label_col, f"--num_trees={trees}",
           f"--num_threads={threads}"] + extra_args
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    out = proc.stdout + proc.stderr
    hits = TIME_RE.findall(out)
    if not hits:
        sys.stderr.write(f"  no timing from {os.path.basename(binary)} on {csv_path}\n")
        sys.stderr.write("  " + out[-500:].replace("\n", "\n  ") + "\n")
        return None
    return float(hits[-1])


def calibrate(binary: str, csv_path: str, label_col: str, probe_trees: int,
              target_s: float, min_trees: int, max_trees: int,
              extra_args: list[str], threads: int, timeout: int) -> tuple[int, float]:
    """Probe once, then scale trees to hit ~target_s. Returns (trees, probe_seconds)."""
    t = run_once(binary, csv_path, label_col, probe_trees, extra_args, threads, timeout)
    if t is None or t <= 0:
        return min_trees, float("nan")
    per_tree = t / probe_trees
    want = int(round(target_s / per_tree)) if per_tree > 0 else min_trees
    return max(min_trees, min(max_trees, want)), t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-dir", required=True,
                    help="dir of <dataset>/train.csv (+ meta.json)")
    ap.add_argument("--arm", action="append", required=True,
                    metavar="NAME=BINARY", help="repeatable; first arm is the baseline")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--probe-trees", type=int, default=16)
    ap.add_argument("--target-seconds", type=float, default=8.0)
    ap.add_argument("--min-trees", type=int, default=48)
    ap.add_argument("--max-trees", type=int, default=20000)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--only", default="")
    ap.add_argument("--train-args", default='--feature_split_type Oblique '
                                            '--numerical_split_type Random '
                                            '--histogram_num_bins=64')
    args = ap.parse_args()

    arms = []
    for spec in args.arm:
        name, _, path = spec.partition("=")
        if not os.path.isfile(path):
            sys.exit(f"arm binary not found: {path}")
        arms.append((name, path))
    baseline = arms[0][0]
    # shlex, not .split(): values like 'Axis Aligned' are one argv token.
    extra_args = shlex.split(args.train_args)

    names = sorted(d for d in os.listdir(args.datasets_dir)
                   if os.path.isfile(os.path.join(args.datasets_dir, d, "train.csv")))
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        names = [n for n in names if n in want]
    if not names:
        sys.exit(f"no datasets with train.csv under {args.datasets_dir}")

    print(f"{len(names)} datasets x {len(arms)} arms x {args.reps} reps, "
          f"threads={args.threads}, target={args.target_seconds}s/run")

    rows, skipped = [], []
    for i, name in enumerate(names, 1):
        ds_dir = os.path.join(args.datasets_dir, name)
        csv_path = os.path.join(ds_dir, "train.csv")
        meta = {}
        meta_path = os.path.join(ds_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)

        trees, probe_s = calibrate(arms[0][1], csv_path, args.label_col,
                                   args.probe_trees, args.target_seconds,
                                   args.min_trees, args.max_trees, extra_args,
                                   args.threads, args.timeout)
        if probe_s != probe_s:  # NaN: the probe produced no timing at all.
            print(f"[{i}/{len(names)}] {name}: ERROR probe produced no timing "
                  f"(bad flags or dataset) -- skipping", flush=True)
            skipped.append(name)
            continue
        print(f"[{i}/{len(names)}] {name}: rows={meta.get('rows','?')} "
              f"feats={meta.get('features','?')} probe={probe_s:.3f}s -> trees={trees}",
              flush=True)

        # Interleave arms: forward order on even reps, reversed on odd.
        samples = {n: [] for n, _ in arms}
        t0 = time.time()
        for rep in range(args.reps):
            order = arms if rep % 2 == 0 else list(reversed(arms))
            for arm_name, binary in order:
                t = run_once(binary, csv_path, args.label_col, trees,
                             extra_args, args.threads, args.timeout)
                if t is not None:
                    samples[arm_name].append(t)
        elapsed = time.time() - t0

        row = {"dataset": name, "rows": meta.get("rows", ""),
               "features": meta.get("features", ""), "trees": trees,
               "reps": args.reps, "threads": args.threads}
        base_med = None
        for arm_name, _ in arms:
            vals = samples[arm_name]
            med = statistics.median(vals) if vals else float("nan")
            row[f"{arm_name}_median_s"] = round(med, 6) if vals else ""
            row[f"{arm_name}_min_s"] = round(min(vals), 6) if vals else ""
            row[f"{arm_name}_n"] = len(vals)
            if arm_name == baseline:
                base_med = med
        for arm_name, _ in arms[1:]:
            med = row.get(f"{arm_name}_median_s")
            if base_med and med not in ("", None) and base_med > 0:
                row[f"{arm_name}_delta_pct"] = round(100.0 * (med - base_med) / base_med, 2)
                row[f"{arm_name}_speedup"] = round(base_med / med, 4) if med else ""
        rows.append(row)
        d = row.get(f"{arms[-1][0]}_delta_pct", "")
        print(f"      {baseline}={row.get(f'{baseline}_median_s')}s  "
              f"{arms[-1][0]}={row.get(f'{arms[-1][0]}_median_s')}s  "
              f"delta={d}%  ({elapsed:.0f}s wall)", flush=True)

        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    deltas = [r[f"{arms[-1][0]}_delta_pct"] for r in rows
              if isinstance(r.get(f"{arms[-1][0]}_delta_pct"), float)]
    if deltas:
        print(f"\n{arms[-1][0]} vs {baseline}: median delta {statistics.median(deltas):.2f}%  "
              f"mean {statistics.mean(deltas):.2f}%  "
              f"best {min(deltas):.2f}%  worst {max(deltas):.2f}%  (n={len(deltas)})")
    if skipped:
        print(f"SKIPPED {len(skipped)}: {', '.join(skipped)}")
    print(f"CSV: {args.out}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
