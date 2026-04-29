#!/usr/bin/env python3
"""Read Phase A oblique sweep CSV and recommend the (density, num_proj_exp)
config that gives oblique its best Pareto position.

Decision rule: for each (density, exp) config, compute the geometric mean
across datasets of `(oblique_µs / axis_baseline_µs at matched accuracy)`
on the apples-to-apples Generic engine. Lower = better. The axis baseline
comes from sweep_v1.csv (SUSY) and sweep_v2.csv (bioresponse).

Tie-breaker: prefer the config with higher max accuracy reached.
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def load(csv_path):
    rows = []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            try:
                r["max_depth"] = int(r["max_depth"])
                r["accuracy"] = float(r["accuracy"]) if r["accuracy"] else None
                r["generic_us_per_ex"] = (
                    float(r["generic_us_per_ex"])
                    if r["generic_us_per_ex"] else None)
                r["density"] = float(r["density"])
                r["num_proj_exp"] = float(r["num_proj_exp"])
            except (ValueError, KeyError):
                continue
            rows.append(r)
    return rows


def axis_baseline_for_acc(axis_rows, target_acc):
    """Return min Generic-engine µs of any axis-aligned config with
    accuracy >= target_acc (i.e. the cheapest axis way to match it)."""
    matches = [r for r in axis_rows
               if r["accuracy"] is not None
               and r["generic_us_per_ex"] is not None
               and r["accuracy"] >= target_acc - 0.001]
    if not matches:
        return None
    return min(r["generic_us_per_ex"] for r in matches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase_a_csv", required=True)
    ap.add_argument("--sweep_v1_csv",
                    default=str(Path(__file__).resolve().parents[2]
                                / "benchmarks/results/oblique_gbt_inference"
                                / "sweep_v1.csv"))
    ap.add_argument("--sweep_v2_csv",
                    default=str(Path(__file__).resolve().parents[2]
                                / "benchmarks/results/oblique_gbt_inference"
                                / "sweep_v2.csv"))
    args = ap.parse_args()

    phase_a = load(Path(args.phase_a_csv))
    if not phase_a:
        print("Phase A CSV empty.")
        return

    # Build axis baseline indexed by dataset.
    axis_by_ds = defaultdict(list)
    for csvp in (args.sweep_v1_csv, args.sweep_v2_csv):
        for r in load(Path(csvp)):
            if r["split_type"] == "Axis Aligned":
                axis_by_ds[r["dataset"]].append(r)

    # Group Phase A rows by (density, exp).
    by_cfg = defaultdict(list)
    for r in phase_a:
        if r["accuracy"] is None or r["generic_us_per_ex"] is None:
            continue
        by_cfg[(r["density"], r["num_proj_exp"])].append(r)

    # For each config, compute per-dataset geometric mean of µs ratio
    # (oblique µs / axis cheapest-matching-acc µs), at each oblique depth.
    print("\nConfig comparison (apples-to-apples Generic engine):")
    print(f"{'density':>8} {'exp':>5} {'avg log-ratio':>14} "
          f"{'avg max acc':>12} {'depths covered':>16}")
    scored = []
    for (dens, exp), sub in sorted(by_cfg.items()):
        log_ratios = []
        max_accs = []
        per_ds_max_acc = defaultdict(float)
        for r in sub:
            ax = axis_by_ds.get(r["dataset"], [])
            base = axis_baseline_for_acc(ax, r["accuracy"])
            if base and base > 0:
                log_ratios.append(math.log(r["generic_us_per_ex"] / base))
            per_ds_max_acc[r["dataset"]] = max(
                per_ds_max_acc[r["dataset"]], r["accuracy"])
        avg_log = sum(log_ratios) / len(log_ratios) if log_ratios else None
        avg_max_acc = (sum(per_ds_max_acc.values()) / len(per_ds_max_acc)
                       if per_ds_max_acc else None)
        scored.append((dens, exp, avg_log, avg_max_acc, len(sub)))
        print(f"{dens:>8.1f} {exp:>5.1f} "
              f"{(avg_log if avg_log is not None else 'NA'):>14} "
              f"{(avg_max_acc if avg_max_acc is not None else 'NA'):>12} "
              f"{len(sub):>16}")

    # Pick lowest log-ratio (= oblique closest to / beating axis).
    feasible = [s for s in scored if s[2] is not None]
    if not feasible:
        print("\nNo config has any matched-accuracy axis baseline.")
        return
    winner = min(feasible, key=lambda s: s[2])
    print(f"\nWINNER: density={winner[0]} exp={winner[1]}  "
          f"avg_log_ratio={winner[2]:.3f} (=> "
          f"{math.exp(winner[2]):.2f}× slower than axis on average)")
    print(f"  Use these flags for Phase B:")
    print(f"  --density={winner[0]} --num_proj_exp={winner[1]}")


if __name__ == "__main__":
    main()
