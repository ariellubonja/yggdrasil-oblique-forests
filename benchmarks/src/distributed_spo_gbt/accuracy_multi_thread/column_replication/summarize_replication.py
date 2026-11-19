#!/usr/bin/env python3
"""Summarize run_replication_sweep.py results: accuracy vs column replication.

Writes next to results.csv:
  summary_by_dataset.csv  dataset x replication: mean/sd test accuracy over seeds, n seeds
  summary_pooled.csv      replication: mean accuracy over datasets (of the per-dataset seed
                          means), mean gain vs replication 0, wins/ties/losses vs 0,
                          mean rank over datasets
  summary_by_dataset.md   the per-dataset table (mean acc, delta vs 0 in points)
Usage: summarize_replication.py <out_dir>
"""
import csv, os, sys
from collections import defaultdict
from statistics import mean, pstdev


def main():
    out = sys.argv[1]
    rows = [r for r in csv.DictReader(open(os.path.join(out, "results.csv"))) if r["rc"] == "0" and r["test_acc"]]
    acc = defaultdict(list)  # (dataset, r) -> [acc per seed]
    for r in rows:
        acc[(r["dataset"], float(r["replication"]))].append(float(r["test_acc"]))
    datasets = sorted({d for d, _ in acc})
    levels = sorted({l for _, l in acc})
    with open(os.path.join(out, "summary_by_dataset.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "replication", "n_seeds", "acc_mean", "acc_sd", "delta_vs_0_pts"])
        for d in datasets:
            base = mean(acc[(d, 0.0)]) if (d, 0.0) in acc else float("nan")
            for l in levels:
                if (d, l) not in acc:
                    continue
                v = acc[(d, l)]
                w.writerow([d, l, len(v), f"{mean(v):.4f}", f"{pstdev(v):.4f}", f"{100 * (mean(v) - base):+.2f}"])
    # Pooled.
    per_ds_mean = {k: mean(v) for k, v in acc.items()}
    # Complete = every level has the same number of seeds (a failed cell drops the dataset).
    full = [d for d in datasets if all((d, l) in acc for l in levels)
            and len({len(acc[(d, l)]) for l in levels}) == 1]
    eps = 1e-9
    lines = ["replication,n_datasets,acc_mean,gain_vs_0_pts,wins,ties,losses,mean_rank"]
    md = ["| dataset | F | " + " | ".join(f"{l:.1f}" for l in levels) + " |",
          "|---|---|" + "---|" * len(levels)]
    # F = numerical columns = sum of the per-worker owned counts in the manager log.
    Fs = {}
    for r in rows:
        if r.get("held_cols"):
            Fs[r["dataset"]] = sum(int(h.split("+")[0]) for h in r["held_cols"].split(";") if h)
    for l in levels:
        vals = [per_ds_mean[(d, l)] for d in full]
        gains = [per_ds_mean[(d, l)] - per_ds_mean[(d, 0.0)] for d in full]
        wins = sum(g > eps for g in gains)
        losses = sum(g < -eps for g in gains)
        ranks = []
        for d in full:
            vs = [per_ds_mean[(d, x)] for x in levels]
            v = per_ds_mean[(d, l)]
            ranks.append(1 + sum(x > v + eps for x in vs) + 0.5 * (sum(abs(x - v) <= eps for x in vs) - 1))
        lines.append(f"{l},{len(full)},{mean(vals):.4f},{100 * mean(gains):+.2f},{wins},{len(full) - wins - losses},{losses},{mean(ranks):.2f}")
    with open(os.path.join(out, "summary_pooled.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")
    for d in datasets:
        base = per_ds_mean.get((d, 0.0))
        cells = []
        for l in levels:
            if (d, l) in per_ds_mean:
                v = per_ds_mean[(d, l)]
                cells.append(f"{v:.4f}" if l == 0.0 or base is None else f"{v:.4f} ({100 * (v - base):+.1f})")
            else:
                cells.append("")
        md.append(f"| {d.split('_', 2)[-1]} | {Fs.get(d, '')} | " + " | ".join(cells) + " |")
    with open(os.path.join(out, "summary_by_dataset.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(lines))
    print(f"{len(rows)} cells, {len(datasets)} datasets ({len(full)} complete), levels {levels}")


if __name__ == "__main__":
    main()
