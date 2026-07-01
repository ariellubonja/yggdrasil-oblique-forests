#!/usr/bin/env python3
"""Collapse parallel_chrono per-function-timing CSVs into per-depth averages.

The input CSV (as produced by parallel_chrono.py) has one repeated block of
columns per thread (blocks are separated by an empty column and each start
with "tree,depth,..."), and within each thread-block the rows for several
trees are stacked back-to-back (depth resets to 0 whenever a new tree
starts). The set of metric columns differs between COARSE and FINE runs, so
the column layout is read from the header instead of being hardcoded.

For every depth this script pools all (thread, tree) rows with that depth --
i.e. it averages across threads and across the multiple trees per thread at
once -- and writes one row per depth with the number of pooled samples in a
"count" column. Note: a depth is only averaged over the trees that actually
reached it, so counts shrink at large depths.

Depth-0 rows carry only the total TreeTrain time; they are kept in the
output, and each file also gets a sanity check printed: mean TreeTrain per
tree vs. the mean of the per-depth NodeTrain sums (these should roughly
agree if the averaging is sound).

Usage:
  avg_per_depth.py <csv> [<csv> ...]
      Writes <name>_avg_per_depth.csv next to each input.

  avg_per_depth.py --compare <csvA> <csvB> [-o out.csv]
      Also writes a long-format comparison of the metric columns the two
      files share: depth, metric, <A>, <B>, ratio B/A. Default output is
      compare_<stemA>_vs_<stemB>.csv next to <csvA>.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict


def norm(name):
    """Metric names carry leading dashes as nesting markers (e.g. -NodeTrain,
    ----ApplyProjection) and the nesting differs between files; strip them so
    the same function matches across files."""
    return name.lstrip("-")


def read_blocks(path):
    """Return (metric_names, samples) where samples is {depth: [row_dict, ...]}."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))

    header_idx = next(i for i, r in enumerate(rows) if "tree" in r)
    header = rows[header_idx]
    block_starts = [i for i, v in enumerate(header) if v == "tree"]

    # Column names of one block: from "tree" up to the blank separator.
    b0 = block_starts[0]
    end = b0 + 1
    while end < len(header) and header[end] not in ("", "tree"):
        end += 1
    names = header[b0:end]
    depth_off = names.index("depth")
    metrics = [n for n in names if n not in ("tree", "depth")]
    offsets = [names.index(n) for n in metrics]

    samples = defaultdict(list)
    for row in rows[header_idx + 1:]:
        for b in block_starts:
            if b + len(names) > len(row) or row[b + depth_off] == "":
                continue
            depth = int(row[b + depth_off])
            samples[depth].append(
                [float(row[b + o]) if row[b + o] != "" else 0.0 for o in offsets])
    return metrics, samples


def averages(metrics, samples):
    """Return {depth: (count, [mean per metric])}."""
    out = {}
    for depth, vals in sorted(samples.items()):
        n = len(vals)
        out[depth] = (n, [sum(v[i] for v in vals) / n for i in range(len(metrics))])
    return out


def sanity_check(path, metrics, samples):
    normed = [norm(m) for m in metrics]
    if "TreeTrain" not in normed or "NodeTrain" not in normed or 0 not in samples:
        return
    tt, nt = normed.index("TreeTrain"), normed.index("NodeTrain")
    n_trees = len(samples[0])
    mean_tree_train = sum(v[tt] for v in samples[0]) / n_trees
    total_node_train = sum(v[nt] for vals in samples.values() for v in vals)
    print(f"  sanity ({os.path.basename(path)}): mean TreeTrain/tree = "
          f"{mean_tree_train:.3f}s, mean sum(NodeTrain)/tree = "
          f"{total_node_train / n_trees:.3f}s over {n_trees} trees")


def stem(path):
    base = os.path.basename(path)
    while base.endswith(".csv"):
        base = base[:-4]
    return base


def write_avg(path):
    metrics, samples = read_blocks(path)
    avg = averages(metrics, samples)
    out_path = os.path.join(os.path.dirname(path), stem(path) + "_avg_per_depth.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["depth", "count"] + metrics)
        for depth, (n, means) in avg.items():
            w.writerow([depth, n] + means)
    print(f"Wrote {out_path}")
    sanity_check(path, metrics, samples)
    return metrics, avg


def label(path):
    for part in path.split(os.sep):
        if part in ("FINE", "COARSE"):
            return f"{part}:{stem(path)}"
    return stem(path)


def write_compare(path_a, path_b, out_path):
    (metrics_a, avg_a), (metrics_b, avg_b) = write_avg(path_a), write_avg(path_b)
    normed_b = [norm(m) for m in metrics_b]
    common = [norm(m) for m in metrics_a if norm(m) in normed_b]
    ia = {m: [norm(x) for x in metrics_a].index(m) for m in common}
    ib = {m: normed_b.index(m) for m in common}
    la, lb = label(path_a), label(path_b)

    if out_path is None:
        out_path = os.path.join(os.path.dirname(path_a),
                                f"compare_{stem(path_a)}_vs_{stem(path_b)}.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["depth", "metric", la, lb, "ratio B/A"])
        for depth in sorted(set(avg_a) | set(avg_b)):
            for m in common:
                a = avg_a[depth][1][ia[m]] if depth in avg_a else ""
                b = avg_b[depth][1][ib[m]] if depth in avg_b else ""
                ratio = b / a if a != "" and b != "" and a != 0 else ""
                w.writerow([depth, m, a, b, ratio])
    print(f"Wrote {out_path}")

    # Per-metric totals over all depths, as a quick overall comparison.
    print(f"\n  totals over all depths ({la} vs {lb}):")
    for m in common:
        ta = sum(v[1][ia[m]] for v in avg_a.values())
        tb = sum(v[1][ib[m]] for v in avg_b.values())
        ratio = f"{tb / ta:6.3f}x" if ta else "   n/a"
        print(f"    {m:<20} {ta:12.4f} {tb:12.4f}  {ratio}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csvs", nargs="+")
    p.add_argument("--compare", action="store_true",
                   help="compare exactly two files on their common columns")
    p.add_argument("-o", "--output", default=None,
                   help="output path for the comparison CSV")
    args = p.parse_args()

    if args.compare:
        if len(args.csvs) != 2:
            sys.exit("--compare needs exactly two CSVs")
        write_compare(args.csvs[0], args.csvs[1], args.output)
    else:
        for path in args.csvs:
            write_avg(path)


if __name__ == "__main__":
    main()
