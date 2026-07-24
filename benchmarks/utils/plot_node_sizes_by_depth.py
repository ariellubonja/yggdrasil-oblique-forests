#!/usr/bin/env python3
"""One chart per depth from a node_sizes_by_depth.csv dump: each node = one column.

Input format (as emitted by oblique_cpu_depthwise_1pass.cc):

    ----DEPTH 2---- tree=0 nodes=2=11000000
    n0,1378949
    n1,9621051
    ----DEPTH 3---- tree=0 nodes=4=11000000
    ...

Writes one PNG per depth (depth_02.png, depth_03.png, ...) into an output
directory: a bar chart with one column per node, height = rows in that node,
columns sorted by size descending (--no-sort keeps the dump's node order).

Usage:
  python3 benchmarks/utils/plot_node_sizes_by_depth.py <node_sizes_by_depth.csv> \
      [-o outdir] [--tree 0] [--no-sort] [--log] [--depths 2-20]
"""
import argparse
import collections
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# dataviz reference palette (light mode).
BLUE = "#2a78d6"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def parse(path, tree=None, depths=None):
    """-> {depth: np.array of node sizes} in node order (n0, n1, ...)."""
    per_depth = collections.defaultdict(list)
    keep = False
    with open(path) as f:
        for line in f:
            if line.startswith("----DEPTH"):
                # ----DEPTH 9---- tree=0 nodes=254=10999973
                parts = line.split()
                depth = int(parts[1].strip("-"))
                cur_tree = int(
                    next(p for p in parts if p.startswith("tree=")).split("=")[1])
                keep = ((tree is None or cur_tree == tree) and
                        (depths is None or depth in depths))
                continue
            if not keep or not line.startswith("n"):
                continue
            per_depth[depth].append(int(line.split(",", 1)[1]))
    return {d: np.asarray(v, dtype=np.int64) for d, v in sorted(per_depth.items())}


def plot_depth(depth, sizes, out, sort=True, log=False):
    if sort:
        sizes = np.sort(sizes)[::-1]
    n = len(sizes)
    x = np.arange(n)

    # Wide enough to keep columns visible when there are few nodes; capped otherwise.
    width = min(max(6.0, 0.28 * n), 22.0)
    fig, ax = plt.subplots(figsize=(width, 4.5))

    if n <= 2000:
        ax.bar(x, sizes, width=0.8 if n <= 400 else 1.0, color=BLUE, linewidth=0)
    else:
        # One Rectangle per node stops rendering in reasonable time past a few
        # thousand nodes (depth 29 has ~164k). Same picture, one artist: a
        # step-filled area, still one unit-wide column per node.
        ax.fill_between(x, 0, sizes, step="mid", color=BLUE, linewidth=0)

    if log:
        ax.set_yscale("log")

    order = "sorted descending" if sort else "node order"
    ax.set_title(f"Depth {depth} — {n:,} nodes, {sizes.sum():,} rows "
                 f"(max {sizes.max():,}, median {int(np.median(sizes)):,}); {order}",
                 color=INK, loc="left")
    ax.set_xlabel("node")
    ax.set_ylabel("rows in node" + (" (log)" if log else ""))
    ax.set_xlim(-0.5, n - 0.5)
    ax.grid(axis="y", which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)
    if not log:
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else
                         (f"{v/1e3:.0f}k" if v >= 1e3 else f"{v:.0f}")))

    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def parse_depth_range(spec):
    if not spec:
        return None
    out = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("-o", "--outdir", default=None,
                    help="output directory (default: <csv>_charts/)")
    ap.add_argument("--tree", type=int, default=0,
                    help="tree index to plot; -1 pools all trees")
    ap.add_argument("--no-sort", dest="sort", action="store_false",
                    help="keep dump node order instead of sorting by size descending")
    ap.add_argument("--log", action="store_true", help="log y axis")
    ap.add_argument("--depths", default=None,
                    help="restrict to depths, e.g. '2-20' or '5,9,13'")
    args = ap.parse_args()

    tree = None if args.tree < 0 else args.tree
    per_depth = parse(args.csv, tree, parse_depth_range(args.depths))
    if not per_depth:
        raise SystemExit(f"no matching depth blocks in {args.csv}")

    outdir = args.outdir or os.path.splitext(args.csv)[0] + "_charts"
    os.makedirs(outdir, exist_ok=True)

    for depth, sizes in per_depth.items():
        out = os.path.join(outdir, f"depth_{depth:02d}.png")
        plot_depth(depth, sizes, out, sort=args.sort, log=args.log)
        print(f"depth {depth:2d}: {len(sizes):>8,} nodes -> {out}")


if __name__ == "__main__":
    main()
