#!/usr/bin/env python3
"""Compare column sharing between DW1 frontiers (full depth vs hot-nodes gate).

Input: the CSVs written by DW1_COL_SHARE_OUT in
oblique_cpu_depthwise_1pass.cc -- one row per (tree, depth) the fused kernel
ran on, with the frontier it was HANDED:

    tree,depth,nodes,rows,cols_touched,num_features,refs,pairs,share,
    shared_cols,max_nodes_col,useful,swept,eff,nodewise,amort

Because the hot gate leaves the trained trees bit-identical, a
--config=dw1_shared_rows run and a --config=dw1_sr_hot_nodes run on the same
seed/data produce the same depths over the same nodes, so the rows join
one-to-one on (tree, depth) and the deltas are attributable to the gate alone.

Columns of the per-depth table:
  nodes   nodes in the frontier the kernel saw (hot only in the gated build)
  share   mean number of nodes referencing a touched column = the sharing the
          depthwise kernel exists to exploit. share=1 means no sharing at all.
  eff     useful / swept: fraction of the colwalk's bag pass that lands a value
  amort   nodewise / swept, in FLOATS: read volume of the stock per-(node,
          projection) gather path over that of the colwalk. Floats, not cache
          lines -- the sweep is sequential (16 floats/line) while the nodewise
          gather gets ~1.9 useful floats/line at depth, so the line-level ratio
          is roughly 8x this number.

A depth present in one file and missing in the other ("-") is a depth the gate
sent entirely to the stock nodewise path (no hot node).

Usage:
  python3 benchmarks/utils/compare_dw1_colshare.py full.csv hot.csv \
      [-l full -l hot] [--tree 0]
"""
import argparse
import csv
import os


def load(path):
    """-> {(tree, depth): row dict}."""
    with open(path) as f:
        return {(int(r["tree"]), int(r["depth"])): r for r in csv.DictReader(f)}


def totals(rows):
    """Aggregate ratios over a set of rows (weighted by their raw counts)."""
    s = lambda c: sum(int(r[c]) for r in rows)
    pairs, cols, useful, swept, nodew = (
        s("pairs"), s("cols_touched"), s("useful"), s("swept"), s("nodewise"))
    return {
        "share": pairs / cols if cols else 0.0,
        "eff": useful / swept if swept else 0.0,
        "amort": nodew / swept if swept else 0.0,
        "swept": swept,
        "useful": useful,
        "nodewise": nodew,
        "rows": s("rows"),
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs=2, help="baseline CSV then gated CSV")
    p.add_argument("-l", "--label", action="append", default=None,
                   help="label per CSV (default: basename without .csv)")
    p.add_argument("--tree", type=int, default=0,
                   help="tree to show in the per-depth table (default 0); "
                        "totals always cover every tree")
    args = p.parse_args()

    labels = args.label or [os.path.splitext(os.path.basename(c))[0]
                            for c in args.csv]
    if len(labels) != 2:
        p.error("need one label per CSV")
    a, b = (load(c) for c in args.csv)
    la, lb = labels

    print(f"per-depth, tree={args.tree}   [{la} | {lb}]\n")
    print(f"{'d':>3} {'nodes':>14} {'rows':>17} {'cols':>11} "
          f"{'share':>15} {'eff':>13} {'amort':>13}")
    both = sorted(k for k in set(a) | set(b) if k[0] == args.tree)
    for k in both:
        x, y = a.get(k), b.get(k)
        pair = lambda c, w: f"{(x[c] if x else '-')}|{(y[c] if y else '-')}".rjust(w)
        num = lambda c, w: (f"{float(x[c]):.2f}" if x else "-") + "|" + \
                           (f"{float(y[c]):.2f}" if y else "-")
        print(f"{k[1]:>3} {pair('nodes', 14)} {pair('rows', 17)} "
              f"{pair('cols_touched', 11)} {num('share', 0):>15} "
              f"{num('eff', 0):>13} {num('amort', 0):>13}")

    print()
    for lab, d in ((la, a), (lb, b)):
        if not d:
            continue
        t = totals(d.values())
        print(f"{lab:>12}: share={t['share']:8.2f}  eff={t['eff']:.4f}  "
              f"amort={t['amort']:.3f}  swept={t['swept'] / 1e6:9.1f}M floats  "
              f"useful={t['useful'] / 1e6:8.1f}M  "
              f"nodewise={t['nodewise'] / 1e6:8.1f}M")

    # Fused coverage: rows the gated build still sends through the kernel.
    rows_a = sum(int(r["rows"]) for r in a.values())
    rows_b = sum(int(r["rows"]) for r in b.values())
    if rows_a:
        print(f"\nfused row coverage: {lb} keeps {rows_b / rows_a:6.1%} of "
              f"{la}'s fused rows ({rows_b:,} / {rows_a:,})")


if __name__ == "__main__":
    main()
