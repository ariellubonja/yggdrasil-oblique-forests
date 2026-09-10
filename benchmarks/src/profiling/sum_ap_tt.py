#!/usr/bin/env python3
"""Per-tree ApplyProjection sums and TreeTrain from a parallel_chrono CSV.

Usage: sum_ap_tt.py <csv> [<csv> ...]
Prints, per file: median over trees of (sum_depth ApplyProjection) and of
TreeTrain (TreeTrain is recorded once per tree at depth -1 / max row).
"""
import sys
import pandas as pd


def analyze(path):
    # Header is on line 2 (line 1 is the thread banner).
    df = pd.read_csv(path, skiprows=1)
    df = df[pd.to_numeric(df["tree"], errors="coerce").notna()].copy()
    for c in ("tree", "ApplyProjection", "TreeTrain"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    per_tree = df.groupby("tree").agg(
        ap=("ApplyProjection", "sum"), tt=("TreeTrain", "sum"))
    return per_tree


def main():
    for path in sys.argv[1:]:
        pt = analyze(path)
        name = path.split("|")[-2].strip() if "|" in path else path
        print(f"{name}: AP median {pt['ap'].median():.2f} s "
              f"(per-tree {', '.join(f'{v:.1f}' for v in pt['ap'])}); "
              f"TT median {pt['tt'].median():.2f} s")


if __name__ == "__main__":
    main()
