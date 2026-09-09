#!/usr/bin/env python3
"""Per-depth train time by split method (Exact HWY / Exact std::sort / vectorized
Random histogram / scalar Random histogram), one panel per dataset.

Input: the single-tree, single-thread coarse parallel_chrono CSVs under
benchmarks/results/per_function_timing/COARSE/<cpu>/<arm dir>/<dataset>/raw/<name>.csv
(rows = one per depth; metric = the depth's summed NodeTrain seconds).

Usage: python3 benchmarks/figures/make_per_depth_methods_figure.py [--out DIR]
Writes per_depth_methods.{pdf,png} and per_depth_methods.csv (tidy long form).
"""
import argparse, sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "benchmarks/results/per_function_timing/COARSE/Intel(R) Xeon(R) Platinum 8488C"

# (label, arm dir, raw file stem per dataset) -- fixed series order = fixed hue order.
ARMS = [
    ("Exact (HWY VQSort)",         "Oblique | Exact",           {"*": "hwy"}),
    ("Random Hist. (vectorized)",  "Oblique | Random",          {"*": "vectorized-64"}),
    ("Exact (std::sort)",          "Oblique | Exact",           {"*": "stdSort"}),
    ("Random Hist. (scalar)",      "Oblique | Random | Scalar", {"HIGGS_with_header": "-1Depth-1Threads", "*": "scalar"}),
]
DATASETS = [("HIGGS", "HIGGS_with_header"), ("SUSY", "SUSY_with_header"),
            ("Epsilon", "epsilon_normalized_train")]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # dataviz reference palette, slots 1-4
STYLES = ["-", "-", "--", "--"]


def load(path: Path) -> pd.DataFrame:
    """First thread block of a parallel_chrono CSV -> depth, nodes, samples, NodeTrain."""
    hdr = pd.read_csv(path, header=None, nrows=2)
    cols = hdr.iloc[1].tolist()
    ncol = next(i for i, c in enumerate(cols) if isinstance(c, float))  # first blank separator
    df = pd.read_csv(path, header=1, usecols=range(ncol))
    df = df[df["depth"] > 0]
    return pd.DataFrame({"depth": df["depth"].astype(int), "nodes": df["nodes"].astype(int),
                         "samples": df["Active Samples"].astype(int),
                         "node_train_s": df["-NodeTrain"].astype(float)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "benchmarks/figures"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    tidy, panels = [], []
    for ds_label, ds_dir in DATASETS:
        series = []
        for label, arm_dir, stems in ARMS:
            stem = stems.get(ds_dir, stems["*"])
            p = BASE / arm_dir / ds_dir / "raw" / f"{stem}.csv"
            if not p.exists():
                print(f"  missing: {p.relative_to(ROOT)}", file=sys.stderr); continue
            d = load(p); d.insert(0, "method", label); d.insert(0, "dataset", ds_label)
            tidy.append(d); series.append((label, d))
        if series:
            panels.append((ds_label, series))
    if not panels:
        sys.exit("no data found")
    pd.concat(tidy).to_csv(out / "per_depth_methods.csv", index=False)

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b"})
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 2.6), squeeze=False)
    for ax, (ds_label, series) in zip(axes[0], panels):
        for label, d in series:
            k = [x[0] for x in ARMS].index(label)
            ax.fill_between(d["depth"], 0, d["node_train_s"], color=COLORS[k], alpha=0.12, lw=0)
            ax.plot(d["depth"], d["node_train_s"], STYLES[k], color=COLORS[k], lw=1.6, label=label)
        ax.set_title(ds_label, fontsize=10, loc="left")
        ax.set_xlabel("Tree depth"); ax.set_ylim(bottom=0); ax.set_xlim(left=1)
        ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
    axes[0][0].set_ylabel("Train time per depth (s)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"per_depth_methods.{ext}", dpi=200, bbox_inches="tight")
    print("wrote", out / "per_depth_methods.{pdf,png,csv}")


if __name__ == "__main__":
    main()
