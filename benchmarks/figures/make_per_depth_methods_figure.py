#!/usr/bin/env python3
"""Per-depth train time by split method (Exact HWY / Exact std::sort / vectorized
Random histogram / scalar Random histogram), one panel per dataset.

Input: the single-tree, single-thread coarse parallel_chrono CSVs under
benchmarks/results/per_function_timing/COARSE/<cpu>/<arm dir>/<dataset>/raw/<name>.csv
(rows = one per depth; metric = the depth's summed NodeTrain seconds).

Usage: python3 benchmarks/figures/make_per_depth_methods_figure.py [--out DIR]
           [--split {Oblique,"Axis Aligned"}] [--datasets HIGGS,SUSY,Epsilon]
Writes per_depth_methods[_aa].{pdf,png} and the matching .csv (tidy long form).
--split="Axis Aligned" reads the "Axis Aligned | ..." arm dirs (same four arms,
axis-aligned RF, num_candidate_attributes = sqrt(F)) and defaults to HIGGS only.
--ensemble=Boosting reads the "Boosting | Oblique | ..." dirs (SPO-GBT, 5 trees, depth 50,
1 thread) and plots the per-depth NodeTrain averaged over the trees in the run
(the RF panels are single-tree runs, so no averaging happens there).
"""
import argparse, sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "benchmarks/results/per_function_timing/COARSE/Intel(R) Xeon(R) Platinum 8488C"

# (label, arm dir suffix, raw file stem per dataset) -- fixed series order = fixed hue order.
# The arm dir is "<split> | <suffix>", e.g. "Oblique | Exact", "Axis Aligned | Random | Scalar".
ARMS = [
    ("Exact (HWY VQSort)",         "Exact",           {"*": "hwy"}),
    ("Random Hist. (vectorized)",  "Random",          {"*": "vectorized-64"}),
    ("Exact (std::sort)",          "Exact",           {"*": "stdSort"}),
    ("Random Hist. (scalar)",      "Random | Scalar", {"*": "scalar"}),
]
# Per-split overrides of the raw stem (the oblique HIGGS scalar run kept the helper's default name).
STEM_OVERRIDES = {"Oblique": {("Random Hist. (scalar)", "HIGGS_with_header"): "-1Depth-1Threads"}}
DATASET_DIRS = {"HIGGS": "HIGGS_with_header", "SUSY": "SUSY_with_header",
                "Epsilon": "epsilon_normalized_train"}
DEFAULT_DATASETS = {"Oblique": "HIGGS,SUSY,Epsilon", "Axis Aligned": "HIGGS"}
STEM_OUT = {"Oblique": "per_depth_methods", "Axis Aligned": "per_depth_methods_aa"}
STEM_OUT_GBT = "per_depth_methods_gbt"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # dataviz reference palette, slots 1-4
STYLES = ["-", "-", "--", "--"]


def load(path: Path) -> pd.DataFrame:
    """First thread block of a parallel_chrono CSV -> depth, nodes, samples, NodeTrain.

    Multi-tree runs (GBT: 5 trees in one process) are averaged per depth over the trees,
    so the y axis is always "seconds per depth for one tree"."""
    hdr = pd.read_csv(path, header=None, nrows=2)
    cols = hdr.iloc[1].tolist()
    ncol = next(i for i, c in enumerate(cols) if isinstance(c, float))  # first blank separator
    df = pd.read_csv(path, header=1, usecols=range(ncol))
    df = df[df["depth"] > 0]
    d = pd.DataFrame({"depth": df["depth"].astype(int), "nodes": df["nodes"].astype(int),
                      "samples": df["Active Samples"].astype(int),
                      "node_train_s": df["-NodeTrain"].astype(float)})
    n_trees = df["tree"].nunique()
    if n_trees > 1:
        d = d.groupby("depth", as_index=False).mean(numeric_only=True)
        d["nodes"] = d["nodes"].round().astype(int); d["samples"] = d["samples"].round().astype(int)
        d.insert(1, "trees_averaged", n_trees)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "benchmarks/figures"))
    ap.add_argument("--split", choices=list(STEM_OUT), default="Oblique")
    ap.add_argument("--datasets", default=None, help="comma list of " + ",".join(DATASET_DIRS))
    ap.add_argument("--ensemble", choices=["Bagging", "Boosting"], default="Bagging")
    ap.add_argument("--max_depth", type=int, default=None, help="crop the x axis (e.g. 10 for GBT)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    gbt = a.ensemble == "Boosting"
    datasets = (a.datasets or DEFAULT_DATASETS["Oblique" if gbt else a.split]).split(",")
    stem_out = STEM_OUT_GBT if gbt else STEM_OUT[a.split]
    arm_prefix = f"Boosting | {a.split}" if gbt else a.split

    tidy, panels = [], []
    for ds_label in datasets:
        ds_dir = DATASET_DIRS[ds_label]
        series = []
        for label, arm_suffix, stems in ARMS:
            overrides = {} if gbt else STEM_OVERRIDES.get(a.split, {})
            stem = overrides.get((label, ds_dir), stems.get(ds_dir, stems["*"]))
            p = BASE / f"{arm_prefix} | {arm_suffix}" / ds_dir / "raw" / f"{stem}.csv"
            if not p.exists():
                print(f"  missing: {p.relative_to(ROOT)}", file=sys.stderr); continue
            d = load(p)
            if a.max_depth is not None:
                d = d[d["depth"] <= a.max_depth]
            d.insert(0, "method", label); d.insert(0, "dataset", ds_label)
            tidy.append(d); series.append((label, d))
        if series:
            panels.append((ds_label, series))
    if not panels:
        sys.exit("no data found")
    pd.concat(tidy).to_csv(out / f"{stem_out}.csv", index=False)

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b"})
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 2.6), squeeze=False)
    for ax, (ds_label, series) in zip(axes[0], panels):
        for label, d in series:
            k = [x[0] for x in ARMS].index(label)
            ax.fill_between(d["depth"], 0, d["node_train_s"], color=COLORS[k], alpha=0.12, lw=0)
            ax.plot(d["depth"], d["node_train_s"], STYLES[k], color=COLORS[k], lw=1.6, label=label)
        title = ds_label if a.split == "Oblique" else f"{ds_label}, axis-aligned RF"
        if gbt:
            title = f"{ds_label}, SPO-GBT"
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("Tree depth"); ax.set_ylim(bottom=0); ax.set_xlim(left=1)
        ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
    axes[0][0].set_ylabel("Train time per depth (s)")
    h, l = axes[0][0].get_legend_handles_labels()
    ncol = 4 if len(panels) > 1 else 2
    fig.legend(h, l, loc="upper center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, 1.04 if len(panels) > 1 else 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.93 if len(panels) > 1 else 0.84))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem_out}.{ext}", dpi=200, bbox_inches="tight")
    print("wrote", out / (stem_out + ".{pdf,png,csv}"))


if __name__ == "__main__":
    main()
