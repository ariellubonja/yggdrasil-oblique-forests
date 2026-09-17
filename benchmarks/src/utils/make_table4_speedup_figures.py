#!/usr/bin/env python3
"""Speedup figures for tab:overall (Table 4): one figure per baseline arm.

x = dataset (the 17 tab:overall rows, in table order), y = speedup = T(baseline)/T(arm),
one point per depth (6, 10, 16, 24, full) for each of the three vectorized arms; the five
points of one arm are joined by a thin line so the depth trend is visible. Baselines:
Exact (Highway VQSort), Random histogram scalar 64 bins, Random histogram scalar 256 bins.

Source: benchmarks/results/runtime/speedup_map_by_dataset/speedup_map.csv (+ speedup_map_rep*.csv,
median over reps) -- the same cells make_overall_depth_table.py tabulates.
Emits pgfplots (Overleaf MCP is text-only) into paper/spaa27/figures/results/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "benchmarks" / "results" / "runtime" / "speedup_map_by_dataset"
OUT = ROOT / "paper" / "spaa27" / "figures" / "results"

DATASETS = [  # (key in speedup_map.csv, tick label) -- tab:overall row order
    ("higgs_10500000", "HIGGS 10.5M$\\times$28"),
    ("SUSY", "SUSY 4.5M$\\times$18"),
    ("EPSILON", "Epsilon 400k$\\times$2000"),
    ("GiveMeSomeCredit", "GMSC 150k$\\times$10"),
    ("trunk_1000000_x_32", "Trunk 1M$\\times$32"),
    ("trunk_1000000_x_512", "Trunk 1M$\\times$512"),
    ("trunk_1000000_x_2048", "Trunk 1M$\\times$2048"),
    ("trunk_15000_x_400000", "Trunk 15k$\\times$400k"),
    ("trunk_15000_x_1600000", "Trunk 15k$\\times$1.6M"),
    ("trunk_150000_x_4096", "Trunk 150k$\\times$4096"),
    ("trunk_150000_x_40000", "Trunk 150k$\\times$40k"),
    ("trunk_150000_x_160000", "Trunk 150k$\\times$160k"),
    ("trunk_150000_x_400000", "Trunk 150k$\\times$400k"),
    ("trunk_1500000_x_4096", "Trunk 1.5M$\\times$4096"),
    ("trunk_1500000_x_16384", "Trunk 1.5M$\\times$16384"),
    ("trunk_1500000_x_40000", "Trunk 1.5M$\\times$40k"),
    ("trunk_4500000_x_4096", "Trunk 4.5M$\\times$4096"),
]

ARMS = [  # (arm, legend, color hex, cluster offset)
    ("spo_rf_rand_vec", "Vec. Random Hist.\\ (AVX2, 64 bins)", "2A78D6", -0.26),
    ("spo_rf_rand256_vec", "Vec. Random Hist.\\ (AVX-512, 256 bins)", "EB6834", 0.0),
    ("spo_rf_dyn_vec", "Vec. Dynamic Hist.\\ (AVX2, 64 bins \\& HWY)", "1BAF7A", 0.26),
]

DEPTHS = [  # (max_depth value, legend, mark, sub-offset within the arm cluster)
    (6, "depth 6", "triangle*", -0.088),  # marks: more sides = deeper tree
    (10, "depth 10", "square*", -0.044),
    (16, "depth 16", "heptagon*", 0.0),
    (24, "depth 24", "octagon*", 0.044),
    (-1, "full depth", "*", 0.088),
]


def polygon_mark(name: str, sides: int) -> str:
    """pgf has no heptagon/octagon plot marks; declare filled regular n-gons."""
    pts = [f"\\pgfqpointpolar{{{(90 + 360 * k / sides) % 360:.2f}}}{{\\pgfplotmarksize}}"
           for k in range(sides)]
    body = "\\pgfpathmoveto{%s}" % pts[0] + "".join("\\pgfpathlineto{%s}" % q for q in pts[1:])
    return f"\\pgfdeclareplotmark{{{name}}}{{{body}\\pgfpathclose\\pgfusepathqfill}}"


MARK_DEFS = [polygon_mark("heptagon*", 7), polygon_mark("octagon*", 8)]

FIGS = [  # (baseline arm, out file, label, y-axis label, caption)
    ("spo_rf_exact_hwy", "table4_speedup_vs_exact_hwy.tex", "fig:t4-speedup-exact",
     "Speedup over Exact (HWY)",
     "Speedup over exact splitting (Highway VQSort) per dataset and tree depth, the cells of "
     "Table~\\ref{tab:overall}; marks left to right = depths 6, 10, 16, 24, purity; dashed line = "
     "parity. 240 trees, 48 threads, m7i.metal-24xl, %s."),
    ("spo_rf_rand_scalar", "table4_speedup_vs_rand64.tex", "fig:t4-speedup-rand64",
     "Speedup over Random 64, scalar",
     "Speedup over random histograms, 64 bins, scalar binner; layout as Figure~\\ref{fig:t4-speedup-exact}."),
    ("spo_rf_rand256_scalar", "table4_speedup_vs_rand256.tex", "fig:t4-speedup-rand256",
     "Speedup over Random 256, scalar",
     "Speedup over random histograms, 256 bins, scalar binner; layout as Figure~\\ref{fig:t4-speedup-exact}."),
]


def load_cells() -> tuple[pd.DataFrame, int]:
    """(dataset, arm, depth) -> median train_s over reps; returns (frame, rep count)."""
    files = [RES / "speedup_map.csv"] + sorted(RES.glob("speedup_map_rep*.csv"))
    parts = []
    for k, f in enumerate(files, 1):
        d = pd.read_csv(f)
        d = d[(d.status == "OK") & (d.min_examples == 1) & (d.family == "rf")]
        parts.append(d[["dataset", "arm", "max_depth", "train_s"]].assign(rep=k))
    df = pd.concat(parts)
    g = df.groupby(["dataset", "arm", "max_depth"], as_index=False).agg(
        train_s=("train_s", "median"), n=("rep", "nunique"))
    return g, int(g.n.max()) if len(g) else 0


def speedups(cell: dict, base: str) -> list[float]:
    """All speedup values of one figure (baseline / arm), for the shared y-range."""
    return [cell[(k, base, d)] / cell[(k, arm, d)] for arm, *_ in ARMS for k, _ in DATASETS
            for d, *_ in DEPTHS if (k, base, d) in cell and (k, arm, d) in cell]


def emit(cell: dict, base: str, out: str, label: str, ylab: str, caption: str,
         ylim: tuple[float, float]) -> tuple[str, list[str]]:
    notes: list[str] = []
    ys: list[float] = []
    plots: list[str] = []
    for arm, _, _, off in ARMS:
        pts: list[tuple[float, float]] = []
        for di, (depth, _, _, sub) in enumerate(DEPTHS):
            for i, (key, _) in enumerate(DATASETS):
                num, den = cell.get((key, base, depth)), cell.get((key, arm, depth))
                if num is None or den is None or den <= 0:
                    notes.append(f"missing cell: {key} {arm} d={depth}")
                    continue
                pts.append((i + 1 + off + sub, num / den, i, di))
        ys += [p[1] for p in pts]
        ai = [a[0] for a in ARMS].index(arm)
        # Thin connector per (dataset, arm): the depth trend.
        for i in range(len(DATASETS)):
            line = sorted((p for p in pts if p[2] == i), key=lambda p: p[3])
            if len(line) > 1:
                coords = " ".join(f"({x:.3f},{y:.4f})" for x, y, _, _ in line)
                plots.append(f"\\addplot[color=sp{ai}, line width=0.35pt, opacity=0.55, mark=none,"
                             f" forget plot] coordinates {{{coords}}};")
        for di, (_, _, mark, _) in enumerate(DEPTHS):
            coords = " ".join(f"({x:.3f},{y:.4f})" for x, y, _, d in pts if d == di)
            plots.append(f"\\addplot[color=sp{ai}, mark={mark}, mark size=2.4pt, only marks,"
                         f" forget plot] coordinates {{{coords}}};")

    ymin, ymax = ylim
    ticks = ", ".join(lbl for _, lbl in DATASETS)

    L = ["% Generated by benchmarks/src/utils/make_table4_speedup_figures.py -- do not hand-edit."]
    L += [f"\\definecolor{{sp{i}}}{{HTML}}{{{c}}}" for i, (_, _, c, _) in enumerate(ARMS)]
    L += MARK_DEFS
    L += [
        "\\begin{figure*}[!t]",
        "    \\centering",
        "    \\begin{tikzpicture}",
        "    \\begin{axis}[",
        "        width=\\textwidth, height=5.2cm,",
        "        xmin=0.45, xmax=%.2f," % (len(DATASETS) + 0.55),
        "        ymin=%.2f, ymax=%.2f," % (ymin, ymax),
        "        xtick={%s}," % ",".join(str(i + 1) for i in range(len(DATASETS))),
        "        xticklabels={%s}," % ticks,
        "        xticklabel style={rotate=45, anchor=east, font=\\scriptsize},",
        "        yticklabel style={font=\\scriptsize},",
        "        ylabel={%s}," % ylab,
        "        ylabel style={font=\\small},",
        "        tick align=outside, tick pos=left,",
        "        legend columns=5, legend style={font=\\scriptsize, draw=gray!40,",
        "            at={(0.5,1.02)}, anchor=south, /tikz/every even column/.append style={column sep=6pt}},",
        "    ]",
    ]
    L += ["    " + p for p in plots]
    # Legend: row 1 = arms (color), padded to 5 cells; row 2 = depths (mark).
    for i, (_, name, _, _) in enumerate(ARMS):
        L += [f"    \\addplot[color=sp{i}, mark=none, line width=1.4pt] coordinates {{(-9,1) (-8,1)}};",
              f"    \\addlegendentry{{{name}}}"]
    L += ["    \\addlegendimage{empty legend}", "    \\addlegendentry{}"] * (5 - len(ARMS))
    for _, name, mark, _ in DEPTHS:
        L += [f"    \\addplot[color=black!65, mark={mark}, mark size=2.4pt, only marks]"
              f" coordinates {{(-9,1)}};", f"    \\addlegendentry{{{name}}}"]
    L += [
        "    \\draw[gray!70, dashed, line width=0.6pt] (axis cs:0.45,1) -- (axis cs:%.2f,1);"
        % (len(DATASETS) + 0.55),
        "    \\end{axis}",
        "    \\end{tikzpicture}",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        "\\end{figure*}",
        "",
    ]
    return "\n".join(L), notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()

    df, nrep = load_cells()
    cell = {(r.dataset, r.arm, int(r.max_depth)): r.train_s for r in df.itertuples()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # One y-range for all three figures; parity (1) always visible.
    allv = [v for base, *_ in FIGS for v in speedups(cell, base)]
    ylim = (min(0.94, min(allv) - 0.12), max(allv) + 0.30)
    runs = "single run" if nrep <= 1 else f"median of {nrep} runs"
    for base, out, label, ylab, caption in FIGS:
        tex, notes = emit(cell, base, out, label, ylab, caption.replace("%s", runs), ylim)
        (args.out_dir / out).write_text(tex)
        print(f"wrote {args.out_dir / out}")
        for n in notes:
            print("  " + n)
        # Geometric mean per arm, for the record.
        for arm, name, _, _ in ARMS:
            import math
            v = [cell[(k, base, d)] / cell[(k, arm, d)] for k, _ in DATASETS for d, *_ in DEPTHS
                 if (k, base, d) in cell and (k, arm, d) in cell]
            gm = math.exp(sum(math.log(x) for x in v) / len(v))
            print(f"  {arm:22s} geomean {gm:.2f}x  min {min(v):.2f}  max {max(v):.2f}  n={len(v)}")


if __name__ == "__main__":
    main()
