#!/usr/bin/env python3
"""fig:overall (+ fig:overall-depth24) -- end-to-end training time normalized to Exact (HWY), grouped bars.

Variants (--variant, default: all):
  grid     fig:overall, figure*, 2x2 panels: SPO-GBT depth 6; SPO-RF depth 6, depth 16, full depth.
           Dynamic bars are omitted at depth 6 (GBT: no dynamic arm, directive 2026-09-24; RF: user
           2026-09-25) -> e2e_normalized_grid.tex
  depth24  fig:overall-depth24, single-column figure, one panel: SPO-RF depth 24 -> e2e_normalized_depth24.tex
x = the datasets where the dynamic exact fallback gains most over the plain vectorized histogram at
full depth (ranked 2026-09-25, two trunk shapes at most); y = T(arm) / T(Exact HWY). Bars per dataset
are grouped Exact-Rand64-Rand256 | Vec64-Vec256 | DynVec64-DynVec256 with gaps between blocks; the
y-range is fixed to 0-1.5 and taller bars are cut with a white break mark.
Source cells: the same speedup_map.csv (+ reps, median) that make_overall_depth_table.py tabulates.
Emits pgfplots (Overleaf MCP is text-only) to paper/spaa27/figures/results/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks" / "src" / "spo_vs_gbt"))
import make_overall_depth_table as tbl  # noqa: E402  (shared cell loader)

OUT_DIR = ROOT / "paper" / "spaa27" / "figures" / "results"

DATASETS = [  # (key in speedup_map.csv, tick label)
    ("EPSILON", "Epsilon"),
    ("JANE_STREET", "Jane Street"),
    ("SUSY", "SUSY"),
    ("higgs_10500000", "HIGGS"),
    ("YOUTUBE8M", "YouTube-8M"),
    ("trunk_150000_x_4096", "Trunk 150k$\\times$4096"),
    ("trunk_15000_x_400000", "Trunk 15k$\\times$400k"),
]
# (arm suffix, legend, color hex, slot). Slots 0-2 = baselines, 3-4 = vectorized, 5-6 = dynamic.
# Palette = the previous fig:overall (Google colors) plus shades: 64-bin = lighter, 256-bin = darker;
# Dynamic 64 = gold, Dynamic 256 = red (user, 2026-09-25).
ARMS = [
    ("exact_hwy", "Exact (HWY)", "4285F4", 0),
    ("rand_scalar", "Random 64 (scalar)", "34A853", 1),
    ("rand256_scalar", "Random 256 (scalar)", "0B6B3A", 2),
    ("rand_vec", "Vec.\\ Random 64 (AVX2)", "B57BFF", 3),
    ("rand256_vec", "Vec.\\ Random 256 (AVX-512)", "6A1B9A", 4),
    ("dyn_vec", "Vec.\\ Dynamic 64 (AVX2 \\& HWY)", "FBBC04", 5),
    ("dyn256_vec", "Vec.\\ Dynamic 256 (AVX-512 \\& HWY)", "EA4335", 6),
]
BAR_W = 0.1  # x units; cluster = 7 bars + two gaps of 0.5 bar (after slots 2 and 4)
GAP = 0.5
YMAX = 1.5  # fixed y-range in every panel; taller bars are cut off and marked with a white break

# Mirrors the author's Overleaf caption / Description (2026-09-25); edit here, not on Overleaf.
CAPTION = ("End-to-end training time on select datasets, normalized to the performance of "
           "SO-YDF with exact splits. Panels: SPO-GBT at depth 6; SPO-RF at depth 6, depth 16 "
           "and full depth (purity). Blocks of each cluster, left to right: Exact and scalar random "
           "histograms; our vectorized histograms; our dynamic-vectorized histograms (not run at "
           "depth 6). Bars above 1.5 are cut off (white break mark).")
DESCRIPTION = ("End-to-end evaluation reveals the benefit of our methods - up to 35\\% reduction "
               "in runtime with only Dynamic, and up to 50\\% with Vectorized Dynamic. It also "
               "reveals the similarity in performance of RF Exact to SO Exact.")
# Panel = (family, depth, title, include dynamic arms)
VARIANTS = {
    "grid": dict(
        out="e2e_normalized_grid.tex", label="fig:overall", caption=CAPTION, description=DESCRIPTION,
        wide=True, legend_name="e2elegendgrid",
        panels=[("gbt", 6, "SPO-GBT, depth 6", False), ("rf", 6, "SPO-RF, depth 6", False),
                ("rf", 16, "SPO-RF, depth 16", True), ("rf", -1, "SPO-RF, full depth (purity)", True)]),
    "depth24": dict(
        out="e2e_normalized_depth24.tex", label="fig:overall-depth24",
        caption="SPO-RF at depth 24; datasets, bars and axes as in Figure~\\ref{fig:overall}.",
        description="", wide=False, legend_name="e2elegendd24",
        panels=[("rf", 24, "SPO-RF, depth 24", True)]),
}


def slot_shift(slot: int) -> float:
    """Bar centre offset in x units: blocks 0-2 | 3-4 | 5-6, centred on the dataset tick."""
    gaps_before = (slot >= 3) + (slot >= 5)
    return (slot - 3.5 + GAP * gaps_before) * BAR_W


def panel_plots(cells: dict, fam: str, depth: int, with_dyn: bool, legend: bool,
                notes: list[str]) -> list[str]:
    n = len(DATASETS)
    plots: list[str] = []
    cut: list[float] = []  # x centres of bars taller than YMAX
    for suffix, name, _, slot in ARMS:
        if suffix.startswith("dyn") and not with_dyn:
            continue
        arm = f"spo_{fam}_{suffix}"
        pts = []
        for i, (key, _) in enumerate(DATASETS):
            v = cells.get((fam, key, arm, depth))
            base = cells.get((fam, key, f"spo_{fam}_exact_hwy", depth))
            if v is None or base is None:
                notes.append(f"missing: {fam} {key} {arm} d={depth}")
                continue
            pts.append((i + 1, v / base))
            if v / base > YMAX:
                cut.append(i + 1 + slot_shift(slot))
        if not pts:
            continue
        coords = " ".join(f"({x},{y:.4f})" for x, y in pts)
        plots.append(f"\\addplot[bar{slot}, /pgf/bar shift={slot_shift(slot):.4f}] coordinates {{{coords}}};")
        if legend:
            plots.append(f"\\addlegendentry{{{name}}}")
    plots.append(f"\\draw[gray!70, dashed, line width=0.5pt] (axis cs:0.45,1) -- (axis cs:{n + 0.55},1);")
    # Cut-off marker: two white slashes across the bar top, like an axis break.
    for xc in cut:
        x0, x1 = xc - BAR_W * 0.62, xc + BAR_W * 0.62
        for y0 in (YMAX - 0.12, YMAX - 0.06):
            plots.append(f"\\draw[white, line width=1.1pt] (axis cs:{x0:.4f},{y0:.3f}) -- (axis cs:{x1:.4f},{y0 + 0.05:.3f});")
    return plots


def emit(cells: dict, v: dict) -> tuple[str, list[str]]:
    n = len(DATASETS)
    notes: list[str] = []
    wide = v["wide"]
    ncol = 2 if wide else 1
    nrow = (len(v["panels"]) + ncol - 1) // ncol
    L = ["% Generated by benchmarks/src/utils/make_e2e_grid_figure.py -- do not hand-edit.",
         "% Needs \\usepackage{pgfplots}\\pgfplotsset{compat=1.18}\\usepgfplotslibrary{groupplots}."]
    L += [f"\\definecolor{{e2ec{s}}}{{HTML}}{{{c}}}" for _, _, c, s in ARMS]
    L += [f"\\pgfplotsset{{bar{s}/.style={{ybar, fill=e2ec{s}, draw=e2ec{s}!70!black, line width=0.2pt}}}}"
          for _, _, _, s in ARMS]
    L += [
        "\\begin{figure*}[!t]" if wide else "\\begin{figure}[t]",
        "    \\centering",
        "    \\begin{tikzpicture}",
        "    \\begin{groupplot}[",
        f"        group style={{group size={ncol} by {nrow}, horizontal sep=0.75cm, vertical sep=1.05cm,",
        "            xticklabels at=edge bottom},",
        "        width=0.53\\textwidth, height=4.5cm," if wide else "        width=\\linewidth, height=4.6cm,",
        f"        /pgf/bar width={BAR_W}, ybar, ybar legend,",
        f"        xmin=0.45, xmax={n + 0.55}, ymin=0, ymax={YMAX}, ytick={{0,0.5,1,1.5}},",
        f"        xtick={{{','.join(str(i + 1) for i in range(n))}}},",
        f"        xticklabels={{{', '.join(lbl for _, lbl in DATASETS)}}},",
        "        xticklabel style={rotate=28, anchor=north east, font=\\scriptsize, yshift=-1pt},",
        "        yticklabel style={font=\\scriptsize}, title style={font=\\small, yshift=-4pt},",
        "" if wide else "        ylabel={Runtime Normalized to Exact}, ylabel style={font=\\small},",
        "        tick align=outside, tick pos=left, axis line style={black!50},",
        f"        legend columns={3 if wide else 2}, legend cell align=left,",
        "        legend style={font=\\scriptsize, draw=gray!40, /tikz/every even column/.append style={column sep=5pt}},",
        "        legend image code/.code={\\draw[#1] (0cm,-0.08cm) rectangle (0.25cm,0.14cm);},",
        "    ]",
    ]
    L = [x for x in L if x != ""]
    for pi, (fam, depth, title, with_dyn) in enumerate(v["panels"]):
        last = pi == len(v["panels"]) - 1  # legend built from the last panel (all arms)
        opts = f"title={{{title}}}" + (f", legend to name={v['legend_name']}" if last else "")
        L.append(f"    \\nextgroupplot[{opts}]")
        L += ["    " + s for s in panel_plots(cells, fam, depth, with_dyn, last, notes)]
    L.append("    \\end{groupplot}")
    if wide:  # one y label spanning both rows (longer than a single panel is tall)
        L += ["    \\path (group c1r1.outer west) -- (group c1r2.outer west)",
              "        node[midway, rotate=90, anchor=south, font=\\small] {Runtime Normalized to Exact};"]
    L += [
        "    \\end{tikzpicture}",
        "",
        "    \\vspace{2pt}",
        f"    \\ref{{{v['legend_name']}}}",
        f"    \\caption{{{v['caption']}}}",
        *([f"    \\Description{{{v['description']}}}"] if v["description"] else []),
        f"    \\label{{{v['label']}}}",
        "\\end{figure*}" if wide else "\\end{figure}",
        "",
    ]
    return "\n".join(L), notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS) + ["all"], default="all")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    cells = {}
    for fam in ("rf", "gbt"):
        df, _ = tbl.load_cells(family=fam)
        cells.update({(fam, r.dataset, r.arm, int(r.max_depth)): r.train_s for r in df.itertuples()})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in (sorted(VARIANTS) if args.variant == "all" else [args.variant]):
        tex, notes = emit(cells, VARIANTS[name])
        out = args.out_dir / VARIANTS[name]["out"]
        out.write_text(tex)
        print(f"wrote {out}")
        for m in notes:
            print("  " + m)


if __name__ == "__main__":
    main()
