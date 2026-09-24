#!/usr/bin/env python3
"""fig:dynamic-breakeven -- SPO-RF train time relative to all-exact vs the dynamic switch threshold.

y = T(threshold) / T(all nodes exact, Highway VQSort), so y < 1 is where Dynamic beats exact and
each curve's minimum is the breakeven. Main figure = AVX2 64-bin vs AVX-512 256-bin on the two
datasets both sweeps cover, at the thresholds both scanned; appendix = all datasets with the
vectorized binners, and the scalar 64-bin binner. Sweeps: benchmarks/results/runtime/dynamic_histogram_breakeven/.
Captions live in main.tex and must not carry machine specs (CLAUDE.md directive).
All-exact references (same dataset files as the sweeps) come from REFS below.
Emits pgfplots (Overleaf MCP is text-only) into paper/spaa27/figures/results/.
"""
from __future__ import annotations

import argparse
import csv
import io
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "benchmarks" / "results" / "runtime"
SWEEP = RES / "dynamic_histogram_breakeven"

# All-exact (Highway VQSort) train_s, 240 trees, 48 threads, m7i 8488C, full depth, on the SAME
# dataset files the sweeps use. (file, dataset-key-in-file, note). SUSY has no matched reference:
# the speedup map ran the 4.5M-row train split, the sweep the 5M-row SUSY_with_header.
REFS = {
    "HIGGS_with_header": (RES / "ablation_vectorized_dynamic" / "dfs_exact_hwy.csv", "ablation"),
    "trunk_1500000_x_4096": (RES / "ablation_vectorized_dynamic" / "dfs_exact_hwy.csv", "ablation"),
    "trunk_150000_x_40000": (RES / "ablation_vectorized_dynamic" / "dfs_exact_hwy.csv", "ablation"),
    "trunk_15000_x_400000": (RES / "ablation_vectorized_dynamic" / "dfs_exact_hwy.csv", "ablation"),
    "epsilon_normalized_train": (RES / "speedup_map_by_dataset" / "large_results.csv", "large:EPSILON"),
    # Same boot as the natural sweeps (commit 38600462, 2026-09-23), all nodes exact.
    "SUSY_with_header": (SWEEP / "dynamic_threshold_sweep_natural_susy_exact.csv", "sweep"),
    "youtube8m_video_train": (SWEEP / "dynamic_threshold_sweep_natural_youtube8m_exact.csv", "sweep"),
}

# Fixed categorical slots (dataviz palette, validated); marker = secondary encoding.
DATASETS = [  # (sweep key, legend, color hex, mark)
    ("HIGGS_with_header", "HIGGS 11M$\\times$28", "2A78D6", "*"),
    ("SUSY_with_header", "SUSY 5M$\\times$18", "EB6834", "square*"),
    ("epsilon_normalized_train", "Epsilon 400k$\\times$2000", "1BAF7A", "triangle*"),
    ("trunk_1500000_x_4096", "Trunk 1.5M$\\times$4096", "EDA100", "diamond*"),
    ("trunk_150000_x_40000", "Trunk 150k$\\times$40k", "E87BA4", "pentagon*"),
    ("trunk_15000_x_400000", "Trunk 15k$\\times$400k", "008300", "o"),
    ("youtube8m_video_train", "YouTube-8M 3.9M$\\times$1152", "7A5195", "star"),
]

PANELS = {
    # Main text: the two datasets swept under both vectorized binners, common thresholds only.
    "isa": {
        "out": "dynamic_breakeven.tex",
        "series": [  # (glob, line style, legend)
            ("dynamic_threshold_sweep_hwysort_avx512_random256_thr*.csv", "solid", "256 bins, AVX-512"),
            ("dynamic_threshold_sweep_hwysort_vectorized_random_3runs_2datasets.csv", "dashed", "64 bins, AVX2"),
        ],
        "datasets": {"HIGGS_with_header", "trunk_1500000_x_4096"},
        # Google palette (fig:decompose), one color per (binner, dataset) curve; legend per curve.
        "colors": {("solid", "HIGGS_with_header"): "4285F4", ("solid", "trunk_1500000_x_4096"): "EA4335",
                   ("dashed", "HIGGS_with_header"): "FBBC05", ("dashed", "trunk_1500000_x_4096"): "34A853"},
        "common_thresholds": True,
        "xmin": 80, "xmax": 6500, "xtick": "{100,1000,5000}", "band": False, "mark_min": True,
        "legend_all": True,
        "ref_label": "anchor=south east] at (axis cs:6500,1)",
    },
    # Every natural dataset, both vectorized binners, from the 2026-09-22/23 sweeps
    # (commit cc1277fe; 26-point ladder 100..10000, 1 run), linear axis. Trunks hidden, not deleted.
    "all": {
        "out": "dynamic_breakeven_all.tex",
        "series": [
            ("dynamic_threshold_sweep_natural_*_avx512_256.csv", "solid", "256 bins, AVX-512"),
            ("dynamic_threshold_sweep_natural_*_avx2_64.csv", "dashed", "64 bins, AVX2"),
        ],
        "datasets": {"HIGGS_with_header", "SUSY_with_header", "epsilon_normalized_train", "youtube8m_video_train"},
        # Google primaries, one per dataset; no point marks; AVX2 = short dashes.
        "palette": {"HIGGS_with_header": "4285F4", "SUSY_with_header": "EA4335",
                    "epsilon_normalized_train": "FBBC05", "youtube8m_video_train": "34A853"},
        "marks": False, "dash": "dash pattern=on 2pt off 1.5pt",
        "xmode": "normal", "xmin": 0, "xmax": 10300, "xtick": "{0,2000,4000,6000,8000,10000}",
        "extra_axis": "scaled x ticks=false, ", "band": True, "mark_min": False,
        "ref_label": None, "grid": False, "ylabel": "Train Time over Exact", "xlabel": "Dynamic Switch Threshold",
        # Darker band for the AVX2 minima: fixed 200..400 (user choice 2026-09-23; computed argmin range is 300..500).
        "band_dashed": (200, 400),
        # Legend built after the plots, row-wise over 3 columns: dataset keys or "series:<style>".
        "legend_order": ["HIGGS_with_header", "SUSY_with_header", "series:dashed",
                         "epsilon_normalized_train", "youtube8m_video_train", "series:solid"],
        "legend_names": {"HIGGS_with_header": "HIGGS", "SUSY_with_header": "SUSY",
                         "epsilon_normalized_train": "Epsilon", "youtube8m_video_train": "YouTube-8M",
                         "series:dashed": "AVX2", "series:solid": "AVX-512"},
    },
    "scalar": {
        "out": "dynamic_breakeven_scalar.tex",
        "series": [
            ("dynamic_threshold_sweep_hwysort_scalar_thr*.csv", "solid", "64 bins, scalar"),
        ],
        "xmin": 800, "xmax": 28000, "xtick": "{1000,10000}", "band": False, "mark_min": False,
        "ref_label": "anchor=north west] at (axis cs:800,1)",
    },
}
# Zoom of "all" on the breakeven region: thresholds 0..2000, y capped at 0.95 (reference line off-range).
PANELS["all_zoom"] = dict(PANELS["all"], out="dynamic_breakeven_all_zoom.tex", xmin=0, xmax=2000,
                          xtick="{0,500,1000,1500,2000}", ymax=0.95)


def load_sweep(pattern: str) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for f in sorted(SWEEP.glob(pattern)):
        body = f.read_text().split("====================\n", 1)[1]
        for r in csv.DictReader(io.StringIO(body)):
            out.setdefault(r["dataset"], {})[int(r["dynamic_split_threshold"])] = float(r["median_s"])
    return out


def load_refs() -> dict[str, float]:
    refs: dict[str, float] = {}
    for key, (path, how) in REFS.items():
        text = path.read_text()
        if how in ("ablation", "sweep"):
            body = text.split("====================\n", 1)[1]
            refs[key] = next(float(r["median_s"]) for r in csv.DictReader(io.StringIO(body)) if r["dataset"] == key)
        else:
            name = how.split(":")[1]
            vals = [float(r["train_s"]) for r in csv.DictReader(io.StringIO(text))
                    if r["dataset"] == name and r["method"] == "spo_rf_exact_hwy" and r["max_depth"] == "-1"]
            refs[key] = statistics.mean(vals)
    return refs


def emit(panel: dict, refs: dict[str, float]) -> tuple[str, list[str]]:
    notes: list[str] = []
    L = ["% Generated by benchmarks/src/utils/make_breakeven_figure.py -- do not hand-edit."]
    colors = panel.get("colors", {})
    palette = panel.get("palette", {})
    dash = panel.get("dash", "dashed")  # tikz style used for the "dashed" series
    L += [f"\\definecolor{{brk{i}}}{{HTML}}{{{palette.get(k, c)}}}" for i, (k, _, c, _) in enumerate(DATASETS)]
    L += [f"\\definecolor{{brk{st}{i}}}{{HTML}}{{{c}}}" for (st, k), c in colors.items()
          for i, (kk, *_) in enumerate(DATASETS) if kk == k]
    plots: list[str] = []
    mins: list[tuple[int, float]] = []
    mins_dashed: list[tuple[int, float]] = []
    ys: list[float] = []
    legend_ds: set[int] = set()
    series = [(load_sweep(pattern), style) for pattern, style, _ in panel["series"]]
    keep = panel.get("datasets")
    for data, style in series:
        for i, (key, name, _, mark) in enumerate(DATASETS):
            if key not in data or (keep and key not in keep):
                continue
            if not panel.get("marks", True):
                mark = "none"
            if key not in refs:
                notes.append(f"{name}: no matched all-exact reference -> omitted")
                continue
            thr = set(data[key])
            if panel.get("common_thresholds"):
                for other, _ in series:
                    thr &= set(other.get(key, {}))
            pts = sorted((t, v / refs[key]) for t, v in data[key].items() if t in thr)
            ys += [y for _, y in pts]
            tmin, ymin = min(pts, key=lambda p: p[1])
            flat = [t for t, y in pts if y <= ymin * 1.01]
            notes.append(f"{name:24s} {style:6s} n={len(pts):2d} min {ymin:.3f} at thr {tmin}"
                         f" (within 1%: {min(flat)}..{max(flat)})")
            if style == "solid":
                mins.append((tmin, ymin))
            else:
                mins_dashed.append((tmin, ymin))
            coords = " ".join(f"({t},{y:.4f})" for t, y in pts)
            col = f"brk{style}{i}" if (style, key) in colors else f"brk{i}"
            legend_name = next((lg for pat, st, lg in panel["series"] if st == style), "")
            in_legend = (style, key) in colors or (style == "solid" and i not in legend_ds)
            if panel.get("legend_order"):
                in_legend = False  # legend is emitted afterwards in the requested order
            forget = "" if in_legend else ", forget plot"
            tikz_style = dash if style == "dashed" else style
            plots.append(f"\\addplot[color={col}, mark={mark}, {tikz_style}{forget}] coordinates {{{coords}}};")
            if in_legend:
                plots.append(f"\\addlegendentry{{{name}{', ' + legend_name if (style, key) in colors else ''}}}")
                legend_ds.add(i)
            if panel["mark_min"]:
                plots.append(f"\\addplot[color={col}, mark={mark}, mark size=2.6pt, only marks, forget plot]"
                             f" coordinates {{({tmin},{ymin:.4f})}};")
    ylo, yhi = min(ys) - 0.02, max(max(ys), 1.0) + 0.02
    ylo, yhi = round(ylo, 2), round(yhi, 2)
    yhi = panel.get("ymax", yhi)
    band_lo, band_hi = (min(t for t, _ in mins), max(t for t, _ in mins)) if mins else (0, 0)
    L += [
        "\\begin{tikzpicture}",
        f"\\begin{{axis}}[width=\\linewidth, height=5.2cm, xmode={panel.get('xmode', 'log')}, xmin={panel['xmin']}, xmax={panel['xmax']},",
        f"  ymin={ylo}, ymax={yhi}, xtick={panel['xtick']}, log ticks with fixed point,",
        f"  xlabel={{{panel.get('xlabel', 'dynamic switch threshold (rows; nodes below it use exact)')}}},",
        f"  ylabel={{{panel.get('ylabel', 'train time / all-exact')}}}, tick label style={{font=\\scriptsize}},",
        f"  label style={{font=\\scriptsize}}, {'grid=major, grid style={gray!20}' if panel.get('grid', True) else 'grid=none'}, axis line style={{gray!60}}, tick style={{draw=none}},",
        "  every axis plot/.append style={line width=0.7pt, mark size=1.1pt},",
        f"  legend style={{font=\\scriptsize, draw=none, fill=none, at={{(0.5,1.02)}}, anchor=south, legend columns={2 if panel.get('legend_all') else 3},",
        "    /tikz/every even column/.append style={column sep=4pt}}, legend cell align=left,",
        f"  {panel.get('extra_axis', '')}scaled y ticks=false, yticklabel style={{/pgf/number format/fixed, /pgf/number format/precision=2}}]",
        f"\\addplot[gray!70, dashed, line width=0.6pt, forget plot] coordinates {{({panel['xmin']},1) ({panel['xmax']},1)}};",
    ]
    if panel.get("ref_label"):
        L.append(f"\\node[font=\\scriptsize, gray!90, {panel['ref_label']} {{all nodes exact (VQSort)}};")
    if panel["band"]:
        L.insert(-1, f"\\fill[gray!14] (axis cs:{band_lo},{ylo}) rectangle (axis cs:{band_hi},{yhi});")
    bd = panel.get("band_dashed")
    if bd and mins_dashed:
        # True = computed argmin range of the dashed series; a (lo, hi) tuple = fixed band.
        dlo, dhi = bd if isinstance(bd, tuple) else (min(t for t, _ in mins_dashed), max(t for t, _ in mins_dashed))
        L.insert(-1, f"\\fill[gray!30] (axis cs:{dlo},{ylo}) rectangle (axis cs:{dhi},{yhi});")
        notes.append(f"breakeven band (argmin range, dashed series): {dlo}..{dhi}")
    L += plots
    if panel.get("legend_order"):
        idx = {k: i for i, (k, *_) in enumerate(DATASETS)}
        for item in panel["legend_order"]:
            if item.startswith("series:"):
                st = item.split(":")[1]
                L.append(f"\\addlegendimage{{color=black, {dash if st == 'dashed' else st}, line width=0.7pt, mark=none}}")
            else:
                L.append(f"\\addlegendimage{{color=brk{idx[item]}, solid, line width=0.7pt, mark=none}}")
            L.append(f"\\addlegendentry{{{panel['legend_names'][item]}}}")
    elif not colors:
        for _, style, name in panel["series"]:
            L.append(f"\\addlegendimage{{color=black, {dash if style == 'dashed' else style}, line width=0.7pt, mark=none}}")
            L.append(f"\\addlegendentry{{{name}}}")
    L += ["\\end{axis}", "\\end{tikzpicture}"]
    notes.append(f"breakeven band (argmin range, solid series): {band_lo}..{band_hi}")
    return "\n".join(L) + "\n", notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(ROOT / "paper" / "spaa27" / "figures" / "results"))
    a = ap.parse_args()
    refs = load_refs()
    for k, v in refs.items():
        print(f"ref all-exact {k:26s} {v:8.2f} s")
    for pname, panel in PANELS.items():
        tex, notes = emit(panel, refs)
        out = Path(a.outdir) / panel["out"]
        out.write_text(tex)
        print(f"\n[{pname}] -> {out}")
        print("\n".join("  " + n for n in notes))


if __name__ == "__main__":
    main()
