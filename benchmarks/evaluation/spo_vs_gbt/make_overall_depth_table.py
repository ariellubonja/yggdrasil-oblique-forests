#!/usr/bin/env python3
"""tab:overall -- end-to-end SPO-RF training time (s) by dataset, one section per depth.

Rows: the pinned speedup-map selection (B7, 2026-09-08): HIGGS, SUSY, Epsilon,
GiveMeSomeCredit, trunk 1M x {32,512,2048} and the row-column shapes with rows > 100k
or cols > 100k (D8-rev2; 15k x 4096 and 15k x 40k are run but not tabulated).
Columns: 6 split finders -- Exact (std::sort), Exact (Highway VQSort),
Random histogram (scalar, 64 bins), Vectorized random histogram (AVX2, 64 bins),
Dynamic (scalar, 64 bins), Vectorized dynamic (AVX2, 64 bins).
Depths: 6, 10, 16, 24, full (purity). 240 trees, min_examples 1, 48 threads, seed 1.

Source: benchmarks/results/spo_vs_gbt/speedup_map.csv (rep 1) plus any
speedup_map_rep<k>.csv beside it (reps 2..); a cell shows median +- sample std
over the reps present (a single run shows just the value) and the caption states
the rep count. Missing cells print as --.
Emits table_overall_depth.tex (do not hand-edit) and a plain-text preview.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "benchmarks" / "results" / "spo_vs_gbt"

ARMS = [  # (arm, header line 1, header line 2)
    ("spo_rf_exact_stdsort", "Exact", "std::sort"),
    ("spo_rf_exact_hwy", "Exact", "HWY"),
    ("spo_rf_rand_scalar", "Random Hist.", "scalar, 64 bins"),
    ("spo_rf_rand_vec", "Vec. Random Hist.", "AVX2, 64 bins"),
    ("spo_rf_dyn_scalar", "Dynamic Hist.", "scalar, 64 bins \\& HWY"),
    ("spo_rf_dyn_vec", "Vec. Dynamic Hist.", "AVX2, 64 bins \\& HWY"),
]
GROUPS = [("Baselines", 3), ("Our Methods", 3)]  # column blocks, in ARMS order
# Speedup shown beside a cell = (best of these reference arms) / cell; missing refs are skipped.
SPEEDUP_REF = {
    "spo_rf_rand_vec": ["spo_rf_rand_scalar"],
    "spo_rf_dyn_scalar": ["spo_rf_exact_hwy", "spo_rf_rand_scalar"],
    "spo_rf_dyn_vec": ["spo_rf_exact_hwy", "spo_rf_rand_vec"],
}
DATASETS = [  # (key in speedup_map.csv, display name)
    ("higgs_10500000", "HIGGS 10.5M$\\times$28"),
    ("SUSY", "SUSY 4.5M$\\times$18"),
    ("EPSILON", "Epsilon 400k$\\times$2000"),
    ("GiveMeSomeCredit", "GiveMeSomeCredit 150k$\\times$10"),
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
DEPTHS = [6, 10, 16, 24, -1]
DEPTH_LABEL = {-1: "Full depth (purity)"}
# One table* float per group; the whole table does not fit on one page.
DEPTH_PAGES = [[6, 10, 16], [24, -1]]


def load_cells() -> tuple[pd.DataFrame, int]:
    """(dataset, arm, depth) -> median / std of train_s over reps; returns (df, max reps seen)."""
    files = [RES / "speedup_map.csv"] + sorted(RES.glob("speedup_map_rep*.csv"))
    parts = []
    for k, f in enumerate(files, 1):
        d = pd.read_csv(f)
        d = d[(d.status == "OK") & (d.min_examples == 1) & (d.family == "rf")]
        d = d.assign(rep=k)
        parts.append(d[["dataset", "arm", "max_depth", "train_s", "rep"]])
    df = pd.concat(parts)
    g = df.groupby(["dataset", "arm", "max_depth"], as_index=False).agg(
        train_s=("train_s", "median"), std_s=("train_s", lambda x: x.std(ddof=1)),
        n=("rep", "nunique"))
    return g, int(g.n.max()) if len(g) else 0


def _num(v: float) -> str:
    return f"{v:.1f}" if v < 100 else f"{v:.0f}"


def _fmt(c: tuple[float, float] | None, pm: str = "$\\pm$") -> str:
    """median +- std; std omitted when only one rep (NaN)."""
    if c is None:
        return "--"
    med, sd = c
    if sd != sd:  # NaN: single run
        return _num(med)
    return f"{_num(med)} {pm} {_num(sd) if med < 100 else f'{sd:.0f}'}"


def _speedup(cell: dict, key: str, arm: str, depth: int) -> str:
    """ ($ref/value$\\times$) against the fastest available reference arm; empty when unavailable."""
    me = cell.get((key, arm, depth))
    refs = [cell.get((key, r, depth)) for r in SPEEDUP_REF.get(arm, [])]
    refs = [r[0] for r in refs if r is not None]
    if me is None or not refs or me[0] <= 0:
        return ""
    return f" ({min(refs) / me[0]:.1f}$\\times$)"


def _depth_label(d: int) -> str:
    return DEPTH_LABEL.get(d, f"Depth {d}")


def emit_tex(df: pd.DataFrame, nrep: int) -> str:
    cell = {(r.dataset, r.arm, int(r.max_depth)): (r.train_s, r.std_s) for r in df.itertuples()}
    ncol = len(ARMS) + 1
    reps = "single run" if nrep <= 1 else f"median $\\pm$ sample std over {nrep} runs"
    caption = (
        "End-to-end SPO-RF training time (s) by tree depth: 240 trees, min\\_examples 1, "
        "48 threads, m7i.metal-24xl, " + reps + ". Exact = presorted scan with std::sort or "
        "Highway VQSort. Histogram finders use 64 bins; vectorized variants use the AVX2 "
        "upper\\_bound kernel. The dynamic switch to exact is binner-dependent: below 250 "
        "examples for the vectorized binner, 4600 for the scalar one. Trunk = synthetic "
        "$R\\times C$ dataset. -- = not run.")
    npage = len(DEPTH_PAGES)
    L = ["% Generated by benchmarks/evaluation/spo_vs_gbt/make_overall_depth_table.py -- do not hand-edit."]
    for page, depths in enumerate(DEPTH_PAGES):
        covers = ", ".join(_depth_label(d).lower() for d in depths)
        L += [
            "\\begin{table*}[p]", "    \\centering",
        ] + (["    \\ContinuedFloat"] if page else []) + [
            "    \\footnotesize",
            "    \\setlength{\\tabcolsep}{4pt}",
            "    \\begin{tabular}{l|" + "|".join("c" * n for _, n in GROUPS) + "}",
            "        \\hline",
            "        \\multirow{3}{*}{\\textbf{Dataset}} & " + " & ".join(
                f"\\multicolumn{{{n}}}{{c{'|' if i < len(GROUPS) - 1 else ''}}}{{\\textbf{{{g}}}}}"
                for i, (g, n) in enumerate(GROUPS)) + " \\\\",
            "         & " + " & ".join(f"\\textbf{{{h1}}}" for _, h1, _ in ARMS) + " \\\\",
            "         & " + " & ".join(f"\\scriptsize {h2}" for _, _, h2 in ARMS) + " \\\\",
            "        \\hline",
        ]
        for d in depths:
            L.append(f"        \\multicolumn{{{ncol}}}{{l}}{{\\textit{{{_depth_label(d)}}}}} \\\\")
            for key, name in DATASETS:
                cells = [_fmt(cell.get((key, arm, d))) + _speedup(cell, key, arm, d)
                         for arm, _, _ in ARMS]
                L.append(f"        {name} & " + " & ".join(cells) + " \\\\")
            L.append("        \\hline")
        part = f"(Part {page + 1} of {npage}: {covers}.) " if npage > 1 else ""
        L += [
            "    \\end{tabular}",
            "    \\caption{" + part + caption + "}",
            "    \\label{tab:overall" + ("" if page == 0 else f"-p{page + 1}") + "}",
            "\\end{table*}", "",
        ]
    return "\n".join(L)


def emit_text(df: pd.DataFrame) -> str:
    cell = {(r.dataset, r.arm, int(r.max_depth)): (r.train_s, r.std_s) for r in df.itertuples()}
    heads = [f"{h1} ({h2})" for _, h1, h2 in ARMS]
    w = 28
    out = []
    for d in DEPTHS:
        out.append(f"== {DEPTH_LABEL.get(d, f'Depth {d}')}")
        out.append("Dataset".ljust(w) + "".join(h[:22].rjust(24) for h in heads))
        for key, name in DATASETS:
            name = name.replace("$\\times$", "x")
            vals = [cell.get((key, arm, d)) for arm, _, _ in ARMS]
            out.append(name.ljust(w) + "".join(_fmt(v, "+-").rjust(24) for v in vals))
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results" / "overleaf-spaa27" / "table_overall_depth.tex"))
    ap.add_argument("--text-only", action="store_true")
    a = ap.parse_args()
    df, nrep = load_cells()
    print(emit_text(df))
    if not a.text_only:
        Path(a.out).write_text(emit_tex(df, nrep))
        print(f"wrote {a.out} (reps={nrep})")


if __name__ == "__main__":
    main()
