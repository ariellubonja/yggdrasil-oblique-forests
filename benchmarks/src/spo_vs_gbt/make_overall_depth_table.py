#!/usr/bin/env python3
"""tab:train-time-by-depth -- end-to-end SPO-RF training time (s) by dataset, one section per depth.

Rows: the pinned speedup-map selection (B7, 2026-09-08): HIGGS, SUSY, Epsilon,
GiveMeSomeCredit, trunk 1M x {32,512,2048} and the row-column shapes with rows > 100k
or cols > 100k (D8-rev2; 15k x 4096 and 15k x 40k are run but not tabulated).
Columns: 7 split finders -- Exact (Highway VQSort),
Random histogram (scalar; 64 and 256 bins), Vectorized random histogram (AVX2 64 bins,
AVX-512 256 bins), Vectorized dynamic (AVX2 64 bins, threshold 250; AVX-512 256 bins,
threshold 1000). Dynamic (scalar) and Exact (std::sort) are commented out.
Speedups beside Our Methods are over Exact (HWY).
Depths: 6, 10, 16, 24, full (purity). 240 trees, min_examples 1, 48 threads, seed 1.

Source: benchmarks/results/runtime/speedup_map_by_dataset/speedup_map.csv (rep 1) plus any
speedup_map_rep<k>.csv beside it (reps 2..); a cell shows median +- sample std
over the reps present (a single run shows just the value). Missing cells print as --.
Emits table_train_time_by_depth.tex (do not hand-edit) and a plain-text preview.
--gbt: same layout for the SPO-GBT arms (300 trees, depth 6, min_examples 5; no dynamic
arm, user directive 2026-09-24) -> table_train_time_gbt_depth6.tex, label tab:train-time-gbt-depth6.
--sort: appendix table Exact std::sort vs Exact HWY -> table_train_time_exact_sort.tex.
Without --gbt, the GBT table is also appended inside the last SPO-RF float, so Table 5 shares
Table 4 Part 2's page (2026-09-25); main.tex no longer inputs table_train_time_gbt_depth6.tex.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "benchmarks" / "results" / "runtime" / "speedup_map_by_dataset"

ARMS = [  # (arm, header line 1, header line 2)
    # ("spo_rf_exact_stdsort", "Exact", "std::sort"),  # dropped 2026-09-23
    ("spo_rf_exact_hwy", "Exact", "HWY"),
    ("spo_rf_rand_scalar", "Random Hist.", "scalar, 64 bins"),
    ("spo_rf_rand256_scalar", "Random Hist.", "scalar, 256 bins"),
    ("spo_rf_rand_vec", "Vec. Random Hist.", "AVX2, 64 bins"),
    ("spo_rf_rand256_vec", "Vec. Random Hist.", "AVX-512, 256 bins"),
    # ("spo_rf_dyn_scalar", "Dynamic Hist.", "scalar, 64 bins \\& HWY"),  # dropped 2026-09-14
    ("spo_rf_dyn_vec", "Vec. Dynamic Hist.", "AVX2, 64 bins \\& HWY"),
    ("spo_rf_dyn256_vec", "Vec. Dynamic Hist.", "AVX-512, 256 bins \\& HWY"),
]
GROUPS = [("Baselines", 3), ("Our Methods", 4)]  # column blocks, in ARMS order
# Speedup shown beside a cell = (best of these reference arms) / cell; missing refs are skipped.
SPEEDUP_REF = {
    "spo_rf_rand_vec": ["spo_rf_exact_hwy"],
    "spo_rf_rand256_vec": ["spo_rf_exact_hwy"],
    "spo_rf_dyn_vec": ["spo_rf_exact_hwy"],
    "spo_rf_dyn256_vec": ["spo_rf_exact_hwy"],
}
DATASETS = [  # (key in speedup_map.csv, display name)
    ("higgs_10500000", "HIGGS 10.5M$\\times$28"),
    ("SUSY", "SUSY 4.5M$\\times$18"),
    ("EPSILON", "Epsilon 400k$\\times$2000"),
    ("GiveMeSomeCredit", "GiveMeSomeCredit 150k$\\times$10"),
    ("YOUTUBE8M", "YouTube-8M 3.9M$\\times$1152"),
    ("SHIFTS_WEATHER", "Shifts weather 3.1M$\\times$127"),
    ("CLIMSIM", "ClimSim 10.1M$\\times$124"),
    ("JANE_STREET", "Jane Street 10.2M$\\times$82"),
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
# Mirrors the author's Overleaf caption (2026-09-25); edit here, not on Overleaf.
CAPTION = "End-to-end SPO-RF training time (s) by tree depth."
LABEL = "tab:train-time-by-depth"

# --gbt: SPO-GBT arms, one depth section. Dynamic arms were not run (directive 2026-09-24).
GBT_ARMS = [
    ("spo_gbt_exact_hwy", "Exact", "HWY"),
    ("spo_gbt_rand_scalar", "Random Hist.", "scalar, 64 bins"),
    ("spo_gbt_rand256_scalar", "Random Hist.", "scalar, 256 bins"),
    ("spo_gbt_rand_vec", "Vec. Random Hist.", "AVX2, 64 bins"),
    ("spo_gbt_rand256_vec", "Vec. Random Hist.", "AVX-512, 256 bins"),
]
GBT_GROUPS = [("Baselines", 3), ("Our Methods", 2)]
GBT_SPEEDUP_REF = {"spo_gbt_rand_vec": ["spo_gbt_exact_hwy"],
                   "spo_gbt_rand256_vec": ["spo_gbt_exact_hwy"]}
GBT_DEPTHS = [6]
GBT_DEPTH_PAGES = [[6]]
GBT_CAPTION = "End-to-end SPO-GBT training time (s) at depth 6."
GBT_LABEL = "tab:train-time-gbt-depth6"

# --sort: appendix table, Exact std::sort vs Exact HWY; depths as column pairs.
SORT_ARMS = [("spo_rf_exact_stdsort", "std::sort"), ("spo_rf_exact_hwy", "HWY")]
SORT_CAPTION = ("End-to-end SPO-RF training time (s) of the Exact split finder with std::sort "
                "vs.\\ Highway VQSort, by tree depth. Speedup of HWY over std::sort in parentheses.")
SORT_LABEL = "tab:train-time-exact-sort"


def use_gbt() -> None:
    """Switch the module-level layout to the SPO-GBT table."""
    global ARMS, GROUPS, SPEEDUP_REF, DEPTHS, DEPTH_PAGES, CAPTION, CAPTION_P1_NOTE, LABEL
    ARMS, GROUPS, SPEEDUP_REF = GBT_ARMS, GBT_GROUPS, GBT_SPEEDUP_REF
    DEPTHS, DEPTH_PAGES, CAPTION, CAPTION_P1_NOTE, LABEL = (
        GBT_DEPTHS, GBT_DEPTH_PAGES, GBT_CAPTION, "", GBT_LABEL)


def load_cells(max_reps: int = 0, family: str = "rf") -> tuple[pd.DataFrame, int]:
    """(dataset, arm, depth) -> median / std of train_s over reps; returns (df, max reps seen)."""
    min_ex = {"rf": 1, "gbt": 5}[family]  # the study's per-family defaults
    files = [RES / "speedup_map.csv"] + sorted(RES.glob("speedup_map_rep*.csv"))
    if max_reps > 0:  # --reps N: use only reps 1..N
        files = files[:max_reps]
    parts = []
    for k, f in enumerate(files, 1):
        d = pd.read_csv(f)
        d = d[(d.status == "OK") & (d.min_examples == min_ex) & (d.family == family)]
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


def emit_tex(df: pd.DataFrame, nrep: int, bare: bool = False) -> str:
    """bare: tabular + caption + label only, no float wrapper (single-page layouts)."""
    cell = {(r.dataset, r.arm, int(r.max_depth)): (r.train_s, r.std_s) for r in df.itertuples()}
    ncol = len(ARMS) + 1
    npage = len(DEPTH_PAGES)
    L = ["% Generated by benchmarks/src/spo_vs_gbt/make_overall_depth_table.py -- do not hand-edit."]
    for page, depths in enumerate(DEPTH_PAGES):
        covers = ", ".join(_depth_label(d).lower() for d in depths)
        L += ([] if bare else [
            "\\begin{table*}[p]", "    \\centering",
        ]) + (["    \\ContinuedFloat"] if page else []) + [
            "    \\footnotesize",
            "    \\setlength{\\tabcolsep}{2.5pt}",
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
            "    \\caption{" + part + CAPTION + "}",
            "    \\label{" + LABEL + ("" if page == 0 else f"-p{page + 1}") + "}",
        ] + ([] if bare else ["\\end{table*}", ""])
    return "\n".join(L[1:] if bare else L)


def emit_sort_tex(df: pd.DataFrame) -> str:
    """One table*: rows = datasets with a std::sort run, columns = (std::sort, HWY) per depth."""
    cell = {(r.dataset, r.arm, int(r.max_depth)): (r.train_s, r.std_s) for r in df.itertuples()}
    ref, hwy = SORT_ARMS[0][0], SORT_ARMS[1][0]
    rows = [(k, n) for k, n in DATASETS if any((k, ref, d) in cell for d in DEPTHS)]
    L = ["% Generated by benchmarks/src/spo_vs_gbt/make_overall_depth_table.py --sort -- do not hand-edit.",
         "\\begin{table*}[p]", "    \\centering", "    \\footnotesize",
         "    \\setlength{\\tabcolsep}{2.5pt}",
         "    \\begin{tabular}{l|" + "|".join("cc" for _ in DEPTHS) + "}", "        \\hline",
         "        \\multirow{2}{*}{\\textbf{Dataset}} & " + " & ".join(
             f"\\multicolumn{{2}}{{c{'|' if i < len(DEPTHS) - 1 else ''}}}{{\\textbf{{{_depth_label(d)}}}}}"
             for i, d in enumerate(DEPTHS)) + " \\\\",
         "         & " + " & ".join(f"\\scriptsize {h}" for d in DEPTHS for _, h in SORT_ARMS) + " \\\\",
         "        \\hline"]
    for key, name in rows:
        cells = []
        for d in DEPTHS:
            r, h = cell.get((key, ref, d)), cell.get((key, hwy, d))
            sp = f" ({r[0] / h[0]:.1f}$\\times$)" if r and h and h[0] > 0 else ""
            cells += [_fmt(r and (r[0], float('nan'))), _fmt(h and (h[0], float('nan'))) + sp]  # median only
        L.append(f"        {name} & " + " & ".join(cells) + " \\\\")
    L += ["        \\hline", "    \\end{tabular}", "    \\caption{" + SORT_CAPTION + "}",
          "    \\label{" + SORT_LABEL + "}", "\\end{table*}", ""]
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
    ap.add_argument("--out", default=None, help="default: paper/spaa27/tables/table_train_time_{by_depth,gbt_depth6}.tex")
    ap.add_argument("--text-only", action="store_true")
    ap.add_argument("--reps", type=int, default=0, help="use only reps 1..N (0 = all)")
    ap.add_argument("--gbt", action="store_true", help="SPO-GBT arms, depth 6 only")
    ap.add_argument("--sort", action="store_true", help="appendix: Exact std::sort vs HWY")
    a = ap.parse_args()
    if a.sort:
        out = a.out or str(ROOT / "paper" / "spaa27" / "tables" / "table_train_time_exact_sort.tex")
        Path(out).write_text(emit_sort_tex(load_cells(a.reps, "rf")[0]))
        print(f"wrote {out}")
        return
    if a.gbt:
        use_gbt()
    if a.out is None:
        a.out = str(ROOT / "paper" / "spaa27" / "tables" / ("table_train_time_gbt_depth6.tex" if a.gbt else "table_train_time_by_depth.tex"))
    df, nrep = load_cells(a.reps, "gbt" if a.gbt else "rf")
    print(emit_text(df))
    if not a.text_only:
        if a.gbt:
            Path(a.out).write_text(emit_tex(df, nrep))
        else:  # Table 5 (GBT) rides in Table 4 Part 2's float so both share a page
            rf_lines = emit_tex(df, nrep).split("\n")
            use_gbt()
            gdf, gnrep = load_cells(a.reps, "gbt")
            gbt = "    \\vspace{10pt}\n" + emit_tex(gdf, gnrep, bare=True)
            end = len(rf_lines) - 1 - rf_lines[::-1].index("\\end{table*}")
            Path(a.out).write_text("\n".join(rf_lines[:end] + [gbt] + rf_lines[end:]))
        print(f"wrote {a.out} (reps={nrep})")


if __name__ == "__main__":
    main()
