#!/usr/bin/env python3
"""fig:speedup-heatmap at three depths -- row x column map of the SPO-RF speedup of
Dyn-Vec (AVX2, 64 bins, threshold 250) over Exact (Highway VQSort), one panel
each for max_depth 6, 16 and unlimited (purity). 240 trees, 48 threads, m7i.

Sources (committed under benchmarks/results/runtime/):
  speedup_map_by_dataset/speedup_map*.csv      trunk grid + HIGGS/SUSY/Epsilon/YouTube-8M/
                                               Shifts weather/ClimSim/Jane Street,
                                               median over reps per (cell, arm)
Speedup = median(train_s exact_hwy) / median(train_s dyn_vec). Circles: Trunk
synthetic datasets; squares: natural datasets. Writes fig_rowcol_map_depths.{pdf,png}
(background = inverse-distance interpolation of the speedup over the 3 nearest points
in log-log space, a prediction for unmeasured shapes) + rowcol_map_depths.csv under speedup_map_by_dataset/analysis and copies the PDF to
paper/spaa27/figures/results/spo_vs_gbt_rowcol_map_depths.pdf.
"""
from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "benchmarks" / "results" / "runtime"
SPM = RES / "speedup_map_by_dataset"
PAPER_FIG = ROOT / "paper" / "spaa27" / "figures" / "results"

EXACT, DYN = "spo_rf_exact_hwy", "spo_rf_dyn_vec"
DEPTHS = [(6, "Depth 6"), (16, "Depth 16"), (-1, "Unlimited Depth")]
NATURAL = {"higgs_10500000": "HIGGS", "SUSY": "SUSY", "EPSILON": "Epsilon",
           "YOUTUBE8M": "YouTube-8M", "SHIFTS_WEATHER": "Shifts weather",
           "CLIMSIM": "ClimSim", "JANE_STREET": "Jane Street"}
# Name placement (pgf anchor, xshift pt, yshift pt); default = centred below the square.
# HIGGS, Jane Street and ClimSim share y ~= 10M rows and the last two squares overlap: per
# panel the better speedup is drawn on top (value inside), the covered one's value goes
# with its name (user directive 2026-09-25).
NAME_POS = {"HIGGS": ("south", 0, 5), "ClimSim": ("west", 4, 0), "Jane Street": ("north", 0, -5)}
NAME_DEFAULT = ("north", 0, -5)
OVERLAP = ("ClimSim", "Jane Street")


def covered(t: pd.DataFrame) -> set[str]:
    """Datasets whose marker is hidden under a better one in this panel."""
    g = t[t["dataset"].isin(OVERLAP)]
    return set() if len(g) < 2 else {g.loc[g["speedup"].idxmin(), "dataset"]}
CMAP = "Spectral_r"  # blue (low) -> yellow -> red (high); user pick 2026-09-24
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
                     "mathtext.fontset": "stix"})


def load_points() -> pd.DataFrame:
    df = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(str(SPM / "speedup_map*.csv"))))
    df = df[(df["status"] == "OK") & df["arm"].isin([EXACT, DYN])
            & (pd.to_numeric(df["min_examples"], errors="coerce") == 1)]
    keep = df["dataset"].str.startswith("trunk_") | df["dataset"].isin(NATURAL)
    df = df[keep & df["max_depth"].isin([d for d, _ in DEPTHS])]
    med = (df.groupby(["dataset", "rows", "features", "max_depth", "arm"])["train_s"]
             .median().unstack("arm").dropna().reset_index())
    med["min_reps"] = (df.groupby(["dataset", "max_depth", "arm"]).size()
                       .unstack("arm").min(axis=1).reindex(
                           list(zip(med["dataset"], med["max_depth"]))).values)
    med["kind"] = med["dataset"].map(lambda d: "trunk" if d.startswith("trunk_") else "natural")
    med["dataset"] = med["dataset"].replace(NATURAL)

    # YouTube-8M: depth-table runs (all depths) since 2026-09-24.
    tab = med.copy()
    tab["speedup"] = tab[EXACT] / tab[DYN]
    return tab.sort_values(["max_depth", "rows", "features"]).reset_index(drop=True)


def idw_field(t: pd.DataFrame, xlim: tuple[float, float], ylim: tuple[float, float],
              k: int = 3, power: float = 2.0, n: int = 220):
    """Inverse-distance-weighted speedup over the k nearest points in log-log space."""
    pts = np.column_stack([np.log10(t["features"]), np.log10(t["rows"])])
    gx = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), n)
    gy = np.logspace(np.log10(ylim[0]), np.log10(ylim[1]), n)
    X, Y = np.meshgrid(gx, gy)
    q = np.column_stack([np.log10(X.ravel()), np.log10(Y.ravel())])
    d, i = cKDTree(pts).query(q, k=min(k, len(pts)))
    w = 1.0 / np.maximum(d, 1e-6) ** power
    z = (w * t["speedup"].to_numpy()[i]).sum(axis=1) / w.sum(axis=1)
    return X, Y, z.reshape(X.shape)


def plot(tab: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.6), sharey=True)
    vmin, vmax = 1.0, float(tab["speedup"].max())  # scale starts at "no speedup"
    xlim = (float(tab["features"].min()) / 4, float(tab["features"].max()) * 4)
    ylim = (float(tab["rows"].min()) / 1.7, float(tab["rows"].max()) * 2.6)
    for ax, (depth, title) in zip(axes, DEPTHS):
        t = tab[tab["max_depth"] == depth].sort_values("speedup")  # best drawn last = on top
        hidden = covered(t)
        X, Y, Z = idw_field(t, xlim, ylim)
        ax.pcolormesh(X, Y, Z, cmap=CMAP, vmin=vmin, vmax=vmax, alpha=0.8,
                      shading="nearest", rasterized=True, zorder=1)
        for kind, marker, size in (("trunk", "o", 330), ("natural", "s", 380)):
            g = t[t["kind"] == kind]
            if g.empty:
                continue
            sc = ax.scatter(g["features"], g["rows"], c=g["speedup"], cmap=CMAP,
                            vmin=vmin, vmax=vmax, s=size, marker=marker,
                            edgecolors="black", linewidths=0.6, zorder=3)
            for _, r in g.iterrows():
                # Value inside the marker (text colour by marker luminance); name outside.
                rgb = plt.get_cmap(CMAP)((r["speedup"] - vmin) / max(vmax - vmin, 1e-9))[:3]
                lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                inline = r["dataset"] in hidden
                if not inline:
                    ax.annotate(f"{r['speedup']:.2f}", (r["features"], r["rows"]), ha="center",
                                va="center", fontsize=5.6, color="black" if lum > 0.55 else "white",
                                zorder=4)
                if kind == "natural":
                    anc, dx, dy = NAME_POS.get(r["dataset"], NAME_DEFAULT)
                    txt = r["dataset"] + (f" {r['speedup']:.2f}" if inline else "")
                    ax.annotate(txt, (r["features"], r["rows"]), textcoords="offset points",
                                xytext=(2.4 * dx, 2.4 * dy), ha="left" if "west" in anc else "center",
                                va="bottom" if "south" in anc else "center" if anc == "west" else "top",
                                fontsize=7, zorder=4)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel("Features (log scale)")
        ax.set_title(title, fontsize=10)
        ax.tick_params(which="both", length=0)
    axes[0].set_ylabel("Rows (log scale)")
    fig.colorbar(sc, ax=axes, label="Speedup over Exact", shrink=0.9, pad=0.02)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def write_pgf(tab: pd.DataFrame, out: Path, n: int = 60) -> None:
    """pgfplots groupplot twin of plot(): Overleaf's MCP is text-only, so the
    paper gets this .tex (\\input it) instead of the PDF."""
    vmin, vmax = 1.0, float(tab["speedup"].max())  # scale starts at "no speedup"
    xlim = (float(tab["features"].min()) / 2.5, float(tab["features"].max()) * 2.5)
    ylim = (float(tab["rows"].min()) / 1.7, float(tab["rows"].max()) * 2.6)
    cm = plt.get_cmap(CMAP)
    stops = "; ".join(f"rgb255({i}cm)=({int(r*255)},{int(g*255)},{int(b*255)})"
                      for i, (r, g, b, _) in enumerate(cm(np.linspace(0, 1, 11))))
    L = [f"% Generated by benchmarks/src/spo_vs_gbt/make_rowcol_map_depths.py -- do not hand-edit.",
         "% Needs \\usepackage{pgfplots}\\pgfplotsset{compat=1.18}\\usepgfplotslibrary{groupplots}.",
         "% Background: inverse-distance-weighted speedup (3 nearest points, log-log), see idw_field().",
         "\\begin{tikzpicture}",
         "\\pgfplotsset{",
         f"  colormap={{{CMAP.lower().replace('_', '')}}}{{{stops}}},",
         "  rcmark/.style={scatter, only marks, scatter src=explicit,",
         "    scatter/use mapped color={draw=black, fill=mapped color, line width=0.4pt}},",
         "  rclabel/.style={only marks, mark=none, point meta=explicit, forget plot,",
         "    nodes near coords={\\pgfmathprintnumber[fixed,fixed zerofill,precision=2]{\\pgfplotspointmeta}},",
         "    nodes near coords style={font=\\fontsize{4.2}{5}\\selectfont, anchor=center, inner sep=0pt}},",
         "  rcname/.style={only marks, mark=none, forget plot,",
         "    visualization depends on={value \\thisrow{name} \\as \\nm},",
         "    nodes near coords={\\nm}, nodes near coords style={font=\\tiny, inner sep=1pt}},",
         "}",
         "\\begin{groupplot}[",
         "  group style={group size=3 by 1, horizontal sep=0.35cm, ylabels at=edge left, yticklabels at=edge left},",
         "  width=0.335\\linewidth, height=0.33\\linewidth, xmode=log, ymode=log,",
         f"  point meta min={vmin:.4f}, point meta max={vmax:.4f},",
         f"  xmin={xlim[0]:.6g}, xmax={xlim[1]:.6g}, ymin={ylim[0]:.6g}, ymax={ylim[1]:.6g},",
         "  xtick={1e1,1e2,1e3,1e4,1e5,1e6}, ytick={1e4,1e5,1e6,1e7},",
         "  xlabel={Features}, ylabel={Rows}, label style={font=\\small}, tick label style={font=\\scriptsize},",
         "  title style={font=\\small, yshift=-3pt}, tick style={draw=none}, enlargelimits=false, clip=true,",
         "]"]
    for k, (depth, title) in enumerate(DEPTHS):
        t = tab[tab["max_depth"] == depth].sort_values("speedup")  # best drawn last = on top
        hidden = covered(t)
        X, Y, Z = idw_field(t, xlim, ylim, n=n)
        cb = (", colorbar, colorbar style={width=0.18cm, tick label style={font=\\scriptsize}, yticklabel={\\pgfmathprintnumber[fixed,fixed zerofill,precision=1]{\\tick}},"
              " ylabel={Speedup over Exact}, ylabel style={font=\\scriptsize}}") if k == 2 else ""
        L.append(f"\\nextgroupplot[title={{\\strut {title}}}{cb}]")
        L.append(f"\\addplot[surf, shader=interp, point meta=explicit, mesh/rows={n}, mesh/cols={n}, forget plot] table[x=x, y=y, meta=z] {{")
        L.append("x y z")
        L += [f"{x:.5g} {y:.5g} {z:.4f}" for x, y, z in zip(X.ravel(), Y.ravel(), Z.ravel())]
        L.append("};")
        for kind, mark, ms in (("trunk", "*", "6.2pt"), ("natural", "square*", "5.6pt")):
            g = t[t["kind"] == kind]
            if g.empty:
                continue
            rows = "\n".join(f"{r.features:.5g} {r.rows:.5g} {r.speedup:.4f}" for r in g.itertuples())
            L.append(f"\\addplot[rcmark, mark={mark}, mark size={ms}] table[x=x, y=y, meta=meta] {{\nx y meta\n{rows}\n}};")
            g = g[~g["dataset"].isin(hidden)]  # their value goes with the name
            for colour in ("black", "white"):
                gg = g[[("black" if _lum(cm, r, vmin, vmax) > 0.55 else "white") == colour for r in g["speedup"]]]
                if gg.empty:
                    continue
                rows = "\n".join(f"{r.features:.5g} {r.rows:.5g} {r.speedup:.4f}" for r in gg.itertuples())
                L.append(f"\\addplot[rclabel, nodes near coords style={{text={colour}}}] table[x=x, y=y, meta=meta] {{\nx y meta\n{rows}\n}};")
        nat = t[t["kind"] == "natural"]
        for pos in sorted({NAME_POS.get(d, NAME_DEFAULT) for d in nat["dataset"]}):
            gg = nat[[NAME_POS.get(d, NAME_DEFAULT) == pos for d in nat["dataset"]]]
            # pgfplots splits table cells on spaces: tie the words with ~.
            rows = "\n".join(f"{r.features:.5g} {r.rows:.5g} {r.dataset.replace(' ', '~')}"
                             + (f"~{r.speedup:.2f}" if r.dataset in hidden else "")
                             for r in gg.itertuples())
            anc, dx, dy = pos
            L.append(f"\\addplot[rcname, nodes near coords style={{anchor={anc}, xshift={dx}pt, yshift={dy}pt}}] table[x=x, y=y] {{\nx y name\n{rows}\n}};")
    L += ["\\end{groupplot}", "\\end{tikzpicture}", ""]
    out.write_text("\n".join(L))


def _lum(cm, v: float, vmin: float, vmax: float) -> float:
    r, g, b, _ = cm((v - vmin) / max(vmax - vmin, 1e-9))
    return 0.299 * r + 0.587 * g + 0.114 * b


def main() -> None:
    global CMAP
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=Path, default=SPM / "analysis")
    ap.add_argument("--no_paper_copy", action="store_true")
    ap.add_argument("--cmap", default=CMAP, help="matplotlib colormap name")
    ap.add_argument("--name", default="fig_rowcol_map_depths", help="output basename")
    ap.add_argument("--pgf", type=Path, default=PAPER_FIG / "speedup_heatmap_rows_features.tex",
                    help="pgfplots twin for Overleaf (text-only MCP); '' to skip")
    args = ap.parse_args()
    CMAP = args.cmap
    tab = load_points()
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)
    tab.to_csv(args.out_dir / "rowcol_map_depths.csv", index=False)
    fig = args.out_dir / "figures" / f"{args.name}.pdf"
    plot(tab, fig)
    if not args.no_paper_copy:
        shutil.copy(fig, PAPER_FIG / "spo_vs_gbt_rowcol_map_depths.pdf")
        if str(args.pgf):
            write_pgf(tab, args.pgf)
    print(tab.to_string(index=False))
    print(f"\nwrote {fig} (+ .png), {args.out_dir / 'rowcol_map_depths.csv'}")


if __name__ == "__main__":
    main()
