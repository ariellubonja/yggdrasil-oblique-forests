"""Overlaid base-vs-PR seed histograms for PR #409 (gcc = CI toolchain)."""
import csv
import statistics as st
from collections import defaultdict

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "results_matrix_100seeds.csv")
OUT = os.path.join(HERE, "pr409_seed_hist.png")

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BASE, PR = "#2a78d6", "#eb6834"

d = defaultdict(list)
with open(SRC) as f:
    for r in csv.DictReader(f):
        if r["compiler"] == "gcc":
            d[(r["test"], r["arm"])].append(float(r["metric"]))

# (test key, title, metric label, checked-in EXPECT_NEAR (center, tol), flagged draw, flag label)
PANELS = [
    ("AdultNWTAWeights", "Adult NoWinnerTakeAllWithWeights", "test accuracy (higher better)",
     (0.8415, 0.012), 0.8287, "internal PR draw 0.8287\n(“high accuracy loss”)"),
    ("SimPTELowerBound", "SimPTE LowerBound", "Qini (higher better)",
     (0.10889, 0.002), 0.10649, "internal PR draw 0.10649\n(“seems like noise”)"),
    ("AbaloneSPO", "Abalone SparseOblique", "test RMSE (lower better)",
     (2.054, 0.01), 2.064157, "GitHub CI PR draw 2.0642\n(failed: band edge 2.064)"),
]

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
})

fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), facecolor=SURFACE)
fig.subplots_adjust(left=0.045, right=0.99, top=0.72, bottom=0.16, wspace=0.16)

for ax, (key, title, mlabel, (center, tol), flag, flag_label) in zip(axes, PANELS):
    b, h = d[(key, "base")], d[(key, "head")]
    lo, hi = center - tol, center + tol
    xmin = min(min(b), min(h), flag, lo if key != "AdultNWTAWeights" else min(b))
    xmax = max(max(b), max(h), center)
    pad = (xmax - xmin) * 0.06
    xmin, xmax = xmin - pad, xmax + pad
    bins = np.linspace(min(min(b), min(h)), max(max(b), max(h)), 23)

    ax.set_facecolor(SURFACE)
    ax.axvspan(lo, hi, color="#000000", alpha=0.045, lw=0, zorder=0)
    for vals, col, name in ((b, BASE, "upstream master"), (h, PR, "PR #409")):
        ax.hist(vals, bins=bins, histtype="stepfilled", color=col, alpha=0.28, lw=0, zorder=2)
        ax.hist(vals, bins=bins, histtype="step", color=col, lw=1.6, zorder=3, label=name)
    ax.axvline(st.mean(b), color=BASE, lw=1.2, ls=(0, (1, 1.5)), zorder=4)
    ax.axvline(st.mean(h), color=PR, lw=1.2, ls=(0, (1, 1.5)), zorder=4)

    ymax = ax.get_ylim()[1] * 1.5
    ax.set_ylim(0, ymax)
    ax.axvline(center, color=INK2, lw=1.1, ls="--", zorder=4)
    ax.annotate(f"checked-in\nexpectation\n{center:g}", xy=(center, ymax * 0.97), ha="center",
                va="top", fontsize=8.5, color=INK2,
                xytext=(0, 0), textcoords="offset points", zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none"))
    ax.annotate(flag_label, xy=(flag, 0), xytext=(flag, ymax * 0.70), ha="center", va="bottom",
                fontsize=8.2, color=INK, zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0, shrinkB=0),
                bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none"))
    ax.set_xlim(xmin, xmax)
    diff = st.mean(h) - st.mean(b)
    t = diff / np.sqrt(st.variance(b) / len(b) + st.variance(h) / len(h))
    ax.set_title(f"{title} — {mlabel}\n"
                 f"master {st.mean(b):.4f} ± {st.stdev(b):.4f}    PR {st.mean(h):.4f} ± {st.stdev(h):.4f}\n"
                 f"PR − base = {diff:+.5f}   (t = {t:+.2f}, n = {len(b)} + {len(h)})",
                 fontsize=9.2, loc="left", color=INK, pad=8, linespacing=1.4)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_ylabel("seeds (of 100)" if ax is axes[0] else "")
    ax.tick_params(length=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)

fig.text(0.045, 0.958, "PR #409 (lazy candidate shuffle) vs upstream master — 100 training seeds each, gcc 13.3 (the GitHub CI toolchain)",
         fontsize=12.5, color=INK, weight="semibold")
fig.text(0.045, 0.905, "Distributions coincide (all |t| ≤ 0.73). Grey = current EXPECT_NEAR band; dashed = its centre. "
         "Arrows = the single default-seed draws that were read as regressions.",
         fontsize=9.5, color=INK2)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.985), ncol=2,
           frameon=False, fontsize=9.5, handlelength=1.4)
fig.text(0.045, 0.03, "Seed → train_config_.random_seed; train/test split fixed. Per-seed values: benchmarks/results/pr409_seed_study/results_matrix_100seeds.csv",
         fontsize=8, color=MUTED)
fig.savefig(OUT, dpi=160, facecolor=SURFACE)
print("wrote", OUT)
