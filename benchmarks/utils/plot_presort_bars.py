#!/usr/bin/env python3
"""Grouped bar chart: in_node vs force_presorted vs auto, one group per (N, D)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV = "benchmarks/results/presort_breakeven_full_alienware.csv"
OUT = "benchmarks/results/presort_breakeven_bars.png"

df = pd.read_csv(CSV)
df = df[df["median_s"] != "N/A"].copy()
df["median_s"] = df["median_s"].astype(float)

inode = df[df["strategy"] == "in_node"].set_index(["rows", "cols"])["median_s"]
presort = df[df["strategy"] == "force_presorted"].set_index(["rows", "cols"])["median_s"]
auto = df[df["strategy"] == "auto"].set_index(["rows", "cols"])["median_s"]

keys = sorted(set(inode.index) & set(presort.index) & set(auto.index),
              key=lambda nd: (nd[1], nd[0]))  # sort by D then N

print(f"cells with all 3 strategies: {len(keys)}")

strategies = ["in_node", "auto", "force_presorted"]
colors = {"in_node": "#3a8ddc", "auto": "#67b04f", "force_presorted": "#d6584d"}

bar_w = 0.26
group_gap = 0.4  # extra gap between successive D groups

x = []
xpos = 0.0
labels = []
prev_d = None
group_centers_by_d = {}
for (n, d) in keys:
    if prev_d is not None and d != prev_d:
        xpos += group_gap
    x.append(xpos)
    labels.append(f"N={n:,}")
    group_centers_by_d.setdefault(d, []).append(xpos)
    prev_d = d
    xpos += 1.0
x = np.array(x)

fig, ax = plt.subplots(figsize=(18, 6.5))
for k, strat in enumerate(strategies):
    series = {"in_node": inode, "force_presorted": presort, "auto": auto}[strat]
    vals = [series.loc[k] for k in keys]
    offset = (k - 1) * bar_w
    bars = ax.bar(x + offset, vals, width=bar_w,
                  label=strat, color=colors[strat], edgecolor="black", linewidth=0.3)
    for xi, vi in zip(x + offset, vals):
        ax.text(xi, vi, f"{vi:.2g}", ha="center", va="bottom",
                fontsize=6.5, rotation=0)

ax.set_yscale("log")
ax.set_ylabel("Training time (s, log scale)")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.grid(axis="y", which="both", linestyle=":", linewidth=0.4, alpha=0.6)

# D-group annotations above the plot
ymax = max(presort.loc[k] for k in keys)
ytop = ymax * 4
for d, centers in group_centers_by_d.items():
    cx = (min(centers) + max(centers)) / 2.0
    ax.text(cx, ytop, f"D={d}", ha="center", va="bottom",
            fontsize=11, fontweight="bold")
    ax.axvspan(min(centers) - 0.5, max(centers) + 0.5,
               color="gray", alpha=0.04, zorder=0)

ax.set_ylim(top=ytop * 1.8)
ax.legend(loc="upper left", framealpha=0.9)

ax.set_title(
    "Random Forest, Axis-Aligned + Exact numerical splits — "
    "per-cell training time by sorting strategy\n"
    "Default bagging, num_trees=30, 3-run median, Alienware (6 P-cores). "
    "Cells grouped by D, sorted by N within each group.",
    fontsize=11,
)

plt.tight_layout()
plt.savefig(OUT, dpi=140)
print(f"wrote {OUT}")
