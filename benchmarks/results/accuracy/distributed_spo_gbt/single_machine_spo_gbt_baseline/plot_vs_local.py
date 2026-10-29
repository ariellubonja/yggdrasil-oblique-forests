# Column-replication charts with the single-machine SPO-GBT baseline (not for the paper).
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
here = Path(__file__).parent
root = here.parents[2]
v = pd.read_csv(root / "benchmarks/results/accuracy/distributed_spo_gbt/column_replication_vertex/results.csv")
l = pd.read_csv(here / "results.csv")
v, l = v[v.rc == 0], l[l.rc == 0]
m = v.groupby(["dataset", "replication"]).test_acc.mean().unstack()
loc = l.groupby("dataset").test_acc.mean().reindex(m.index)
assert loc.notna().all()
R = m.columns.values
d = m.sub(m[0.0], axis=0) * 100
dl = (loc - m[0.0]) * 100
gap = m.sub(loc, axis=0) * 100
w = lambda s: f"{(s > 1e-9).sum()}/{(s.abs() <= 1e-9).sum()}/{(s < -1e-9).sum()}"
print(f"local mean acc {loc.mean():.4f}; r=0 {m[0.0].mean():.4f}; local - r0 = {dl.mean():+.2f} pts (W/T/L {w(dl)})")
print(pd.DataFrame({"dist - local (pts)": gap.mean().round(2), "W/T/L vs local": [w(gap[c]) for c in R]}))
rng = np.random.default_rng(0); idx = rng.integers(0, len(d), (10000, len(d)))
ci = lambda x: np.percentile(x.values[idx].mean(1), [2.5, 97.5])
BLUE, ORANGE, GREEN, MUTED, INK = "#2a78d6", "#eb6834", "#008300", "#52514e", "#0b0b0b"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "savefig.dpi": 200})
XL = 1.2
fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.2, 5.0), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.15]))
lo, hi = np.array([ci(d[c]) for c in R]).T
a1.axhline(0, color=MUTED, lw=0.8)
a1.fill_between(R, lo, hi, color=BLUE, alpha=0.15, lw=0)
a1.plot(R, d.mean(), color=BLUE, lw=2, marker="o", ms=4, label="distributed, $W{=}4$")
llo, lhi = ci(dl)
a1.axhline(dl.mean(), color=GREEN, lw=1.2, ls="--")
a1.errorbar([XL], [dl.mean()], yerr=[[dl.mean() - llo], [lhi - dl.mean()]], color=GREEN, marker="s", ms=5, capsize=3,
            label="single-machine SPO-GBT")
a1.set_ylabel(r"$\Delta$ mean acc. vs $r{=}0$ (pts)"); a1.legend(frameon=False, fontsize=7, loc="upper left")
a2.axhline(0, color=MUTED, lw=0.8)
pos = list(R[1:]) + [XL]; data = [d[c] for c in R[1:]] + [dl]
bp = a2.boxplot(data, positions=pos, widths=0.06, showfliers=False, medianprops=dict(color=INK))
for b in bp["boxes"][-1:]: b.set_color(GREEN)
for x, s in zip(pos, data):
    a2.scatter(x + rng.uniform(-0.025, 0.025, len(s)), s, s=5, color=GREEN if x == XL else BLUE, alpha=0.5, lw=0, zorder=3)
a2.plot(R[1:], d.mean()[1:], color=ORANGE, marker="D", ms=3, lw=1, label="mean")
a2.plot([XL], [dl.mean()], color=ORANGE, marker="D", ms=3)
a2.axhline(dl.mean(), color=GREEN, lw=1.2, ls="--")
a2.set_ylim(-5, 5); a2.set_ylabel(r"per-dataset $\Delta$ (pts)"); a2.set_xlabel("column replication $r$")
a2.set_xticks(pos); a2.set_xticklabels([f"{x:.1f}" for x in R[1:]] + ["local"], fontsize=7)
a2.set_xlim(-0.05, XL + 0.07); a1.set_xticks([0] + pos); a2.set_xticks([0] + pos)
a2.set_xticklabels(["0.0"] + [f"{x:.1f}" for x in R[1:]] + ["local"], fontsize=7)
fig.text(0.01, 0.005, f"pts = percentage points of test accuracy; bottom clipped to ±5 pts", fontsize=6, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(here / "vs_local_delta_r0.png"); plt.close(fig)
# Version 2: gap to the local baseline
fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.2, 5.0), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.15]))
lo, hi = np.array([ci(gap[c]) for c in R]).T
a1.axhline(0, color=GREEN, lw=1.2, ls="--", label="single-machine SPO-GBT")
a1.fill_between(R, lo, hi, color=BLUE, alpha=0.15, lw=0)
a1.plot(R, gap.mean(), color=BLUE, lw=2, marker="o", ms=4, label="distributed, $W{=}4$")
a1.set_ylabel("mean acc. − local (pts)"); a1.legend(frameon=False, fontsize=7, loc="lower right")
a2.axhline(0, color=GREEN, lw=1.2, ls="--")
a2.boxplot([gap[c] for c in R], positions=R, widths=0.06, showfliers=False, medianprops=dict(color=INK))
for c in R: a2.scatter(c + rng.uniform(-0.025, 0.025, len(gap)), gap[c], s=5, color=BLUE, alpha=0.5, lw=0, zorder=3)
a2.plot(R, gap.mean(), color=ORANGE, marker="D", ms=3, lw=1)
a2.set_ylim(-5, 5); a2.set_ylabel("per-dataset acc. − local (pts)"); a2.set_xlabel("column replication $r$"); a2.set_xticks(R); a2.set_xlim(-0.06, 1.06)
a2.set_xticklabels([f"{x:.1f}" for x in R], fontsize=7)
fig.text(0.01, 0.005, "pts = percentage points of test accuracy; bottom clipped to ±5 pts", fontsize=6, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(here / "vs_local_gap.png"); plt.close(fig)
print("outside ±5 (gap):", int((gap.abs() > 5).sum().sum()))
