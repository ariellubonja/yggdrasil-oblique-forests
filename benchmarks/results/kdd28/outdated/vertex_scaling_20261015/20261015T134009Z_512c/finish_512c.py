import sys, pandas as pd, matplotlib, matplotlib.ticker
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = sys.argv[1]
res = pd.read_csv(f"{D}/results.csv")
# aa_n1600k_w16: Vertex re-provisioned the job after its first run finished; the rerun resumed from the
# 50-tree checkpoint and overwrote eval.json. Restore the first run's values (read from GCS 2026-10-15 ~15:00Z).
m = (res.tag == "aa_n1600k_w16")
assert m.sum() == 1
res.loc[m, ["setup_s", "tree_training_s", "train_seconds"]] = [412.362, 2493.385, 2906.33]
res.to_csv(f"{D}/results.csv", index=False)
COL = {"aa": ("#2a78d6", "o"), "spo": ("#eb6834", "s")}
WEAK = {(100_000 * 2**k, 2**k) for k in range(5)}
weak = res[[(r, w) in WEAK for r, w in zip(res.rows, res.workers)]].sort_values("workers")
strong = res[res.rows == 1_600_000].sort_values("workers").copy()
t2 = strong[strong.workers == 2].set_index("arm")["tree_training_s"]
strong["speedup_vs_w2"] = [t2[a] / t for a, t in zip(strong.arm, strong.tree_training_s)]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for arm, g in weak.groupby("arm"):
    c, mk = COL[arm]
    ax[0].plot(g.workers, g.tree_training_s, marker=mk, color=c, lw=2, ms=8, label=f"{arm} tree training")
    ax[0].plot(g.workers, g.train_seconds, marker="x", ls="--", color=c, lw=1.5, alpha=.5, label=f"{arm} incl. setup")
ax[0].set(xscale="log", xlabel="workers (rows = 100k x workers)", ylabel="seconds", title="Weak scaling, 50 trees, 512 cols")
ax[0].set_xticks([1, 2, 4, 8, 16], [1, 2, 4, 8, 16]); ax[0].set_ylim(bottom=0); ax[0].legend(frameon=False)
for arm, g in strong.groupby("arm"):
    c, mk = COL[arm]
    ax[1].plot(g.workers, g.speedup_vs_w2, marker=mk, color=c, lw=2, ms=8, label=arm)
ax[1].plot([1, 16], [0.5, 8], "k:", lw=1, label="ideal")
ax[1].set(xscale="log", yscale="log", xlabel="workers", ylabel="speedup vs 2 workers (tree training)",
          title="Strong scaling, 1.6M x 512 (spo 1-worker cell OOM)")
ax[1].set_xticks([1, 2, 4, 8, 16], [1, 2, 4, 8, 16]); ax[1].set_yticks([.5, 1, 2, 4, 8], [.5, 1, 2, 4, 8]); ax[1].legend(frameon=False)
ax[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
for a in ax:
    a.grid(alpha=.25, lw=.5); a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{D}/scaling.png", dpi=130)
print(strong[["arm", "workers", "tree_training_s", "speedup_vs_w2"]].round(2).to_string(index=False))
