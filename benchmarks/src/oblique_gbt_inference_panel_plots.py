#!/usr/bin/env python3
"""Across-dataset summary plots: 2-panel and 4-panel.

Style ported from origin/oblique_gbt:plot_{2,4}panel.py.

For each (dataset, depth, split) combination in sweep_v3.csv, compute
mean accuracy and mean inference µs/example across the 10 seeds. Pair
axis-aligned vs oblique by (dataset, depth) on matching depths.

Outputs (next to the input CSV):
  - <stem>_summary_2panel.png — boxplots: acc Δ + speed ratio per depth
  - <stem>_summary_4panel.png — scatter + 2 boxplots + Pareto sample
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def aggregate(csv_path: Path) -> pd.DataFrame:
    """Mean across seeds per (dataset, split_type, max_depth). Returns
    a wide frame with one row per (dataset, depth) pair, with axis_*
    and oblique_* columns; only keeps depths present in BOTH arms."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["accuracy", "generic_us_per_ex"])
    grp = (df.groupby(["dataset", "split_type", "max_depth"])
             .agg(acc_mean=("accuracy", "mean"),
                  acc_std=("accuracy", "std"),
                  gen_us_mean=("generic_us_per_ex", "mean"),
                  gen_us_std=("generic_us_per_ex", "std"),
                  best_us_mean=("best_us_per_ex", "mean"),
                  n_features=("n_features", "first"),
                  n_seeds=("seed", "nunique"))
             .reset_index())
    aa = grp[grp.split_type == "Axis Aligned"].drop(columns="split_type")
    obl = grp[grp.split_type == "Oblique"].drop(columns="split_type")
    aa = aa.rename(columns={c: f"aa_{c}" for c in aa.columns
                             if c not in ("dataset", "max_depth")})
    obl = obl.rename(columns={c: f"obl_{c}" for c in obl.columns
                                if c not in ("dataset", "max_depth")})
    paired = aa.merge(obl, on=["dataset", "max_depth"], how="inner")
    paired["acc_delta"] = paired["obl_acc_mean"] - paired["aa_acc_mean"]
    # Speed ratio: oblique speed (samples/sec) / aa speed.
    # samples_per_sec = 1e6 / us_per_ex, so ratio = aa_us / obl_us.
    paired["speed_ratio"] = (paired["aa_gen_us_mean"]
                              / paired["obl_gen_us_mean"])
    return paired


def plot_2panel(paired: pd.DataFrame, out_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    depths = sorted(paired["max_depth"].unique())

    # Panel 1: accuracy delta
    data_acc = [paired[paired["max_depth"] == d]["acc_delta"].values
                 for d in depths]
    bp1 = ax1.boxplot(data_acc, positions=depths, widths=0.5,
                       patch_artist=True)
    for box in bp1["boxes"]:
        box.set_facecolor("#7fbfff")
    ax1.axhline(0, color="k", ls="--", lw=0.8)
    ax1.set_xlabel("Tree depth")
    ax1.set_ylabel("Oblique accuracy − Axis-aligned accuracy")
    ax1.set_title("Accuracy difference per depth (across 25 datasets)")
    ax1.text(0.02, 0.95, "> 0: oblique better\n< 0: AA better",
             fontsize=9, style="italic", transform=ax1.transAxes,
             verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    _label_outliers(ax1, paired, "acc_delta", depths)

    # Panel 2: speed ratio
    data_speed = [paired[paired["max_depth"] == d]["speed_ratio"].values
                   for d in depths]
    bp2 = ax2.boxplot(data_speed, positions=depths, widths=0.5,
                       patch_artist=True)
    for box in bp2["boxes"]:
        box.set_facecolor("#ffb07f")
    ax2.axhline(1, color="k", ls="--", lw=0.8)
    ax2.axhline(1.25, color="g", ls=":", lw=0.8, label="20% gate")
    ax2.set_xlabel("Tree depth")
    ax2.set_ylabel("Oblique speed / Axis-aligned speed")
    ax2.set_title("Inference speed ratio per depth")
    ax2.text(0.02, 0.95, "> 1: oblique faster\n< 1: oblique slower",
             fontsize=9, style="italic", transform=ax2.transAxes,
             verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    ax2.legend(loc="upper right", fontsize=8)
    _label_outliers(ax2, paired, "speed_ratio", depths)

    fig.suptitle("GBT Oblique vs Axis-Aligned: 25 OpenML datasets × 10 seeds "
                 "(apples-to-apples Generic engine)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out_path}")


def _label_outliers(ax, paired, col, depths):
    for d in depths:
        sub = paired[paired["max_depth"] == d]
        vals = sub[col].values
        if len(vals) < 4:
            continue
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        for _, row in sub.iterrows():
            v = row[col]
            if v < lo or v > hi:
                ax.annotate(row["dataset"][:18], (d, v),
                             fontsize=6.5, ha="left",
                             textcoords="offset points", xytext=(6, 0),
                             color="#444")


def plot_4panel(paired: pd.DataFrame, df_long: pd.DataFrame,
                out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    depths = sorted(paired["max_depth"].unique())

    # 1. scatter: oblique vs aa accuracy, colored by depth
    ax = axes[0, 0]
    sc = ax.scatter(paired["aa_acc_mean"], paired["obl_acc_mean"],
                     c=paired["max_depth"], cmap="viridis", s=30,
                     alpha=0.85, edgecolor="k", linewidth=0.3)
    lo = min(paired["aa_acc_mean"].min(), paired["obl_acc_mean"].min())
    hi = max(paired["aa_acc_mean"].max(), paired["obl_acc_mean"].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_xlabel("AA accuracy")
    ax.set_ylabel("Oblique accuracy")
    ax.set_title("Oblique vs AA accuracy (color = depth)")
    ax.text(0.02, 0.95, "Above diagonal: oblique wins on accuracy",
             fontsize=8, style="italic", transform=ax.transAxes,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    fig.colorbar(sc, ax=ax, label="depth")

    # 2. accuracy delta boxplot
    ax = axes[0, 1]
    data_acc = [paired[paired["max_depth"] == d]["acc_delta"].values
                 for d in depths]
    bp = ax.boxplot(data_acc, positions=depths, widths=0.6,
                     patch_artist=True)
    for box in bp["boxes"]:
        box.set_facecolor("#7fbfff")
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_title("Accuracy difference per depth")
    ax.set_xlabel("Tree depth")
    ax.set_ylabel("Obl − AA accuracy")
    ax.text(0.02, 0.95, "> 0: oblique better\n< 0: AA better",
             fontsize=8, style="italic", transform=ax.transAxes,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    # 3. speed ratio boxplot
    ax = axes[1, 0]
    data_speed = [paired[paired["max_depth"] == d]["speed_ratio"].values
                   for d in depths]
    bp = ax.boxplot(data_speed, positions=depths, widths=0.6,
                     patch_artist=True)
    for box in bp["boxes"]:
        box.set_facecolor("#ffb07f")
    ax.axhline(1, color="k", ls="--", lw=0.8)
    ax.axhline(1.25, color="g", ls=":", lw=0.8, label="20% gate")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Inference speed ratio per depth")
    ax.set_xlabel("Tree depth")
    ax.set_ylabel("Obl speed / AA speed")
    ax.text(0.02, 0.95, "> 1: oblique faster\n< 1: oblique slower",
             fontsize=8, style="italic", transform=ax.transAxes,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    # 4. Pareto curves for a representative subset of datasets — pick
    # ones spanning very-high-dim, mid-dim, low-dim and at least one
    # winner + loser for visual variety.
    ax = axes[1, 1]
    repr_dsets = [
        "internet-advertisements",  # big oblique win, 1558 feat
        "spambase",                  # mid-dim oblique win, 57 feat
        "diabetes",                  # low-dim oblique win, 8 feat
        "wdbc",                      # mid-dim oblique win, 30 feat
        "madelon",                   # high-dim oblique loss, 500 feat
        "monks-problems-2",          # synthetic XOR, oblique loss
    ]
    colors = plt.cm.tab10(np.linspace(0, 1, len(repr_dsets)))
    for i, ds in enumerate(repr_dsets):
        sub = df_long[df_long["dataset"] == ds].copy()
        if sub.empty:
            continue
        sub_grp = (sub.groupby(["split_type", "max_depth"])
                       .agg(acc=("accuracy", "mean"),
                            us=("generic_us_per_ex", "mean"))
                       .reset_index())
        for split, marker, ls in [("Axis Aligned", "o", "-"),
                                   ("Oblique", "s", "--")]:
            sub_split = sub_grp[sub_grp.split_type == split].sort_values(
                "max_depth")
            if sub_split.empty:
                continue
            samp_per_sec = 1e6 / sub_split["us"]
            ax.plot(samp_per_sec, sub_split["acc"], marker=marker,
                     color=colors[i], ls=ls, ms=6, lw=1.5,
                     label=(ds[:24] if split == "Axis Aligned" else None))
    ax.set_xscale("log")
    ax.set_xlabel("Inference speed (samples/sec, log)")
    ax.set_ylabel("Accuracy (10-seed mean)")
    ax.set_title("Pareto: 6 representative datasets")
    style_handles = [
        Line2D([0], [0], color="gray", marker="o", ls="-",
               label="Axis Aligned"),
        Line2D([0], [0], color="gray", marker="s", ls="--",
               label="Oblique"),
    ]
    dataset_handles = ax.get_legend_handles_labels()[0]
    ax.legend(handles=style_handles + dataset_handles, fontsize=7,
               loc="lower left", ncol=1)

    fig.suptitle("GBT Oblique vs Axis-Aligned: 25 OpenML datasets × 10 seeds",
                 fontsize=14, y=1.005)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out_path}")


def cross_depth_winners(df_long: pd.DataFrame) -> pd.DataFrame:
    """For each dataset, find the (oblique d_o, axis d_a) pair that
    maximises the speedup at matched-or-better accuracy.

    Definition: ratio = axis_us / obl_us, considered valid when
        obl_acc(d_o) >= axis_acc(d_a) - 0.001  (oblique no worse on acc)
    AND axis_us(d_a) >= obl_us(d_o)            (otherwise axis is just
                                                 better at everything,
                                                 not a "speedup")
    Maximize ratio across all valid pairs. (Equivalent to: pick the
    most-demanding axis depth oblique can match without going slower —
    that's the strongest 'oblique wins on Pareto' statement.)
    """
    grp = (df_long.groupby(["dataset", "split_type", "max_depth"])
                  .agg(acc=("accuracy", "mean"),
                       us=("generic_us_per_ex", "mean"),
                       n_features=("n_features", "first"))
                  .reset_index())
    rows = []
    for ds, sub in grp.groupby("dataset"):
        ax = sub[sub.split_type == "Axis Aligned"]
        ob = sub[sub.split_type == "Oblique"]
        if ax.empty or ob.empty:
            continue
        best = None
        for _, o in ob.iterrows():
            cand = ax[(ax.acc <= o.acc + 0.001) & (ax.us >= o.us)]
            if cand.empty:
                continue
            most_demanding = cand.loc[cand.us.idxmax()]
            ratio = most_demanding.us / o.us
            d_acc = o.acc - most_demanding.acc
            if best is None or ratio > best["ratio"]:
                best = {"dataset": ds,
                        "n_features": int(o.n_features),
                        "obl_depth": int(o.max_depth),
                        "obl_acc": o.acc,
                        "obl_us": o.us,
                        "axis_depth": int(most_demanding.max_depth),
                        "axis_acc": most_demanding.acc,
                        "axis_us": most_demanding.us,
                        "ratio": ratio,
                        "delta_acc": d_acc}
        if best is None:
            # Oblique is dominated everywhere — report the best-acc
            # oblique vs the closest-acc axis as a "loss" data point.
            o = ob.loc[ob.acc.idxmax()]
            nearest = ax.iloc[(ax.acc - o.acc).abs().argsort()[:1]].iloc[0]
            best = {"dataset": ds,
                    "n_features": int(o.n_features),
                    "obl_depth": int(o.max_depth),
                    "obl_acc": o.acc,
                    "obl_us": o.us,
                    "axis_depth": int(nearest.max_depth),
                    "axis_acc": nearest.acc,
                    "axis_us": nearest.us,
                    "ratio": nearest.us / o.us,
                    "delta_acc": o.acc - nearest.acc}
        rows.append(best)
    return pd.DataFrame(rows).sort_values("ratio", ascending=True)


def plot_cross_depth_winners(df_long: pd.DataFrame, out_path: Path):
    winners = cross_depth_winners(df_long)
    fig, ax = plt.subplots(figsize=(11, 9))
    colors = ["#2ca02c" if r >= 1.25
              else "#ffa726" if r >= 1.0
              else "#d62728"
              for r in winners["ratio"]]
    ypos = np.arange(len(winners))
    bars = ax.barh(ypos, winners["ratio"], color=colors,
                    edgecolor="k", linewidth=0.4)
    ax.axvline(1.0, color="k", ls="--", lw=0.8, label="parity")
    ax.axvline(1.25, color="#2ca02c", ls=":", lw=1.0,
               label="20% gate (≥ 1.25×)")
    ax.set_yticks(ypos)
    labels = [
        f"{r['dataset']:<25} (d{r['obl_depth']}↔{r['axis_depth']}, "
        f"{r['n_features']:>4} feat, Δacc {r['delta_acc']:+.3f})"
        for _, r in winners.iterrows()
    ]
    ax.set_yticklabels(labels, family="monospace", fontsize=9)
    ax.set_xlabel("Speedup at matched-or-better accuracy "
                   "(axis µs/example  ÷  oblique µs/example)")
    ax.set_title("Cross-depth oblique-vs-axis Pareto winner per dataset\n"
                  "10-seed mean, apples-to-apples Generic engine")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    # Annotate ratios on bars
    for bar, r in zip(bars, winners["ratio"]):
        ax.text(r + 0.02, bar.get_y() + bar.get_height()/2,
                 f"{r:.2f}×", va="center", fontsize=8)
    fig.text(0.99, 0.01,
              f"green: oblique passes 20% gate ({(winners['ratio']>=1.25).sum()}/"
              f"{len(winners)})  ·  "
              f"orange: oblique faster but < 20%  ·  "
              f"red: axis faster",
              ha="right", fontsize=8, style="italic", color="#444")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out_path}")
    return winners


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="sweep_v3.csv (long format)")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    paired = aggregate(csv_path)
    df_long = pd.read_csv(csv_path).dropna(
        subset=["accuracy", "generic_us_per_ex"])

    stem = csv_path.stem
    plot_2panel(paired, out_dir / f"{stem}_summary_2panel.png")
    plot_4panel(paired, df_long,
                 out_dir / f"{stem}_summary_4panel.png")
    winners = plot_cross_depth_winners(
        df_long, out_dir / f"{stem}_cross_depth_winners.png")
    winners.to_csv(out_dir / f"{stem}_cross_depth_winners.csv",
                    index=False)

    # Quick text summary to stdout — useful sanity check.
    print()
    print(f"Paired (dataset, depth) rows: {len(paired)}")
    print(f"Datasets: {paired['dataset'].nunique()}")
    print(f"Depths matched: {sorted(paired['max_depth'].unique())}")
    print(f"Median acc Δ across all (ds, depth): "
          f"{paired['acc_delta'].median():+.4f}")
    print(f"Median speed ratio across all (ds, depth): "
          f"{paired['speed_ratio'].median():.2f}× "
          f"(>1 = oblique faster)")
    print(f"Fraction (ds, depth) where oblique wins on speed at "
          f"matched-or-better acc:")
    wins = ((paired["acc_delta"] >= -0.001)
             & (paired["speed_ratio"] >= 1.25)).sum()
    print(f"  {wins}/{len(paired)} ({wins/len(paired)*100:.0f}%)")


if __name__ == "__main__":
    main()
