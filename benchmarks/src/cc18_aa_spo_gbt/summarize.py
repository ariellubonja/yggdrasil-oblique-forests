#!/usr/bin/env python3
"""Summarize run_cc18_aa_spo_gbt.sh results.csv: per-dataset 10-fold means and pooled comparisons.

  python3 summarize.py <results.csv> [--out summary.md]
Only (dataset, fold) cells that finished (rc 0) in every arm are used, so arms are compared on the same folds.
"""
import argparse

import numpy as np
import pandas as pd

ARMS = ["aa_ydf", "spo_harness", "spo_ydf", "aa_spo_ydf"]
LABEL = {"aa_ydf": "AA (YDF)", "spo_harness": "SPO (harness)", "spo_ydf": "SPO (YDF)",
         "aa_spo_ydf": "AA+SPO (YDF)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    df = df[(df.rc == 0) & df.arm.isin(ARMS)].drop_duplicates(["dataset", "fold", "arm"], keep="last")
    arms = [a for a in ARMS if a in set(df.arm)]
    full = df.groupby(["dataset", "fold"]).arm.nunique()
    keep = full[full == len(arms)].index
    df = df.set_index(["dataset", "fold"]).loc[keep].reset_index()
    lines = [f"Cells: {len(keep)} (dataset, fold) pairs complete in all {len(arms)} arms, "
             f"{df.dataset.nunique()} datasets.\n"]

    for metric, better in [("test_acc", "max"), ("test_auc", "max"), ("test_logloss", "min")]:
        per_ds = df.pivot_table(index="dataset", columns="arm", values=metric, aggfunc="mean")[arms]
        sign = 1 if better == "max" else -1
        ranks = (-sign * per_ds).rank(axis=1, method="average")
        lines.append(f"## {metric} (mean over folds; {'higher' if better == 'max' else 'lower'} is better)\n")
        lines.append("| arm | pooled mean | mean rank | W/T/L vs AA | mean Δ vs AA |")
        lines.append("|---|---|---|---|---|")
        for a in arms:
            d = per_ds[a] - per_ds["aa_ydf"]
            w, t, l = (sign * d > 1e-9).sum(), (d.abs() <= 1e-9).sum(), (sign * d < -1e-9).sum()
            lines.append(f"| {LABEL[a]} | {per_ds[a].mean():.4f} | {ranks[a].mean():.2f} | "
                         f"{w}/{t}/{l} | {d.mean():+.4f} |")
        if metric == "test_acc" and {"aa_spo_ydf", "spo_ydf"} <= set(arms):
            d = per_ds["aa_spo_ydf"] - per_ds["spo_ydf"]
            lines.append(f"\nAA+SPO vs SPO (YDF): W/T/L {(d > 1e-9).sum()}/{(d.abs() <= 1e-9).sum()}/"
                         f"{(d < -1e-9).sum()}, mean Δ {d.mean():+.4f}\n")
        if metric == "test_acc":
            lines.append("### Per dataset (accuracy, mean over folds)\n")
            lines.append("| dataset | " + " | ".join(LABEL[a] for a in arms) + " |")
            lines.append("|---" * (len(arms) + 1) + "|")
            for ds, row in per_ds.iterrows():
                lines.append(f"| {ds} | " + " | ".join(f"{row[a]:.4f}" for a in arms) + " |")
        lines.append("")

    nt = df.pivot_table(index="dataset", columns="arm", values="num_trees", aggfunc="mean")[arms]
    ts = df.pivot_table(index="dataset", columns="arm", values="train_s", aggfunc="sum")[arms]
    lines.append("## Trees kept (mean) and total train seconds\n")
    lines.append("| arm | mean trees | median trees | total train_s |")
    lines.append("|---|---|---|---|")
    for a in arms:
        lines.append(f"| {LABEL[a]} | {nt[a].mean():.1f} | {np.median(nt[a]):.1f} | {ts[a].sum():.0f} |")
    text = "\n".join(lines) + "\n"
    if args.out:
        open(args.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
