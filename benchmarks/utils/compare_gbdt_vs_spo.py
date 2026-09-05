#!/usr/bin/env python3
"""Side-by-side of SPO-YDF arms and axis-aligned baselines on shared fold CSVs.

Reads any accuracy CSVs in the parse_log_to_csv shape (accuracy_hve_*.csv from
accuracy.sh, gbdt_*.csv from gbdt_baselines.py), joins on dataset, and writes a
per-dataset table + summary (mean acc, mean rank, paired delta and Wilcoxon p
vs --ref) as Markdown and CSV.  Same-name *_auc / *_logloss / *_time files are
summarised the same way when --metric asks for them.

  python benchmarks/utils/compare_gbdt_vs_spo.py --out benchmarks/results/accuracy/gbdt_vs_spo_cc18 \
      benchmarks/results/accuracy/accuracy_hve_{rf,gbt}_dyn_s1.csv benchmarks/results/accuracy/gbdt_cc18_*_s1_{xgb,lgbm,cat}.csv
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def read(path):
    with open(path) as f:
        skip = next(i for i, l in enumerate(f) if l.startswith("dataset,"))
    df = pd.read_csv(path, skiprows=skip)
    folds = [c for c in df.columns if c.startswith("fold_")]
    df["mean"] = df[folds].mean(axis=1)
    return df[["dataset", "algorithm", "mean"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--out", required=True, help="output stem (.md and .csv appended)")
    ap.add_argument("--ref", default=None, help="reference algorithm (default: first SPO-RF)")
    ap.add_argument("--metric", default="accuracy", help="label for the table")
    ap.add_argument("--lower-better", action="store_true")
    args = ap.parse_args()

    long = pd.concat([read(p) for p in args.csvs])
    wide = long.pivot_table(index="dataset", columns="algorithm", values="mean")
    wide = wide.dropna(how="any")  # only datasets every method finished
    algos = list(wide.columns)
    ref = args.ref or next((a for a in algos if a.startswith(("SPORF", "SPO-RF"))), algos[0])
    sign = -1 if args.lower_better else 1
    ranks = (sign * wide).rank(axis=1, ascending=False)
    rows = []
    for a in algos:
        d = sign * (wide[a] - wide[ref])
        p = wilcoxon(wide[a], wide[ref]).pvalue if a != ref and (d != 0).any() else float("nan")
        rows.append(dict(algorithm=a, mean=wide[a].mean(), mean_rank=ranks[a].mean(),
                         wins_vs_ref=int((d > 0).sum()), ties=int((d == 0).sum()),
                         losses_vs_ref=int((d < 0).sum()), mean_delta_vs_ref=d.mean(),
                         wilcoxon_p=p))
    summ = pd.DataFrame(rows).sort_values("mean_rank")
    wide.to_csv(args.out + ".csv")
    summ.to_csv(args.out + "_summary.csv", index=False)
    with open(args.out + ".md", "w") as f:
        f.write(f"# {args.metric}: {len(wide)} datasets, ref = {ref}\n\n")
        f.write(summ.to_markdown(index=False, floatfmt=".4f") + "\n\n")
        f.write(wide.round(4).to_markdown() + "\n")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nwrote {args.out}.md / .csv / _summary.csv")


if __name__ == "__main__":
    main()
