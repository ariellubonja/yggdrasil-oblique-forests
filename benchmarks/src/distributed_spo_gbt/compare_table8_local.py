#!/usr/bin/env python3
"""Local Table 8 rerun vs the recorded cells: per (dataset, fold, arm) test AUC, exact-match counts and max |Δ|.

  python3 compare_table8_local.py <local results.csv>
Recorded: speedup_map_by_dataset/suite_results.csv (single-machine arms) and distributed_spo_gbt/suite_results.csv.
"""
import sys
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[2] / "results/runtime"
SINGLE = {"SPO": "spo_gbt_exact_hwy_fullproj", "AA": "aa_gbt_exact"}


def recorded():
    p = pd.read_csv(R / "speedup_map_by_dataset/suite_results.csv")
    p = p[(p.status == "OK") & p.method.isin(SINGLE.values())]
    inv = {v: k for k, v in SINGLE.items()}
    p = p.assign(arm="single:" + p.method.map(inv))[["dataset", "fold", "arm", "test_auc", "test_acc"]]
    d = pd.read_csv(R / "distributed_spo_gbt/suite_results.csv")
    d = d[d.rc == 0].drop_duplicates(["dataset", "fold", "workers", "split_axis"], keep="last")
    ax = d.split_axis.map({"SPARSE_OBLIQUE": "SPO", "AXIS_ALIGNED": "AA"})
    d = d.assign(arm=d.workers.astype(str) + ":" + ax)[["dataset", "fold", "arm", "test_auc", "test_acc"]]
    return pd.concat([p, d])


def main():
    loc = pd.read_csv(sys.argv[1])
    loc = loc[loc.rc == 0].drop_duplicates(["dataset", "fold", "arm"], keep="last")
    m = loc.merge(recorded(), on=["dataset", "fold", "arm"], suffixes=("", "_rec"))
    m["d_auc"] = (m.test_auc - m.test_auc_rec).abs()
    g = m.groupby("arm").agg(cells=("d_auc", "size"), exact=("d_auc", lambda s: int((s < 5e-7).sum())),
                             max_abs=("d_auc", "max"), mean_abs=("d_auc", "mean"))
    print(g.to_string())
    off = m[m.d_auc >= 5e-7].sort_values("d_auc", ascending=False)
    if len(off):
        print("\nlargest mismatches:\n" + off.head(15)[["dataset", "fold", "arm", "test_auc", "test_auc_rec"]].to_string(index=False))
    new = loc.merge(recorded(), on=["dataset", "fold", "arm"], how="left", indicator=True)
    print(f"\nlocal cells with no recorded counterpart (e.g. W=32): {int((new._merge == 'left_only').sum())}")


if __name__ == "__main__":
    main()
