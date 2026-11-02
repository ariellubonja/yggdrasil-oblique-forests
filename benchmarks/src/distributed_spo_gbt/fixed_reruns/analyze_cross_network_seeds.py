#!/usr/bin/env python3
"""5-seed cross-network (Table tab:cross-network) analysis: seed 1 vs 5-seed mean per dataset, seed spread of the
headline numbers, paired tests over datasets (in-worker vs cross-network). Needs scipy (~/cc18_venv).
  analyze_cross_network_seeds.py <table3_results_5seed.csv> [--out_md F]"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats


def gm(x):
    return float(np.exp(np.mean(np.log(x))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("--out_md")
    a = ap.parse_args()
    d = pd.read_csv(a.csv)
    w = d.pivot_table(index=["dataset", "seed"], columns="arm", values=["tree_training_s", "test_acc"])
    s = pd.DataFrame({"t_in": w.tree_training_s.in_worker, "t_x": w.tree_training_s.cross_network,
                      "acc_in": w.test_acc.in_worker, "acc_x": w.test_acc.cross_network}).reset_index()
    s["sp"] = s.t_in / s.t_x
    m = s.groupby("dataset")[["t_in", "t_x", "acc_in", "acc_x"]].mean()
    m["sp"] = m.t_in / m.t_x  # ratio of seed-mean times, as in the table body
    sd = s.groupby("dataset")[["acc_in", "acc_x", "sp"]].std()
    s1 = s[s.seed == 1].set_index("dataset")
    per = pd.DataFrame({"sp_s1": s1.sp, "sp_5": m.sp, "sp_sd": sd.sp, "acc_in_s1": s1.acc_in, "acc_in_5": m.acc_in,
                        "acc_in_sd": sd.acc_in, "acc_x_s1": s1.acc_x, "acc_x_5": m.acc_x, "acc_x_sd": sd.acc_x})
    per = per.sort_values("sp_5")
    out = ["## Per dataset: seed 1 vs 5-seed mean (sd over seeds)", "",
           "| dataset | speedup s1 | speedup 5 (sd) | in-worker acc s1 | in-worker 5 (sd) | cross-net acc s1 | cross-net 5 (sd) |",
           "|---|--:|--:|--:|--:|--:|--:|"]
    for k, r in per.iterrows():
        out.append(f"| {k} | {r.sp_s1:.2f} | {r.sp_5:.2f} ({r.sp_sd:.2f}) | {r.acc_in_s1:.4f} | {r.acc_in_5:.4f} ({r.acc_in_sd:.4f}) "
                   f"| {r.acc_x_s1:.4f} | {r.acc_x_5:.4f} ({r.acc_x_sd:.4f}) |")
    by_seed = s.groupby("seed").agg(geo=("sp", gm), acc_in=("acc_in", "mean"), acc_x=("acc_x", "mean"),
                                    t_in=("t_in", "sum"), t_x=("t_x", "sum"))
    out += ["", "## Headline per seed (34 datasets)", "", "| seed | geomean speedup | mean acc in-worker | mean acc cross-net | Σt in | Σt cross |",
            "|--:|--:|--:|--:|--:|--:|"]
    for k, r in by_seed.iterrows():
        out.append(f"| {k} | {r.geo:.3f} | {r.acc_in:.4f} | {r.acc_x:.4f} | {r.t_in:.0f} | {r.t_x:.0f} |")
    out.append(f"| mean ± sd | {by_seed.geo.mean():.3f} ± {by_seed.geo.std():.3f} | {by_seed.acc_in.mean():.4f} ± "
               f"{by_seed.acc_in.std():.4f} | {by_seed.acc_x.mean():.4f} ± {by_seed.acc_x.std():.4f} | | |")
    out.append(f"| table (ratio of seed-mean times) | {gm(m.sp):.3f} | {m.acc_in.mean():.4f} | {m.acc_x.mean():.4f} | "
               f"{m.t_in.sum():.0f} | {m.t_x.sum():.0f} |")
    dacc = m.acc_x - m.acc_in
    lsp = np.log(m.sp)
    t_acc, w_acc = stats.ttest_1samp(dacc, 0), stats.wilcoxon(dacc)
    t_sp, w_sp = stats.ttest_1samp(lsp, 0), stats.wilcoxon(lsp)
    out += ["", "## Paired tests over the 34 datasets (5-seed means)", "",
            f"- Accuracy cross − in: mean {dacc.mean():+.4f} (sd {dacc.std():.4f}), cross-net better on "
            f"{(dacc > 0).sum()}, worse on {(dacc < 0).sum()}, tied {(dacc == 0).sum()}; paired t = {t_acc.statistic:.2f} "
            f"(p = {t_acc.pvalue:.3g}), Wilcoxon p = {w_acc.pvalue:.3g}",
            f"- log speedup: geomean {gm(m.sp):.3f}, < 1 on {(m.sp < 1).sum()}/34; t = {t_sp.statistic:.2f} "
            f"(p = {t_sp.pvalue:.3g}), Wilcoxon p = {w_sp.pvalue:.3g}"]
    txt = "\n".join(out) + "\n"
    if a.out_md:
        open(a.out_md, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
