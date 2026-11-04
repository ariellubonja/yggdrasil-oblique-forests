#!/usr/bin/env python3
"""Paper vs fixed-rerun accuracy for KDD28 Tables 2, 3 and 8, and distributed-vs-single parity (paired by dataset).

  python compare_fixed_reruns.py [--results <dir>] [--out <dir>]
<dir> defaults to benchmarks/results/accuracy/distributed_spo_gbt/fixed_reruns_2026-11-01 (layout: its README.md).
Paper Table 8 cells come from benchmarks/results/runtime/{speedup_map_by_dataset,distributed_spo_gbt}/suite_results.csv.
A rerun folder that is missing is skipped, so this also checks a partial replication.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../../.."))
sys.path.insert(0, os.path.dirname(HERE))
import compare_table8_local  # noqa: E402

T8_ARMS = ["1:SPO", "4:SPO", "8:SPO", "16:SPO", "32:SPO"]


def strip_task(s):
    return s.str.replace(r"^task_\d+_", "", regex=True)


def paired(delta):
    delta = delta.dropna()
    return delta.mean(), delta.std() / np.sqrt(len(delta)), len(delta)


def table2(res, out):
    p = os.path.join(res, "paper_refs")
    rows = pd.read_csv(f"{p}/table2_summary_pooled.csv")[["replication", "acc_mean"]].rename(columns={"acc_mean": "paper"})
    runs = {}
    for name in ["binfix", "fullfix", "fullfix_splitbudget"]:
        f = f"{res}/table2/{name}/results.csv"
        if os.path.exists(f):
            d = pd.read_csv(f)
            d["dataset"] = strip_task(d.dataset)
            runs[name] = d
            m = d.groupby(["replication", "dataset"]).test_acc.mean().groupby("replication").mean().rename(name)
            rows = rows.merge(m.reset_index(), on="replication", how="left")
    print("== Table 2: mean test accuracy over 34 CC18 datasets (5 seeds each)")
    print(rows.round(4).to_string(index=False))
    rows.to_csv(f"{out}/table2_paper_vs_fixed.csv", index=False)

    single = pd.read_csv(f"{p}/table2_single_machine_spo_gbt.csv")
    single["dataset"] = strip_task(single.dataset)
    sm = single.groupby("dataset")[["test_acc", "test_logloss"]].mean()
    paper = pd.read_csv(f"{p}/table2_results.csv")
    paper["dataset"] = strip_task(paper.dataset)
    out_rows = []
    print(f"\n== Table 2 parity vs single-machine SPO-GBT (mean acc {sm.test_acc.mean():.4f}), paired by dataset")
    for name, d in [("paper", paper)] + list(runs.items()):
        for r in [0.0, 1.0]:
            x = d[np.isclose(d.replication, r)].groupby("dataset")[["test_acc", "test_logloss"]].mean().join(sm, rsuffix="_s")
            a, ase, n = paired((x.test_acc - x.test_acc_s) * 100)
            ll, llse, _ = paired(x.test_logloss - x.test_logloss_s)
            out_rows.append((name, r, n, a, ase, ll, llse))
            print(f"  {name:20s} r={r:.0f}  dacc {a:+.3f} +- {ase:.3f} pts   dlogloss {ll:+.4f} +- {llse:.4f}  (n={n})")
    pd.DataFrame(out_rows, columns=["run", "replication", "n", "dacc_pts", "se", "dlogloss", "se_ll"]).to_csv(
        f"{out}/table2_parity_vs_single.csv", index=False)


def table3(res, out):
    def load(pattern):
        files = glob.glob(pattern)
        if not files:
            return None
        d = pd.concat(pd.read_csv(f) for f in files)
        d["dataset"] = strip_task(d.dataset)
        d["arm"] = d.replication.astype(str).map(lambda r: "cross_network" if r == "ship" else "in_worker")
        return d
    paper = pd.read_csv(f"{res}/paper_refs/table3_results.csv")
    cols = {"paper": paper.groupby("arm").test_acc.mean(), "time_paper_s": paper.groupby("arm").tree_training_s.mean()}
    for name in ["binfix", "fullfix"]:
        d = load(f"{res}/table3/{name}/g*/results.csv")
        if d is not None:
            cols[name] = d.groupby("arm").test_acc.mean()
            cols[f"time_{name}_s"] = d.groupby("arm").tree_training_s.mean()
    t = pd.DataFrame(cols)
    print("\n== Table 3: mean test accuracy and tree-training time over 34 CC18 datasets (W=4, 300 trees)")
    print(t.round(4).to_string())
    t.to_csv(f"{out}/table3_paper_vs_fixed.csv")


def table8(res, out):
    rec = compare_table8_local.recorded()
    k = ["dataset", "fold", "arm"]
    m = rec.rename(columns={"test_auc": "paper"})[k + ["paper"]]
    names = ["binfix", "binfix_numeric", "fullfix", "fullfix_numeric", "fullfix_rep", "parity"]
    for name in names:
        f = f"{res}/table8/{name}/results.csv"
        if os.path.exists(f):
            d = pd.read_csv(f)
            d = d[d.rc == 0].drop_duplicates(k, keep="last")
            m = m.merge(d[k + ["test_auc"]].rename(columns={"test_auc": name}), on=k, how="outer")
    present = [n for n in names if n in m]
    g = m[m.arm.isin(T8_ARMS)].groupby(["arm", "dataset"])[["paper"] + present].mean().groupby("arm").mean()
    single = rec[rec.arm == "single:SPO"].groupby("dataset").test_auc.mean()
    print(f"\n== Table 8: mean test AUC over 33 datasets (single-machine SPO {single.mean():.4f})")
    print(g.reindex(T8_ARMS).round(4).to_string())
    g.reindex(T8_ARMS).to_csv(f"{out}/table8_paper_vs_fixed.csv")
    if "fullfix" in m and "fullfix_rep" in m:
        x = m.dropna(subset=["fullfix", "fullfix_rep"])
        print(f"  determinism: fullfix vs fullfix_rep identical in {int((x.fullfix == x.fullfix_rep).sum())}/{len(x)} cells")
    rows = []
    print("  parity vs single-machine SPO, paired by dataset (dAUC +- se):")
    for name in ["paper"] + present:
        for a in T8_ARMS:
            x = m[m.arm == a].groupby("dataset")[name].mean()
            if x.notna().sum() == 0:
                continue
            d, se, n = paired(x - single)
            rows.append((name, a, n, d, se))
            print(f"    {name:16s} {a:7s} {d:+.4f} +- {se:.4f}  (n={n})")
    pd.DataFrame(rows, columns=["run", "arm", "n", "dauc", "se"]).to_csv(f"{out}/table8_parity_vs_single.csv", index=False)
    m.to_csv(f"{out}/table8_cells.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(
        REPO, "benchmarks/results/accuracy/distributed_spo_gbt/fixed_reruns_2026-11-01"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out = args.out or os.path.join(args.results, "comparisons")
    os.makedirs(out, exist_ok=True)
    table2(args.results, out)
    table3(args.results, out)
    table8(args.results, out)


if __name__ == "__main__":
    main()
