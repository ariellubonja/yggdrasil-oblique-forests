#!/usr/bin/env python3
"""Old (2026-10-25) vs fixed-image (2026-11-01) Vertex P=F grid: tree_training_s, setup_s and accuracy per
(dataset, machine, W, arm). AA changed only through the harness (checkpoints off) and unrelated code, so the AA ratio
is the harness/noise baseline; spo_vs_aa = SPO ratio / AA ratio of the same (dataset, machine, W).
  compare_runtime_grid.py [--out_csv F] [--out_md F]"""
import argparse
import glob
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
K = os.path.join(HERE, "../../../results/kdd28")
OLD, NEW = os.path.join(K, "outdated/vertex_p_F_prefix_20261025"), os.path.join(K, "cells/vertex_p_F")
GRIDS = [("n2-highmem-4", "pfhm4_20261025", "pfhm4fix20261101"), ("n2-highmem-16", "pfgrid20261025", "pfgridfix20261101")]


def load(prefix, machine):
    out = {}
    pf = NEW if "fix" in prefix else OLD
    for p in glob.glob(os.path.join(pf, f"{prefix}_*", "*", "eval.json")):
        e = json.load(open(p))
        if e.get("machine_type") == machine and e.get("trees") == 10 and "tree_training_s" in e:
            out[(e["dataset"], e["arm"], e["workers"])] = e
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv"); ap.add_argument("--out_md")
    a = ap.parse_args()
    rows = []
    for machine, old_p, new_p in GRIDS:
        old, new = load(old_p, machine), load(new_p, machine)
        for k in sorted(set(old) | set(new)):
            o, n = old.get(k, {}), new.get(k, {})
            assert not (o and n) or o["p_total"] == n["p_total"], (k, o.get("p_total"), n.get("p_total"))
            rows.append(dict(dataset=k[0], machine=machine, W=k[2], arm=k[1],
                             old_t=o.get("tree_training_s"), new_t=n.get("tree_training_s"),
                             old_setup=o.get("setup_s"), new_setup=n.get("setup_s"),
                             old_acc=o.get("accuracy"), new_acc=n.get("accuracy")))
    d = pd.DataFrame(rows)
    d["ratio"] = d.new_t / d.old_t
    d["setup_ratio"] = d.new_setup / d.old_setup
    d["d_acc"] = d.new_acc - d.old_acc
    aa = d[d.arm == "aa"].set_index(["dataset", "machine", "W"]).ratio
    d["spo_vs_aa"] = [r.ratio / aa.get((r.dataset, r.machine, r.W), float("nan")) if r.arm == "spo" else float("nan")
                      for r in d.itertuples()]
    d["flag"] = (d.ratio - 1).abs().gt(0.20).map({True: "!", False: ""})
    d = d.sort_values(["machine", "dataset", "arm", "W"], key=lambda c: c.map({"n2-highmem-4": 0, "n2-highmem-16": 1})
                      if c.name == "machine" else c)
    if a.out_csv:
        d.to_csv(a.out_csv, index=False, float_format="%.4f")
    f = lambda x, s: "--" if pd.isna(x) else format(x, s)  # noqa: E731
    md = ["| machine | dataset | arm | W | old t (s) | new t (s) | new/old | SPO/AA | old acc | new acc | flag |",
          "|---|---|---|--:|--:|--:|--:|--:|--:|--:|:-:|"]
    for r in d.itertuples():
        md.append(f"| {r.machine} | {r.dataset} | {r.arm} | {r.W} | {f(r.old_t, '.1f')} | {f(r.new_t, '.1f')} | "
                  f"{f(r.ratio, '.3f')} | {f(r.spo_vs_aa, '.3f')} | {f(r.old_acc, '.4f')} | {f(r.new_acc, '.4f')} | {r.flag} |")
    txt = "\n".join(md) + "\n"
    if a.out_md:
        open(a.out_md, "w").write(txt)
    print(txt)
    both = d.dropna(subset=["ratio"])
    print(both.groupby("arm").ratio.describe().round(3).to_string())


if __name__ == "__main__":
    main()
