#!/usr/bin/env python3
"""Render results.csv (+ anchors.csv) of the distributed SPO-GBT test as a markdown table.

Columns: per-tree time = tree_training_s / trees (includes the workers loading their columns,
excludes dataset-cache creation); 'x slower' = per-tree time / the single-machine anchor's
train_s / trees (anchor = the single-machine arm of the same split type and tree count --
spo_gbt_exact_hwy for SPARSE_OBLIQUE, aa_gbt_exact for AXIS_ALIGNED -- from anchors.csv, same boot;
300-tree oblique rows without an anchor fall back to the 2026-09-24 speedup_map / large_results
cells); dAcc / dAUC = distributed minus anchor test metrics.
Usage: summarize.py [results.csv] [--trees N]  (--trees filters the distributed rows)
"""
import csv, pathlib, sys
H = pathlib.Path(__file__).resolve().parents[3] / "benchmarks/results/runtime"
R = H / "distributed_spo_gbt"
PAPER_TIME = {"SUSY": 637.547, "EPSILON": 96.4502, "HIGGS": 1618.82, "YOUTUBE8M": 1020.52}  # speedup_map.csv 2026-09-24
PAPER_ACC = {"HIGGS": (0.732302, 0.812333), "SUSY": (0.80309, 0.876895), "EPSILON": (0.8572, 0.934762)}  # large_results.csv

def f(x, d=3):
    try: return f"{float(x):.{d}f}"
    except (TypeError, ValueError): return "--"

ARM = {"SPARSE_OBLIQUE": "spo_gbt_exact_hwy", "AXIS_ALIGNED": "aa_gbt_exact"}
anchors = {}  # (dataset, arm, trees) -> last rc=0 row
if (R / "anchors.csv").exists():
    for r in csv.DictReader(open(R / "anchors.csv")):
        if r["rc"] == "0": anchors[(r["dataset"], r["arm"], int(r["trees"]))] = r
only_trees = int(sys.argv[sys.argv.index("--trees") + 1]) if "--trees" in sys.argv else None
args = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[i - 1] != "--trees"]
rows = [r for r in csv.DictReader(open(R / (args[0] if args else "results.csv")))
        if r["rc"] == "0" and (only_trees is None or int(r["trees"]) == only_trees)]
print("| dataset | workers | split | trees | tree-train s | s/tree | x slower vs 1-box | acc | AUC | dAcc | dAUC | anchor src |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    ds, n = r["dataset"], int(r["trees"])
    try: per = float(r["tree_training_s"]) / n
    except ValueError: per = None
    a = anchors.get((ds, ARM.get(r["split_axis"], ""), n))
    if a and a.get("train_s"):
        ref_t, ref_acc, ref_auc, src = float(a["train_s"]) / int(a["trees"]), a["test_acc"], a["test_auc"], "same-boot"
    elif n == 300 and r["split_axis"] == "SPARSE_OBLIQUE":
        ref_t = PAPER_TIME.get(ds, None); ref_t = ref_t / 300 if ref_t else None
        ref_acc, ref_auc = PAPER_ACC.get(ds, (None, None)); src = "paper 09-24"
    else:
        ref_t, ref_acc, ref_auc, src = None, None, None, "none"
    slow = f"{per / ref_t:.1f}x" if (per and ref_t) else "--"
    dacc = f"{float(r['test_acc']) - float(ref_acc):+.4f}" if (r.get("test_acc") and ref_acc) else "--"
    dauc = f"{float(r['test_auc']) - float(ref_auc):+.4f}" if (r.get("test_auc") and ref_auc) else "--"
    print(f"| {ds} | {r['workers']} | {r['split_axis'].replace('SPARSE_OBLIQUE','oblique').replace('AXIS_ALIGNED','axis')} | {n} | {f(r['tree_training_s'],1)} | {f(per,2)} | {slow} | {f(r['test_acc'],4)} | {f(r['test_auc'],4)} | {dacc} | {dauc} | {src} |")
if anchors:
    print("\nAnchors (single-machine paper binary, 48 threads, same boot):")
    for (ds, arm, n), a in anchors.items():
        print(f"- {ds} {arm} T={n}: train_s {f(a['train_s'],1)} ({f(float(a['train_s'])/n,2)} s/tree), acc {f(a['test_acc'],4)}, AUC {f(a['test_auc'],4)}")
