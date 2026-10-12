#!/usr/bin/env python3
"""Table-9-style comparison: paper GBT arms (suite_results.csv, 5 CV folds / 1 chrono fold)
vs the in-worker distributed GBT (run_dgbt_suite.sh) at 1/4/8/16 workers. Test ROC AUC,
mean +- std over folds, best per row in bold (per family: SPO block, AA block).
Usage: compare_suite.py [dist_csv] [--metric test_auc] [--latex]"""
import argparse, sys
import numpy as np, pandas as pd

REPO = "/home/ubuntu/yggdrasil-oblique-forests"
PAPER = f"{REPO}/benchmarks/results/runtime/speedup_map_by_dataset/suite_results.csv"
DIST = f"{REPO}/benchmarks/results/runtime/distributed_spo_gbt/suite_results.csv"
TIE = 0.005  # analyze.py win/tie/loss threshold on AUC

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dist_csv", nargs="?", default=DIST)
    ap.add_argument("--metric", default="test_auc")
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args(); m = args.metric
    p = pd.read_csv(PAPER); p = p[p.status == "OK"]
    d = pd.read_csv(args.dist_csv); d = d[d.rc == 0]
    d = d.drop_duplicates(["dataset", "fold", "workers", "split_axis"], keep="last")
    cols = {}
    for fam, parm, dax in (("SPO", "spo_gbt_exact_hwy_fullproj", "SPARSE_OBLIQUE"),
                           ("AA", "aa_gbt_exact", "AXIS_ALIGNED")):
        cols[(fam, "paper")] = p[p.method == parm].groupby("dataset")[m]
        for w in sorted(d.workers.unique()):
            cols[(fam, f"W={w}")] = d[(d.split_axis == dax) & (d.workers == w)].groupby("dataset")[m]
    stats = {k: g.agg(["mean", "std", "count"]) for k, g in cols.items()}
    datasets = sorted(p.dataset.unique())
    fams = ["SPO", "AA"]
    arms = {f: [k[1] for k in cols if k[0] == f] for f in fams}
    hdr = ["dataset"] + [f"{f} {a}" for f in fams for a in arms[f]]
    rows = []
    for ds in datasets:
        row = [ds]
        for f in fams:
            vals = {}
            for a in arms[f]:
                s = stats[(f, a)]
                if ds in s.index and s.loc[ds, "count"] > 0:
                    vals[a] = (s.loc[ds, "mean"], s.loc[ds, "std"], int(s.loc[ds, "count"]))
            best = max((v[0] for v in vals.values()), default=None)
            for a in arms[f]:
                if a not in vals: row.append("--"); continue
                mu, sd, n = vals[a]
                txt = f"{mu:.4f}" + (f"±{sd:.4f}" if n > 1 else "") + ("" if n in (1, 5) else f" (n={n})")
                row.append(f"**{txt}**" if abs(mu - best) < 5e-5 else txt)
        rows.append(row)
    # summary vs paper and vs W=1 (datasets present in both)
    summ = []
    for f in fams:
        ref = stats[(f, "paper")]["mean"]
        for a in arms[f]:
            s = stats[(f, a)]["mean"]; common = ref.index.intersection(s.index)
            if len(common) == 0: summ.append((f, a, "--", "--", "--", 0)); continue
            dlt = (s[common] - ref[common])
            wtl = f"{(dlt > TIE).sum()}/{(dlt.abs() <= TIE).sum()}/{(dlt < -TIE).sum()}"
            summ.append((f, a, f"{s[common].mean():.4f} (paper {ref[common].mean():.4f})", f"{dlt.mean():+.4f}", wtl, len(common)))
    out = []
    out.append(f"Test {m}, mean±std over folds (chrono datasets: 1 fold). Bold = best in its family block. "
               "Paper = single-machine Exact arms (SPO: P=F projections per node); W=n = distributed, "
               "n in-process workers, P=F total (F/n per worker from its own columns).")
    out.append("| " + " | ".join(hdr) + " |"); out.append("|" + "---|" * len(hdr))
    for r in rows: out.append("| " + " | ".join(r) + " |")
    out.append("")
    out.append("| family | arm | mean AUC over the common datasets | mean Δ vs paper | win/tie/loss vs paper (|Δ|≤0.005 tie) | datasets |")
    out.append("|---|---|---|---|---|---|")
    for s in summ: out.append("| " + " | ".join(map(str, s)) + " |")
    print("\n".join(out))
    if args.latex:
        print("\n% LaTeX rows (dataset & cells), same column order as the markdown header")
        for r in rows:
            print(" & ".join(c.replace("**", "\\textbf{", 1).replace("**", "}") if c.startswith("**") else c for c in r).replace("_", "\\_").replace("±", "$\\pm$") + " \\\\")

if __name__ == "__main__":
    main()
