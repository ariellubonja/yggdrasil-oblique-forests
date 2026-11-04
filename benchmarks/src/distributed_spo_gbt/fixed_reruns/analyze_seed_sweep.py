#!/usr/bin/env python3
"""Open question: is Table 8's W=1-vs-single SPO AUC gap systematic? Seed sweep on the m7i.

Arms: single = train_oblique_forest (paper flags), seeds 1 (paper cells) + sweep seeds; w1 = train_distributed_gbt
10509539 W=1 numeric typing, default seed (Table 8 fullfix_numeric cells) + sweep seeds. Per-arm mean over seeds per
(dataset, fold), then dataset means, then the paired difference with SE across the 33 datasets.
  python3 analyze_seed_sweep.py <sweep_root>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
RES = REPO / "benchmarks/results/accuracy/distributed_spo_gbt/fixed_reruns_2026-11-01"
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else RES / "table8/seed_sweep"
SUITE = REPO / "benchmarks/results/runtime/speedup_map_by_dataset/suite_results.csv"


def load():
  frames = [pd.read_csv(p) for p in sorted(ROOT.glob("sweep_*/results.csv"))]
  sw = pd.concat(frames)
  bad = sw[(sw.rc != 0) | sw.test_auc.isna()]
  if len(bad):
    print("NON-ZERO rc / missing metrics:\n", bad)
  sw = sw[(sw.rc == 0) & sw.test_auc.notna()][["dataset", "fold", "arm", "seed", "test_acc", "test_auc", "test_logloss"]]
  p = pd.read_csv(SUITE)
  p = p[(p.status == "OK") & (p.method == "spo_gbt_exact_hwy_fullproj")]
  paper = p.assign(arm="single", seed=1)[["dataset", "fold", "arm", "seed", "test_acc", "test_auc", "test_logloss"]]
  num = pd.read_csv(RES / "table8/fullfix_numeric/results.csv")
  num = num[num.arm == "1:SPO"].assign(arm="w1", seed=123456)[
      ["dataset", "fold", "arm", "seed", "test_acc", "test_auc", "test_logloss"]]
  return pd.concat([paper, num, sw])


def main():
  d = load()
  cnt = d.groupby(["arm", "seed"]).size().unstack(0)
  print("cells per (seed, arm):\n", cnt.to_string())
  full = d.groupby(["arm", "seed"]).size()
  keep = full[full == 153].index
  d = d.set_index(["arm", "seed"]).loc[keep].reset_index()
  ds = d.groupby(["arm", "seed", "dataset"])[["test_auc", "test_logloss", "test_acc"]].mean()
  per_seed = ds.groupby(["arm", "seed"]).mean()
  print("\nper-seed arm means (dataset-mean over 33 datasets):\n", per_seed.round(5).to_string())
  for m in ["test_auc", "test_logloss", "test_acc"]:
    a = ds[m].unstack(["arm", "seed"])
    s, w = a["single"].mean(axis=1), a["w1"].mean(axis=1)
    diff = w - s
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    sd_seed = {k: per_seed.loc[k][m].std(ddof=1) for k in ["single", "w1"]}
    ns, nw = a["single"].shape[1], a["w1"].shape[1]
    print(f"\n{m}: w1 - single (mean of {nw} vs {ns} seeds) = {diff.mean():+.5f} +- {se:.5f} (t={diff.mean()/se:+.2f}); "
          f"seed-to-seed sd of arm means: single {sd_seed['single']:.5f}, w1 {sd_seed['w1']:.5f}")
    if m == "test_auc":
      # Each single seed vs mean of the w1 seeds: shows how one single draw scatters.
      for seed in a["single"].columns:
        dd = w - a["single"][seed]
        print(f"   w1(mean) - single(seed {seed}) = {dd.mean():+.5f} +- {dd.std(ddof=1)/np.sqrt(len(dd)):.5f}")
      for seed in a["w1"].columns:
        dd = a["w1"][seed] - s
        print(f"   w1(seed {seed}) - single(mean) = {dd.mean():+.5f} +- {dd.std(ddof=1)/np.sqrt(len(dd)):.5f}")
      out = pd.DataFrame({"single_mean": s, "w1_mean": w, "diff": diff}).sort_values("diff")
      out.to_csv(HERE / "seed_sweep_per_dataset.csv")
      print("\nper-dataset (most negative first):\n", out.round(4).head(8).to_string(), "\n...\n",
            out.round(4).tail(5).to_string())


if __name__ == "__main__":
  main()
