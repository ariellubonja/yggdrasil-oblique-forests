#!/usr/bin/env python3
"""One CSV with every coarse CHRONO scope per tree depth, for the five split-search stacks
behind the "Scalar-Dynamic" figure: the pre-Highway stack (Exact std::sort, scalar 64-bin Random
histogram, scalar Dynamic = scalar histogram >=1350 rows / std::sort below) and the current one
(Exact Highway VQSort, AVX2 64-bin Random histogram).

Input: single-tree, single-thread coarse parallel_chrono CSVs under
benchmarks/results/runtime/per_function_timing/COARSE/<cpu>/<arm dir>/<dataset>/raw/<stem>.csv
(first thread block; a scope column absent from a run = 0 in that run, so it is filled with 0).

Output (tidy, one row per dataset x method x depth; depth 0 carries only TreeTrain = the whole tree):
  dataset, method, depth, nodes, active_samples, <one column per chrono scope, seconds>
Scope names are the CSV headers with the nesting dashes stripped (NodeTrain, FindBestCondition,
ObliqueSplitSearch, FindObliqueSetup, SampleProjection, ApplyProjection, EvaluateProj, ...).

Usage: python3 benchmarks/results/figures/make_per_depth_chrono_table.py [--out FILE] [--datasets trunk1m,HIGGS]
"""
import argparse, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "benchmarks/results/runtime/per_function_timing/COARSE/Intel(R) Xeon(R) Platinum 8488C"
DATASET_DIRS = {"trunk1m": "trunk_1000000_x_4096", "HIGGS": "HIGGS_with_header"}
# (method, arm dir, {dataset dir: raw stem, "*": default}) -- the HIGGS scalar run kept the helper's default name.
ARMS = [
    ("Exact (std::sort)",             "Oblique | Exact",                             {"*": "stdSort"}),
    ("Random Hist. (scalar)",         "Oblique | Random | Scalar",                   {"*": "scalar", "HIGGS_with_header": "-1Depth-1Threads"}),
    ("Dynamic Hist. (scalar, thr 1350)", "Oblique | Dynamic Random Histogram | Scalar", {"*": "scalar_stdSort_thr1350"}),
    ("Exact (HWY VQSort)",            "Oblique | Exact",                             {"*": "hwy"}),
    ("Random Hist. (AVX2 64-bin)",    "Oblique | Random",                            {"*": "vectorized-64"}),
]
KEY = ["dataset", "method", "depth", "nodes", "active_samples"]


def load(path: Path) -> pd.DataFrame:
    """First thread block of a parallel_chrono CSV, every column, dashes stripped."""
    hdr = pd.read_csv(path, header=None, nrows=2)
    cols = hdr.iloc[1].tolist()
    ncol = next(i for i, c in enumerate(cols) if isinstance(c, float))  # first blank separator
    df = pd.read_csv(path, header=1, usecols=range(ncol))
    df = df[df["tree"] == df["tree"].min()].drop(columns=["tree"])
    df = df.rename(columns={"Active Samples": "active_samples"})
    df.columns = [c.lstrip("-") for c in df.columns]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "benchmarks/results/figures/per_depth_chrono_scalar_dynamic.csv"))
    ap.add_argument("--datasets", default="trunk1m,HIGGS")
    a = ap.parse_args()
    frames = []
    for ds in a.datasets.split(","):
        ds_dir = DATASET_DIRS[ds]
        for method, arm, stems in ARMS:
            p = BASE / arm / ds_dir / "raw" / f"{stems.get(ds_dir, stems['*'])}.csv"
            if not p.exists():
                sys.exit(f"missing: {p.relative_to(ROOT)}")
            d = load(p)
            d.insert(0, "method", method); d.insert(0, "dataset", ds)
            frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    scopes = [c for c in out.columns if c not in KEY]
    out[scopes] = out[scopes].fillna(0.0)
    out = out[KEY + scopes].sort_values(["dataset", "method", "depth"], key=lambda s: s.map(
        {m: i for i, (m, _, _) in enumerate(ARMS)}) if s.name == "method" else s)
    out.to_csv(a.out, index=False)
    print("wrote", a.out, out.shape)
    # Whole-tree totals per scope (sum over depths >= 1; TreeTrain from depth 0).
    tot = out[out["depth"] >= 1].groupby(["dataset", "method"], sort=False)[scopes].sum()
    tot["TreeTrain"] = out[out["depth"] == 0].set_index(["dataset", "method"])["TreeTrain"]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(tot.round(1).to_string())


if __name__ == "__main__":
    main()
