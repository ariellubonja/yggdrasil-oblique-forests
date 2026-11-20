#!/usr/bin/env python3
"""Builds the kdd28_hao 5-seed inputs of tab:accuracy-hao-vs-ydf-params (SPAA paper eval) and tab:accuracy-hao-vs-ydf-params-over-100-columns
(wide union) from the EC2 mirror plus the seed-1 rows adopted from kdd28_ydf (profile kdd28_hao "_reuses").
Run from the repo root: python3 benchmarks/results/kdd28/accuracy/tab-accuracy-hao-vs-ydf-params/assemble_hao_vs_ydf_params.py"""
import pathlib
import pandas as pd

A = pathlib.Path("benchmarks/results/kdd28/accuracy")
BOX = pathlib.Path("ariel_notes/results/2026-11-06_hao_seeds/box")  # gitignored mirror of ~/results/2026-11-06_hao_seeds
YDF, WEV = A / "tab-accuracy-on-spaa-datasets", A / "tab-accuracy-on-datasets-over-100-columns/wide_eval"
WIDE = set(pd.read_csv(A / "tab-accuracy-on-datasets-over-100-columns/wide_shapes.csv").dataset)  # the 14 datasets with F >= 100


def hao(parts, table):
    """kdd28_hao rows of seeds 1-5 (seed-1 single:SPO smoke rows dropped: exact300 stands in for them)."""
    x = pd.concat([pd.read_csv(BOX / p / f"seed{s}" / table / "results.csv").assign(source=f"hao_seeds:{p}")
                   for p in parts for s in range(1, 6) if (BOX / p / f"seed{s}" / table / "results.csv").exists()])
    x = x[x.rc == 0]
    if table == "distacc":
        x = x[~((x.arm == "single:SPO") & (x.seed == 1))]
    return x


def reused(distacc, exact300):
    """Seed-1 rows of kdd28_ydf: exact300 -> single:SPO, single:AA, 4:AA."""
    e = pd.read_csv(exact300).query("rc == 0").assign(arm="single:SPO", split_axis="SPO", source=f"kdd28_ydf:{exact300}")
    d = pd.read_csv(distacc).query("rc == 0")
    d = d[d.arm.isin(["single:AA", "4:AA"])].assign(source=f"kdd28_ydf:{distacc}")
    return pd.concat([e, d])


def write(df, path, keys):
    path.parent.mkdir(parents=True, exist_ok=True)
    dup = df.duplicated(keys + ["seed"], keep=False)
    assert not dup.any(), f"{path}: duplicate cells {df[dup][keys + ['seed']].head()}"
    df.sort_values(keys + ["seed"]).to_csv(path, index=False)
    print(path, len(df), "rows,", df.dataset.nunique(), "datasets, seeds", sorted(df.seed.unique()))


spaa_ds = None
for name, parts, rd in (("tab-accuracy-hao-vs-ydf-params", ["spaa"], [(YDF / "distacc/results.csv", YDF / "distacc_aaspo/distacc_exact300.csv")]),
                        ("tab-accuracy-hao-vs-ydf-params-over-100-columns", ["spaa", "wide", "wide_eps"],
                         [(YDF / "distacc/results.csv", YDF / "distacc_aaspo/distacc_exact300.csv"),
                          (WEV / "distacc/results.csv", WEV / "distacc_exact300/results.csv")])):
    keep = (lambda d: d[d.dataset.isin(WIDE)]) if name.endswith("over-100-columns") else (lambda d: d)
    out = A / name
    k = ["dataset", "fold", "arm"]
    write(keep(pd.concat([hao(parts, "distacc")] + [reused(*p) for p in rd])), out / "distacc/results.csv", k)
    write(keep(hao(parts, "distacc_aaspo")), out / "distacc_aaspo/results.csv", k)
    write(keep(hao(parts, "colrep_distacc")), out / "colrep_distacc/results.csv", ["dataset", "fold", "replication"])
