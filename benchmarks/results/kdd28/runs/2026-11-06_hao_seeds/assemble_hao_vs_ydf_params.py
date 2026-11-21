#!/usr/bin/env python3
"""Builds the kdd28_hao 5-seed inputs of sec:accuracy-hao-vs-ydf-params (SPAA paper eval at the top level, wide dataset
eval under wide_eval/) from the EC2 mirror plus the seed-1 rows adopted from kdd28_ydf (profile kdd28_hao "_reuses").
Run from the repo root: python3 benchmarks/results/kdd28/runs/2026-11-06_hao_seeds/assemble_hao_vs_ydf_params.py"""
import pathlib
import pandas as pd

R = pathlib.Path("benchmarks/results/kdd28/runs")
BOX = pathlib.Path("ariel_notes/results/2026-11-06_hao_seeds/box")  # gitignored mirror of ~/results/2026-11-06_hao_seeds


def mirror(parts, table):
    """(part, results.csv) of seeds 1-5. Exits if one is missing: a partial mirror would drop seeds from the run CSVs."""
    files = [(p, BOX / p / f"seed{s}" / table / "results.csv") for p in parts for s in range(1, 6)]
    missing = [str(f) for _, f in files if not f.exists()]
    if missing:
        raise SystemExit(f"ERROR mirror incomplete, nothing written: {missing}")
    return files


def hao(parts, table):
    """kdd28_hao rows of seeds 1-5 (seed-1 single:SPO smoke rows dropped: exact300 stands in for them)."""
    x = pd.concat([pd.read_csv(f).assign(source=f"hao_seeds:{p}") for p, f in mirror(parts, table)])
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


WIDE = ["wide", "wide_eps"]  # wide dataset eval: 7 sets in wide/, epsilon in wide_eps/
for _t in ("distacc", "distacc_aaspo", "colrep_distacc"):  # check every input before the first write
    mirror(["spaa"], _t)
for _t in ("distacc", "distacc_aaspo"):
    mirror(WIDE, _t)
out, k = R / "2026-11-06_hao_seeds", ["dataset", "fold", "arm"]
write(pd.concat([hao(["spaa"], "distacc"), reused(R / "2026-11-03_distacc/results.csv", R / "2026-11-04_distacc_aaspo/distacc_exact300.csv")]),
      out / "distacc/results.csv", k)
write(hao(["spaa"], "distacc_aaspo"), out / "distacc_aaspo/results.csv", k)
write(hao(["spaa"], "colrep_distacc"), out / "colrep_distacc/results.csv", ["dataset", "fold", "replication"])
W = R / "2026-11-04_wide_eval"
write(pd.concat([hao(WIDE, "distacc"), reused(W / "distacc/results.csv", W / "distacc_exact300/results.csv")]),
      out / "wide_eval/distacc/results.csv", k)
write(hao(WIDE, "distacc_aaspo"), out / "wide_eval/distacc_aaspo/results.csv", k)
