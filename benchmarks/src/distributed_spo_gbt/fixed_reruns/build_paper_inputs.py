#!/usr/bin/env python3
"""Paper-generator inputs from the fixed reruns (old fixed_reruns_2026-11-01/ layout, now mapped by layout.py into
benchmarks/results/kdd28: paper_inputs/X lands in the float dir that reads it):
  table2/  results.csv + summary_pooled.csv of table2/fullfix_splitbudget (cluster P = F, the single-machine
           budget) + the unchanged single-machine row
           -> make_column_replication_{table,fig}.py
  table3_results.csv  one row per (dataset, arm) like naive_column_shipping/.../results.csv, from table3/fullfix
           (rows/features copied from the paper run) -> make_cross_network_{table,fig}.py
  table3_results_5seed.csv  same + seed column, seeds 1-5 from table3/fullfix_seeds (if present)
  table8_results.csv  run_table8_local.sh layout for make_paper_tables.py --accuracy_csv: SPO distributed arms from
           table8/fullfix (m7i); single-machine arms = the paper's m7i cells (speedup_map suite_results, unchanged by
           the fixes); AA distributed arms from table8_rerun_20261030 (AA is CPU-independent: 765/765 = paper cells)
  python build_paper_inputs.py [--results <fixed_reruns dir>] [--aa_csv <AA rerun csv>]
"""
import argparse
import glob
import os
import shutil

import pandas as pd
from layout import at

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../../.."))
T8 = ["dataset", "fold", "arm", "workers", "split_axis", "threads", "rc", "test_acc", "test_auc", "test_logloss", "wall_s", "git"]


def table2(res):
    d = at(res, "paper_inputs/table2")
    os.makedirs(d, exist_ok=True)
    for f in ["results.csv", "summary_pooled.csv"]:
        shutil.copy(at(res, f"table2/fullfix_splitbudget/{f}"), d)
    shutil.copy(at(res, "paper_refs/table2_single_machine_spo_gbt.csv"), os.path.join(d, "single_machine_spo_gbt.csv"))


def table3_cells(res, sub):
    """table3/<sub>/g*/results.csv -> paper-run columns + seed, one row per (dataset, arm, seed)."""
    paper = pd.read_csv(at(res, "paper_refs/table3_results.csv"))
    size = paper.drop_duplicates("dataset").set_index("dataset")[["rows_train", "features"]]
    n = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(at(res, f"table3/{sub}/g*/results.csv"))))
    n = n[n.rc == 0]
    task = n.dataset
    n["dataset"] = n.dataset.str.replace(r"^task_\d+_", "", regex=True)
    n["arm"] = n.replication.astype(str).map(lambda r: "cross_network" if r == "ship" else "in_worker")
    n = n.join(size, on="dataset")
    assert n.rows_train.notna().all() and (n.groupby("seed").size() == len(paper)).all(), \
        "dataset/arm mismatch with the paper run"
    for c in ["ship_bytes_in", "ship_bytes_out", "ship_cols_in", "cross_worker_splits"]:
        n[c] = n[c].fillna(0).astype("int64")
    for c in ["ship_s_max", "ship_s_sum"]:
        n[c] = n[c].fillna(0)
    n["machine"], n["run_id"] = "n2-highmem-4", "fullfix20261101"
    n["cell"] = task + "/" + n.replication.astype(str).map(lambda r: "ship" if r == "ship" else f"r{float(r):.2f}")
    return n[list(paper.columns) + ["seed"]].sort_values(["dataset", "arm", "seed"])


def table3(res):
    n = table3_cells(res, "fullfix")
    n.drop(columns="seed").to_csv(at(res, "paper_inputs/table3_results.csv"), index=False)


def table3_seeds(res):
    """5-seed cross-network input (seed column kept; the generators average over seeds)."""
    n = table3_cells(res, "fullfix_seeds")
    assert sorted(n.seed.unique()) == [1, 2, 3, 4, 5], sorted(n.seed.unique())
    n.to_csv(at(res, "paper_inputs/table3_results_5seed.csv"), index=False)


def table8(res, aa_csv):
    spo = pd.read_csv(at(res, "table8/fullfix/results.csv"))
    spo = spo[(spo.rc == 0) & (spo.split_axis == "SPARSE_OBLIQUE")]
    aa = pd.read_csv(aa_csv)
    aa = aa[(aa.rc == 0) & (aa.split_axis == "AXIS_ALIGNED") & (aa.workers != "single")]
    s = pd.read_csv(os.path.join(REPO, "benchmarks/results/runtime/speedup_map_by_dataset/suite_results.csv"))
    meth = {"spo_gbt_exact_hwy_fullproj": ("single:SPO", "SPO"), "aa_gbt_exact": ("single:AA", "AA")}
    s = s[(s.status == "OK") & s.method.isin(meth)]
    single = pd.DataFrame({"dataset": s.dataset, "fold": s.fold, "arm": s.method.map(lambda m: meth[m][0]),
                           "workers": "single", "split_axis": s.method.map(lambda m: meth[m][1]), "threads": "",
                           "rc": 0, "test_acc": s.test_acc, "test_auc": s.test_auc, "test_logloss": s.test_logloss,
                           "wall_s": "", "git": "paper suite_results (m7i)"})
    spo = spo.assign(git="dgbt-spo-full-fixes 10509539 (m7i)")
    aa = aa.assign(git="table8_rerun_20261030 (AA)")
    all_ = pd.concat([single, spo[T8], aa[T8]]).sort_values(["dataset", "fold", "arm"])
    all_.to_csv(at(res, "paper_inputs/table8_results.csv"), index=False)
    print(all_.groupby("arm").size().to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None, help="an old-layout fixed_reruns dir (default: the kdd28 tree, layout.py)")
    ap.add_argument("--aa_csv", default=os.path.join(
        REPO, "benchmarks/results/kdd28/appendix-earlier-settings/tab-dist-accuracy/raw/table8_rerun_20261030/results.csv"))
    a = ap.parse_args()
    if a.results:
        os.makedirs(at(a.results, "paper_inputs"), exist_ok=True)
    table2(a.results)
    table3(a.results)
    if glob.glob(at(a.results, "table3/fullfix_seeds/g*/results.csv")):
        table3_seeds(a.results)
    table8(a.results, a.aa_csv)
    print("wrote the paper_inputs files (layout.py map)")


if __name__ == "__main__":
    main()
