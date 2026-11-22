#!/usr/bin/env python3
"""Column-replication inputs of fig:column-replication-where-spo-outperforms-aa{,-regression}, Hao settings (2026-11-21).
Rows = the kdd28_hao pass_rec datasets of the table. Writes derived/hao_{cls,reg}/{colrep,r0,single}.csv (W = 4/8/16/32 SPO at
r > 0, the distributed SPO cell at the same W as r = 0, single-machine SPO), all kdd28_hao seeds 1-5. Each source is checked strictly against its own eval suite
and profile here, restricted to the selected datasets. CC18 rows are renamed "<name> (CC18)" (credit-g, jm1 and qsar-biodeg
are also SPAA rows). Run from the repo root."""
import glob
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT / "benchmarks/src/distributed_spo_gbt/params"))
import params as kdd28  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
R = ROOT / "benchmarks/results/kdd28/runs"
H, N = R / "2026-11-06_hao_seeds", R / "2026-11-21_where_spo_beats_aa_cc18_reg_distributed"
N2 = R / "2026-11-21_colrep_w4_to_w32_r5pct_where_spo_beats_aa_hao"  # W = 8/16/32 + the 0.05-step r (2026-11-22)
W = (4, 8, 16, 32)
SEL = ROOT / "benchmarks/results/kdd28/floats/accuracy/tab-accuracy-hao-vs-ydf-params-single-machine/derived/spo_clearly_beats_aa_per_dataset.csv"
S = range(1, 6)
# suite key -> (eval suite, colrep files, r0/dist files, single files, single arm value, single profile arm)
SRC = {
    "spaa": ("kdd28_tabarena33", [H / "colrep_distacc/results.csv"] + [N2 / f"spaa/seed{s}/colrep_distacc/results.csv" for s in S],
             [H / "distacc/results.csv"],
             [H / "distacc/results.csv"], "single:SPO", "distacc_single_spo"),
    "wide": ("kdd28_wide_datasets", sorted(H.glob("wide_eval/colrep_distacc/*/seed*/results.csv"))
             + [N2 / f"{g}/seed{s}/colrep_distacc/results.csv" for g in ("eps", "phil") for s in S],
             [H / "wide_eval/distacc/results.csv"], [H / "wide_eval/distacc/results.csv"], "single:SPO", "distacc_single_spo"),
    "cc18": ("kdd28_cc18_10fold", [N / f"kdd28_hao/seed{s}/colrep_cc18/results.csv" for s in S]
             + [N2 / f"cc18/seed{s}/colrep_cc18/results.csv" for s in S],
             [N / f"kdd28_hao/seed{s}/cc18_dist/results.csv" for s in S],
             [R / f"2026-11-20_cc18_seeds/kdd28_hao/seed{s}/cc18/results.csv" for s in S], "cc18_spo", "cc18_spo"),
    "regression": ("kdd28_regression", [N / f"kdd28_hao/seed{s}/colrep_reg/results.csv" for s in S]
                   + [N2 / f"reg/seed{s}/colrep_reg/results.csv" for s in S],
                   [N / f"kdd28_hao/seed{s}/reg_dist/results.csv" for s in S],
                   [R / f"2026-11-20_regression_multiclass_tabarena_tabred/kdd28_hao/seed{s}/reg/results.csv" for s in S],
                   "single:SPO", "distacc_single_spo"),
}


def load(files, ds, keep, suite, arm, label):
    x = pd.concat([pd.read_csv(f) for f in files])
    x = x[(x.rc == 0) & x.dataset.isin(ds)]
    x = keep(x)
    kdd28._declare_eval(suite, __file__)
    kdd28._EVAL["subset"] = set(ds)
    kdd28.check_frame(x, "kdd28_hao", [arm], label)
    kdd28._EVAL["subset"] = None
    if set(x.dataset) != set(ds):
        sys.exit(f"ERROR {label}: missing datasets {sorted(set(ds) - set(x.dataset))}")
    return x


def main():
    sel = pd.read_csv(SEL)
    pick = {s: sorted(g.dataset) for s, g in sel[(sel.profile == "kdd28_hao") & sel.pass_rec].groupby("suite")}
    out = {"cls": {"colrep": [], "r0": [], "single": []}, "reg": {"colrep": [], "r0": [], "single": []}}
    for key, (suite, cf, rf, sf, sarm, sprof) in SRC.items():
        ds = pick.get(key, [])
        if not ds:
            continue
        o = out["reg" if key == "regression" else "cls"]
        parts = {"colrep": load(cf, ds, lambda x: x[x.replication > 0], suite, "colrep_distacc_spo", f"{key} colrep"),
                 "r0": load(rf, ds, lambda x: x[x.arm.isin([f"{w}:SPO" for w in W])], suite, "distacc_spo", f"{key} r0"),
                 "single": load(sf, ds, lambda x: x[x.arm == sarm], suite, sprof, f"{key} single")}
        for k, x in parts.items():
            if key == "cc18":
                x = x.assign(dataset=x.dataset + " (CC18)")
            o[k].append(x)
    for grp, parts in out.items():
        d = HERE / "derived" / f"hao_{grp}"
        d.mkdir(parents=True, exist_ok=True)
        for k, xs in parts.items():
            x = pd.concat(xs).sort_values([c for c in ("dataset", "fold", "seed", "workers", "replication") if c in xs[0]])
            x.to_csv(d / f"{k}.csv", index=False)
            print(f"wrote {d / k}.csv: {len(x)} rows, {x.dataset.nunique()} datasets")


if __name__ == "__main__":
    main()
