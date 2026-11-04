#!/usr/bin/env python3
"""Emit the KDD28 column-replication table BODY (tabular only; caption/label live in main.tex).
Per replication level r: mean test accuracy over datasets (of per-dataset seed means), delta and
wins/ties/losses vs single-machine SPO-GBT (same hyperparameters, single_machine_spo_gbt.csv), mean rank
among the r levels. A bold last row gives single-machine SPO-GBT itself (the baseline, not ranked)."""
import argparse, csv, os, pathlib, sys
from collections import defaultdict
from statistics import mean

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "params"))
import params as kdd28  # noqa: E402  (params/kdd28_params.json)

ROOT = pathlib.Path(__file__).resolve().parents[3]
# argv: [src_dir (results.csv, summary_pooled.csv, single_machine_spo_gbt.csv)] [out_body.tex] [run note]
_ap = argparse.ArgumentParser()
_ap.add_argument("src", nargs="?", default=str(ROOT / "benchmarks/results/kdd28/appendix-earlier-settings/tab-column-replication/table2_single10"))
_ap.add_argument("out_pos", nargs="?", default=None, metavar="out")
_ap.add_argument("note", nargs="?", default="Vertex AI run colrep-fullparity_fullparity20261101 with --split_budget, "
                 "dgbt-spo-full-fixes 10509539; single row 10 seeds (1-5 run 2c0990f0, 6-10 parity audit S1)")
_ap.add_argument("--out", default=None, help="output body path (same as the 2nd positional)")
_ap.add_argument("--single_csv", default=None, help="single-machine rows (default: <src>/single_machine_spo_gbt.csv)")
_ap.add_argument("--dist_arm", default="dist_split", help="profile arm of the distributed rows (kdd28_ydf: colrep_spo)")
_ap.add_argument("--single_arms", default="single_spo,single_spo_audit", help="profile arms of the single rows")
_ap.add_argument("--metric", default="test_acc", choices=["test_acc", "test_auc"], help="headline metric")
_ap.add_argument("--single_filter", default="", help="col=value: keep only these single rows (distacc CSV: arm=single:SPO)")
_ap.add_argument("--r0_csv", default="", help="rows reused as r = 0 (distacc results.csv), filtered by --r0_filter")
_ap.add_argument("--r0_filter", default="arm=4:SPO", help="col=value of the --r0_csv rows")
_ap.add_argument("--r0_arm", default="distacc_spo", help="profile arm of the --r0_csv rows")
kdd28.add_profile_arg(_ap, __file__)
_a = _ap.parse_intermixed_args()
if _a.out and _a.out_pos:  # "src --out X note": the 2nd positional is the note
    _a.note, _a.out_pos = _a.out_pos, None
SRC = pathlib.Path(_a.src).resolve()
OUT = pathlib.Path(_a.out or _a.out_pos or ROOT / "paper/distributed_spo_gbt_kdd28/tables/table_column_replication_body.tex")
NOTE = _a.note
SINGLE = pathlib.Path(_a.single_csv).resolve() if _a.single_csv else SRC / "single_machine_spo_gbt.csv"
SINGLE_ARMS = _a.single_arms.split(",")
EPS = 1e-9
DASH = r"\multicolumn{1}{c}{---}"
M = _a.metric


def rows(path, filt=""):
    """rc = 0 rows of a CSV with the metric, optionally only col == value."""
    col, _, val = filt.partition("=")
    return [r for r in csv.DictReader(open(path)) if r["rc"] == "0" and r[M] and (not col or r[col] == val)]


def pooled(dist_rows):
    """replication -> (n_datasets, metric mean, gain_pts, wins, ties, losses, mean_rank)."""
    acc = defaultdict(list)
    for r in dist_rows:
        acc[(r["dataset"], float(r["replication"]))].append(float(r[M]))
    m = {k: mean(v) for k, v in acc.items()}
    datasets = sorted({d for d, _ in acc})
    levels = sorted({l for _, l in acc})
    full = [d for d in datasets if all((d, l) in acc for l in levels)
            and len({len(acc[(d, l)]) for l in levels}) == 1]
    out = {}
    for l in levels:
        gains = [m[(d, l)] - m[(d, 0.0)] for d in full]
        wins, losses = sum(g > EPS for g in gains), sum(g < -EPS for g in gains)
        ranks = []
        for d in full:
            vs, v = [m[(d, x)] for x in levels], m[(d, l)]
            ranks.append(1 + sum(x > v + EPS for x in vs) + 0.5 * (sum(abs(x - v) <= EPS for x in vs) - 1))
        out[l] = (len(full), mean(m[(d, l)] for d in full), 100 * mean(gains), wins,
                  len(full) - wins - losses, losses, mean(ranks))
    return out


def vs_single(single_rows, dist_rows):
    """(single metric mean, {r: (gain_pts, wins, ties, losses)}) of distributed level r vs single-machine SPO-GBT."""
    acc, dist = defaultdict(list), defaultdict(list)
    for r in single_rows:
        acc[r["dataset"]].append(float(r[M]))
    for r in dist_rows:
        dist[(r["dataset"], float(r["replication"]))].append(float(r[M]))
    out = {}
    for l in sorted({l for _, l in dist}):
        ds = sorted(d for d, x in dist if x == l)
        assert set(acc) == set(ds), (l, set(acc) ^ set(ds))
        gains = [mean(dist[(d, l)]) - mean(acc[d]) for d in ds]
        wins, losses = sum(g > EPS for g in gains), sum(g < -EPS for g in gains)
        out[l] = (100 * mean(gains), wins, len(gains) - wins - losses, losses)
    return mean(mean(v) for v in acc.values()), out


def main():
    kdd28.check_csv(SRC / "results.csv", _a.profile, [_a.dist_arm], "results.csv (distributed)")
    single = rows(SINGLE, _a.single_filter)
    if _a.single_filter:
        kdd28.check_rows(single, _a.profile, SINGLE_ARMS, SINGLE.name + " (single)", arm_of=lambda r: SINGLE_ARMS[0])
    else:
        kdd28.check_csv(SINGLE, _a.profile, SINGLE_ARMS, SINGLE.name + " (single)")
    dist = rows(SRC / "results.csv")
    if _a.r0_csv:  # r = 0 cells reused from distacc (same argv bar --column_replication 0, models byte-identical)
        r0 = rows(_a.r0_csv, _a.r0_filter)
        kdd28.check_rows(r0, _a.profile, [_a.r0_arm], pathlib.Path(_a.r0_csv).name + " (r = 0)", arm_of=lambda r: _a.r0_arm)
        dist = [r for r in dist if float(r["replication"]) != 0] + [r | {"replication": "0.0"} for r in r0]
    p = pooled(dist)
    # Cross-check against summarize_replication.py's output (same rounding), when the run has one.
    if not (SRC / "summary_pooled.csv").exists():
        print(f"WARN {SRC}/summary_pooled.csv missing: pooled cross-check skipped", file=sys.stderr)
    for r in csv.DictReader(open(SRC / "summary_pooled.csv")) if (SRC / "summary_pooled.csv").exists() else []:
        n, a, g, w, t, lo, rk = p[float(r["replication"])]
        mine = [str(n), f"{a:.4f}", f"{g:+.2f}", str(w), str(t), str(lo), f"{rk:.2f}"]
        theirs = [r[k] for k in ("n_datasets", "acc_mean", "gain_vs_0_pts", "wins", "ties", "losses", "mean_rank")]
        assert mine == theirs, (r["replication"], mine, theirs)
    lines = ["% auto-generated by benchmarks/src/distributed_spo_gbt/make_column_replication_table.py -- regenerate, do not hand-edit",
             f"% data: {os.path.relpath(SRC, ROOT)}/results.csv ({NOTE}); single: {os.path.relpath(SINGLE, ROOT)}"
             + (f" ({_a.single_filter})" if _a.single_filter else "")
             + (f"; r = 0: {os.path.relpath(pathlib.Path(_a.r0_csv).resolve(), ROOT)} ({_a.r0_filter})" if _a.r0_csv else ""),
             "", r"\begin{tabular}{crrcr}", r"\toprule",
             r"$r$ & \multicolumn{1}{c}{Mean " + ("AUC" if M == "test_auc" else "acc.") + r"} & \multicolumn{1}{c}{$\Delta$ (pts)} & W/T/L & \multicolumn{1}{c}{Mean rank} \\",
             r"\midrule"]
    sa, vs = vs_single(single, dist)
    for l, (n, a, _, _, _, _, rk) in sorted(p.items()):
        g, w, t, lo = vs[l]
        gs = "0.00" if abs(g) < 0.005 else f"{g:+.2f}"  # no "-0.00"
        acc = f"{a:.4f}".lstrip("0")
        lines.append(f"{l:.1f} & {acc} & ${gs}$ & {w}/{t}/{lo} & {rk:.2f} \\\\")
    lines += [r"\midrule",
              rf"\textbf{{single}} & \textbf{{{f'{sa:.4f}'.lstrip('0')}}} & {DASH} & {DASH} & {DASH} \\"]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    arms = SINGLE_ARMS + ([_a.dist_arm, _a.r0_arm] if _a.r0_csv else [])
    OUT.write_text(kdd28.add_setup("\n".join(lines), kdd28.setup_line(_a.profile, arms)))
    print(f"wrote {OUT} ({n} datasets, {len(p)} levels)", file=sys.stderr)


if __name__ == "__main__":
    main()
