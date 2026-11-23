#!/usr/bin/env python3
# Checks 1-4 and 7 of the 2026-11-22 col-rep grid from the CSVs (+ BINARY weights from the cell logs with --logs), and
# derived/colrep_mean_delta.csv. Run from the repo root: python3 <this> [--logs <dir with seed<s>/.../logs>]. Exit 1 on FAIL.
import argparse, csv, glob, math, os, statistics as st
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
SCREEN = "benchmarks/results/kdd28/runs/2026-11-22_wide_real_dataset_candidates_spo_vs_aa_screen_hao"
SETS = {"jasmine": 144, "madeline": 259, "fabert": 800, "robert": 7200}  # F after ID columns (screen what_changed.csv)
WS, SEEDS, FOLDS = (4, 8, 16, 32), (1, 2, 3, 4, 5), ("0", "1", "2", "3", "4")
RS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
read = lambda p: list(csv.DictReader(open(p, newline="")))
ap = argparse.ArgumentParser()
ap.add_argument("--logs", default="", help="raw mirror root holding seed<s>/colrep_distacc_stamped/logs/")
a = ap.parse_args()
res, fails = [], 0


def report(name, ok, msg):
    global fails
    fails += not ok
    res.append(f"{'PASS' if ok else 'FAIL'} {name}: {msg}")


rows = [dict(r, _seed=s) for s in SEEDS for r in read(f"{H}/seed{s}/colrep_distacc_stamped/results.csv")]
# 1. completeness
keys = [(r["dataset"], r["fold"], r["_seed"], int(r["workers"]), float(r["replication"])) for r in rows]
want = {(d, f, s, w, r) for d in SETS for f in FOLDS for s in SEEDS for w in WS for r in RS}
bad_metric = [k for k, r in zip(keys, rows) if r["rc"] != "0" or not r["test_auc"] or math.isnan(float(r["test_auc"]))]
dup = len(keys) - len(set(keys))
seed_col = sum(r["seed"] != str(r["_seed"]) for r in rows)
report("1 completeness", len(rows) == 6000 and set(keys) == want and not dup and not bad_metric and not seed_col,
       f"{len(rows)} rows, {len(set(keys))} unique keys of {len(want)} expected, missing {len(want - set(keys))}, extra "
       f"{len(set(keys) - want)}, duplicates {dup}, rc!=0 or no/NaN AUC {len(bad_metric)}, seed column != dir {seed_col}")
# 2. same data: col-rep cells vs the screen's r = 0 ({W}:SPO) and single-machine (SPO, AA) cells
sha = {}
for r in rows:
    sha.setdefault((r["dataset"], r["fold"]), set()).add(("colrep", r["data_sha"]))
scr = [r for p in glob.glob(f"{SCREEN}/kdd28_*/seed*/distacc_stamped/results.csv") for r in read(p) if r["dataset"] in SETS]
for r in scr:
    sha.setdefault((r["dataset"], r["fold"]), set()).add((r["arm"].split(":")[0] != "single" and "r0" or "single", r["data_sha"]))
n_bad = sum(len({v for _, v in s}) != 1 or {k for k, _ in s} != {"colrep", "r0", "single"} for s in sha.values())
report("2 same data", len(sha) == 20 and n_bad == 0,
       f"{len(sha)} (dataset, fold) pairs, {n_bad} with more than one data_sha or a missing source (colrep, r0, single)")
# 3. same settings (stamps); BINARY + effective W/r from the logs when given
exp = {"profile": "kdd28_hao.colrep_distacc_spo", "trees": "300", "max_depth": "6", "min_examples": "5", "split_type": "EXACT",
       "spo_exponent": "1", "spo_max_projections": "1000", "spo_density": "1.5", "split_budget": "true",
       "local_projection_count": "false", "validation_set_ratio": "", "early_stopping": "", "bin_sha": "802b16fdf3b2",
       "eval_suite": "kdd28_wide_real_candidates", "metric_impl": "ydf_eval_v1"}
off = [(k, r[k]) for r in rows for k, v in exp.items() if r[k] != v]
thr = sum(int(r["threads"]) != 2 * int(r["workers"]) for r in rows)
drift = sum("drift" in r["profile"] for r in rows)
msg = f"stamp mismatches {len(off)} {sorted(set(off))[:4]}, threads != 2W {thr}, drift stamps {drift}"
ok3 = not off and not thr and not drift
if a.logs:
    nb = nw = 0
    for r in rows:
        txt = open(os.path.join(a.logs, f"seed{r['_seed']}", "colrep_distacc_stamped", "logs", r["log"]), errors="replace").read()
        eff = [l for l in txt.splitlines() if "effective-params:" in l]
        nb += not eff or "spo_weights=BINARY" not in eff[-1]
        nw += not eff or f"num_workers={r['workers']}" not in eff[-1] or f"column_replication={float(r['replication']):g}" not in eff[-1]
    msg += f", logs without BINARY {nb}, logs with other W or r {nw}"
    ok3 = ok3 and not nb and not nw
report("3 same settings", ok3, msg)
# 4. replica arithmetic per (dataset, W, r): owned sum = F, owned spread <= 1, replicas = lround(float32(r) * non-owned)
cfg = {}
for r in rows:
    cfg.setdefault((r["dataset"], int(r["workers"]), float(r["replication"])), set()).add(r["held_cols"])
bad4, notes = [], []
for (d, w, rr), hs in sorted(cfg.items()):
    if len(hs) != 1:
        bad4.append((d, w, rr, "held_cols differ across folds/seeds")); continue
    pairs = [tuple(map(int, x.split("+"))) for x in next(iter(hs)).split(";")]
    own = [o for o, _ in pairs]
    f32 = float(np.float32(rr))
    want_rep = [int(math.floor(f32 * (SETS[d] - o) + 0.5)) for o in own]  # std::lround, positive argument
    if len(pairs) != w or sum(own) != SETS[d] or max(own) - min(own) > 1 or [p for _, p in pairs] != want_rep:
        bad4.append((d, w, rr, next(iter(hs))[:60]))
    for o in set(own):  # configs where float32 rounding changes the decimal answer
        n = SETS[d] - o
        if int(math.floor(f32 * n + 0.5)) != int(math.floor(rr * n + 0.5 + 1e-12)):
            notes.append(f"{d} W={w} r={rr}: n={n} float32 gives {int(math.floor(f32 * n + 0.5))}, decimal {int(math.floor(rr * n + 0.5 + 1e-12))}")
report("4 replica arithmetic", len(cfg) == 240 and not bad4,
       f"{len(cfg)} configs, {len(bad4)} bad {bad4[:3]}. Configs where the float32 rule differs from decimal rounding: {len(set(notes))}"
       + (f" (e.g. {sorted(set(notes))[0]})" if notes else ""))
# 7. r = 1 vs single-machine SPO, per dataset and W: mean over seeds of the fold-mean gap, seed sd, flag > 2 SE
single = {(r["dataset"], r["fold"], r["seed"]): float(r["test_auc"]) for r in scr
          if r["arm"] == "single:SPO" and r["profile"] == "kdd28_hao.distacc_single_spo"}
r0 = {(r["dataset"], r["fold"], r["seed"], int(r["arm"].split(":")[0])): float(r["test_auc"]) for r in scr
      if r["arm"].endswith(":SPO") and r["arm"] != "single:SPO"}
auc = {(r["dataset"], r["fold"], str(r["_seed"]), int(r["workers"]), float(r["replication"])): float(r["test_auc"]) for r in rows}
out, flags, lines = [], [], []
for d in SETS:
    for w in WS:
        for rr in [0.0] + RS:
            get = (lambda f, s: r0[(d, f, str(s), w)]) if rr == 0 else (lambda f, s: auc[(d, f, str(s), w, rr)])
            per_seed = [st.mean(get(f, s) - single[(d, f, str(s))] for f in FOLDS) for s in SEEDS]
            m, sd = st.mean(per_seed), st.stdev(per_seed)
            out.append({"dataset": d, "workers": w, "replication": rr, "mean_delta_auc": f"{m:.6f}", "seed_sd": f"{sd:.6f}",
                        "n_cells": len(FOLDS) * len(SEEDS), "source": "screen {W}:SPO cells" if rr == 0 else "this run"})
            if rr == 1.0:
                se = sd / math.sqrt(len(SEEDS))
                lines.append(f"{d} W={w}: {m:+.4f} (seed sd {sd:.4f}, {m / se:+.1f} SE)")
                if abs(m) > 2 * se:
                    flags.append(f"{d} W={w}")
report("7 r=1 vs single SPO", True, "; ".join(lines) + f". Flagged (> 2 SE from 0): {', '.join(flags) or 'none'}")
os.makedirs(f"{H}/derived", exist_ok=True)
with open(f"{H}/derived/colrep_mean_delta.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(out[0])); wr.writeheader(); wr.writerows(out)
print("\n".join(res))
raise SystemExit(1 if fails else 0)
