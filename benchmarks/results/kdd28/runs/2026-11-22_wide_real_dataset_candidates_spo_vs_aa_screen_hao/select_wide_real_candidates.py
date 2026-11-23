#!/usr/bin/env python3
# Rule F of select_spo_clearly_beats_aa.py applied to the wide real-dataset candidates (kdd28_hao SPO vs kdd28_ydf AA).
# BH (a) over the new rows only, (b) pooled with the existing kdd28_hao rows of derived/spo_clearly_beats_aa_per_dataset.csv.
# Writes derived/{candidates_per_dataset.csv,candidates_report.txt,r0_gap.csv}. Reads results only. Run from the repo root.
import csv, glob, math, os, statistics as st
from scipy import stats

H = os.path.dirname(os.path.abspath(__file__))
OLD = "benchmarks/results/kdd28/floats/accuracy/tab-accuracy-hao-vs-ydf-params-single-machine/derived/spo_clearly_beats_aa_per_dataset.csv"
AUC_FLOOR, REG_FLOOR = 0.005, 0.01
# Excluded from the selection (user decision 2026-11-22). The rule columns pass_a/pass_b and the BH families stay as run.
# Evidence: offpaper_<dataset>_patient_grouped_folds/, tested below with the same effect and t.
EXCLUDED = {"ct_slices": "grouped by patient (97 patients), plain KFold puts slices of one patient in train and test"}
read =lambda p: list(csv.DictReader(open(p, newline="")))


def tstats(x):  # copied from select_spo_clearly_beats_aa.py: mean, sd, t, one-sided p (H1 mean > 0)
    n = len(x); m = st.mean(x)
    if n < 2: return m, None, None, None
    sd = st.stdev(x); se = sd / math.sqrt(n)
    if se == 0 and m == 0: return m, sd, 0.0, 1.0
    t = m / se if se > 0 else math.copysign(math.inf, m)
    return m, sd, t, float(stats.t.sf(t, n - 1))


def bh(items, q=0.05):  # items: [(p, key)], same step-up as select_spo_clearly_beats_aa.py
    ps = sorted((p, k) for p, k in items if p is not None); n = len(ps)
    kmax = max([i + 1 for i, (p, _) in enumerate(ps) if p <= q * (i + 1) / n], default=0)
    return {k for _, k in ps[:kmax]}, n, (ps[kmax - 1][0] if kmax else None)


def load(table, arm, metric, prof):
    out, sha, secs = {}, {}, {}
    for p in glob.glob(f"{H}/{prof}/seed*/{table}/results.csv"):
        for r in read(p):
            if r["rc"] == "0" and r["arm"] == arm:
                out.setdefault(r["dataset"], {}).setdefault(int(r["seed"]), {})[r["fold"]] = float(r[metric])
                sha[r["dataset"], r["fold"]] = r["data_sha"]
                secs.setdefault((r["dataset"], r["arm"]), []).append(float(r["train_s"]))
    return out, sha, secs


rows = []
for table, metric in (("distacc_stamped", "test_auc"), ("reg", "test_rmse")):
    spo, sha_s, _ = load(table, "single:SPO", metric, "kdd28_hao")
    aa, sha_a, _ = load(table, "single:AA", metric, "kdd28_ydf")
    assert all(sha_s[k] == sha_a[k] for k in sha_s), table  # same fold files in both arms
    reg = metric == "test_rmse"
    for d in sorted(spo, key=str.lower):
        seeds = sorted(s for s in spo[d] if s in aa.get(d, {}))
        folds = sorted(spo[d][seeds[0]])
        assert all(sorted(spo[d][s]) == folds == sorted(aa[d][s]) for s in seeds), d
        eff = {s: {f: ((aa[d][s][f] - spo[d][s][f]) / aa[d][s][f] if reg else spo[d][s][f] - aa[d][s][f]) for f in folds} for s in seeds}
        es = [st.mean(eff[s].values()) for s in seeds]
        fold_avg = [st.mean(eff[s][f] for s in seeds) for f in folds]
        m, fsd, t, p = tstats(fold_avg)
        _, ssd, ts, ps = tstats(es)
        k, ns = len(folds), len(seeds)
        r = dict(suite="wide_real_candidates", metric="rel_rmse_red" if reg else "auc_edge", dataset=d, folds=k, seeds=ns,
                 aa_mean=st.mean(st.mean(aa[d][s].values()) for s in seeds), spo_mean=st.mean(st.mean(spo[d][s].values()) for s in seeds),
                 effect=m, effect_seed1=es[0], floor=REG_FLOOR if reg else AUC_FLOOR, seed_min=min(es), seed_max=max(es),
                 seeds_pos=sum(e > 0 for e in es), seeds_over=sum(e > (REG_FLOOR if reg else AUC_FLOOR) for e in es),
                 se=(fsd / math.sqrt(k) if k > 1 else (ssd / math.sqrt(ns) if ns > 1 else None)),
                 noise="folds" if k > 1 else ("seeds" if ns > 1 else "none"), p_rule=p if k > 1 else ps,
                 cells_won=sum(v > 0 for s in seeds for v in eff[s].values()), cells=k * ns, seed_effects=es)
        r["valid"] = reg or min(r["aa_mean"], r["spo_mean"]) > 0.5
        rows.append(r)

old = [r for r in read(OLD) if r["profile"] == "kdd28_hao"]
for r in old:
    r["p_rule"] = float(r["p_rule"]) if r["p_rule"] else None
# sanity: our BH over the old rows alone reproduces the stored bh column
ok_old, n_old, _ = bh([(r["p_rule"], ("old", r["suite"], r["dataset"])) for r in old])
assert all((("old", r["suite"], r["dataset"]) in ok_old) == (r["bh"] == "True") for r in old), "BH reimplementation drift"
floor_ok = lambda r: r["valid"] and r["effect"] > r["floor"]
ok_a, n_a, pmax_a = bh([(r["p_rule"], r["dataset"]) for r in rows])
ok_b, n_b, pmax_b = bh([(r["p_rule"], r["dataset"]) for r in rows] + [(r["p_rule"], ("old", r["suite"], r["dataset"])) for r in old])
for r in rows:
    r["bh_a"], r["bh_b"] = r["dataset"] in ok_a, r["dataset"] in ok_b
    allpos = r["seeds_pos"] == r["seeds"]
    r["pass_a"], r["pass_b"] = floor_ok(r) and r["bh_a"] and allpos, floor_ok(r) and r["bh_b"] and allpos
    r["excluded"] = EXCLUDED.get(r["dataset"], "")
    r["selected_a"], r["selected_b"] = r["pass_a"] and not r["excluded"], r["pass_b"] and not r["excluded"]
assert set(EXCLUDED) <= {r["dataset"] for r in rows}, "EXCLUDED names a dataset that was not screened"
flips = []
for r in old:
    newbh = ("old", r["suite"], r["dataset"]) in ok_b
    was = r["pass_rec"] == "True"
    now = r["valid"] == "True" and float(r["effect"]) > float(r["floor"]) and newbh and int(r["seeds_pos"]) == int(r["seeds"])
    if was != now:
        flips.append(f"{r['dataset']}[{r['suite']}] {was}->{now}")

os.makedirs(os.path.join(H, "derived"), exist_ok=True)
cols = ["suite", "metric", "dataset", "folds", "seeds", "aa_mean", "spo_mean", "effect", "effect_seed1", "floor", "valid", "noise",
        "se", "p_rule", "seed_min", "seed_max", "seeds_over", "seeds_pos", "cells_won", "cells", "bh_a", "bh_b", "pass_a", "pass_b",
        "selected_a", "selected_b", "excluded"]
with open(os.path.join(H, "derived", "candidates_per_dataset.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
    for r in rows: w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else ("" if v is None else v)) for k, v in r.items()})
L = [f"Rule F (floor AND BH q=0.05 on the one-sided paired t AND effect > 0 in every seed). SPO = kdd28_hao.distacc_single_spo, AA = kdd28_ydf.distacc_single_aa.",
     f"(a) BH over the {n_a} new rows with a p-value: largest accepted p {pmax_a}. (b) pooled with {len(old)} existing kdd28_hao rows ({n_b} p-values): largest accepted p {pmax_b}.",
     f"BH reimplementation reproduces the stored kdd28_hao bh column on all {len(old)} existing rows.",
     f"Existing kdd28_hao rows whose rule-F verdict flips under pooling: {', '.join(flips) or 'none'}.", ""]
L.append(f"{'dataset':16s} {'metric':12s} k s {'AA':>8s} {'SPO':>8s} {'effect':>8s} {'seed1':>8s} {'SE':>7s} {'p':>7s} pos  (a)  (b)")
for r in sorted(rows, key=lambda r: -r["effect"]):
    L.append(f"{r['dataset'][:16]:16s} {r['metric']:12s} {r['folds']} {r['seeds']} {r['aa_mean']:8.4f} {r['spo_mean']:8.4f} {r['effect']:+8.4f} "
             f"{r['effect_seed1']:+8.4f} {('-' if r['se'] is None else format(r['se'], '.4f')):>7s} {('-' if r['p_rule'] is None else format(r['p_rule'], '.4f')):>7s} "
             f"{r['seeds_pos']}/{r['seeds']}  {'YES' if r['pass_a'] else ' - '}  {'YES' if r['pass_b'] else ' - '}{'' if r['valid'] else '  INVALID (AUC <= .5)'}"
             f"{'  EXCLUDED from the selection, see below' if r['excluded'] else ''}")
L += ["", "Selected (rule F and not excluded): (a) " + ", ".join(r["dataset"] for r in rows if r["selected_a"])
      + ". (b) " + ", ".join(r["dataset"] for r in rows if r["selected_b"]) + "."]
# ---- excluded sets: the same effect and one-sided paired t on their off-paper patient-grouped folds
for d, why in EXCLUDED.items():
    v = {}
    for prof, arm in (("kdd28_hao", "single:SPO"), ("kdd28_ydf", "single:AA")):
        for p in glob.glob(f"{H}/offpaper_{d}_patient_grouped_folds/{prof}/seed*/reg/results.csv"):
            for r in read(p):
                if r["rc"] == "0" and r["arm"] == arm: v.setdefault((int(r["seed"]), r["fold"]), {})[arm] = float(r["test_rmse"])
    assert v and all(len(x) == 2 for x in v.values()), f"{d}: off-paper cells missing an arm"
    seeds, folds = sorted({s for s, _ in v}), sorted({f for _, f in v})
    assert len(v) == len(seeds) * len(folds), f"{d}: off-paper cells not a full seed x fold grid"
    eff = lambda s, f: (v[s, f]["single:AA"] - v[s, f]["single:SPO"]) / v[s, f]["single:AA"]  # noqa: E731
    per_fold = [st.mean(eff(s, f) for s in seeds) for f in folds]
    m, _, t, p = tstats(per_fold)
    verdict = "p > q = 0.05, so BH rejects it at any family size" if p > 0.05 else "p <= q, the BH verdict depends on the family"
    L += ["", f"EXCLUDED {d}: {why}.",
          f"  Off-paper GroupKFold({len(folds)}) by patient, seeds {seeds} (seeds {[s for s in range(1, 6) if s not in seeds]} not run): "
          f"AA RMSE {st.mean(x['single:AA'] for x in v.values()):.4f}, SPO {st.mean(x['single:SPO'] for x in v.values()):.4f}, "
          f"effect {m:+.4f} (floor {REG_FLOOR}), folds won {sum(e > 0 for e in per_fold)}/{len(folds)}, t {t:.3f}, one-sided p {p:.4f}: {verdict}."]
# ---- passers: distributed SPO at r = 0 (kdd28_hao.distacc_spo) vs single SPO, and the cost of the full colrep r grid
R_GRID = [round(0.05 * i, 2) for i in range(1, 11)] + [0.6, 0.7, 0.8, 0.9, 1.0]
gap_rows = []
for table, dtable, metric in (("distacc_stamped", "distacc_stamped", "test_auc"), ("reg", "reg_dist", "test_rmse")):
    single, _, _ = load(table, "single:SPO", metric, "kdd28_hao")
    for W in (4, 8, 16, 32):
        dist, _, secs = load(dtable, f"{W}:SPO", metric, "kdd28_hao")
        for d in sorted(dist):
            gap = lambda x, y: (x - y) / y * 100 if metric == "test_rmse" else x - y  # noqa: E731
            by_seed = {s: [gap(dist[d][s][f], single[d][s][f]) for f in dist[d][s] if f in single.get(d, {}).get(s, {})] for s in dist[d]}
            g = [v for s in by_seed for v in by_seed[s]]
            pairs = g
            sec = secs[(d, f"{W}:SPO")]
            nf = len({f for s in dist[d] for f in dist[d][s]})
            gap_rows.append({"dataset": d, "metric": "rmse_pct" if metric == "test_rmse" else "auc_points", "workers": W,
                             "cells": len(pairs), "seeds": len(dist[d]), "folds": nf, "gap_mean": st.mean(g),
                             "gap_seed_sd": st.stdev([st.mean(v) for v in by_seed.values()]) if len(by_seed) > 1 else None,
                             "train_s_mean": st.mean(sec), "colrep_grid_cells": len(R_GRID) * 5 * nf,
                             "colrep_grid_cpu_h_est": len(R_GRID) * 5 * nf * st.mean(sec) * W * 2 / 3600})
if gap_rows:
    with open(os.path.join(H, "derived", "r0_gap.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(gap_rows[0])); w.writeheader()
        for r in gap_rows: w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else ("" if v is None else v)) for k, v in r.items()})
    L += ["", "Distributed SPO at r = 0 minus single SPO (AUC points, or % RMSE: + = worse), kdd28_hao, mean over seeds x folds.",
          "Colrep grid estimate: r in " + ",".join(map(str, R_GRID)) + ", 5 seeds, cells x measured r=0 train_s (shared box). CPU-h = cells x s x 2W threads.",
          f"{'dataset':16s} {'W':>3s} {'cells':>5s} {'gap':>9s} {'seedSD':>8s} {'train_s':>8s} {'grid_cells':>10s} {'grid_cpu_h':>10s}"]
    for r in gap_rows:
        L.append(f"{r['dataset'][:16]:16s} {r['workers']:3d} {r['cells']:5d} {r['gap_mean']:+9.4f} {('-' if r['gap_seed_sd'] is None else format(r['gap_seed_sd'], '.4f')):>8s} "
                 f"{r['train_s_mean']:8.1f} {r['colrep_grid_cells']:10d} {r['colrep_grid_cpu_h_est']:10.1f}")
open(os.path.join(H, "derived", "candidates_report.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
