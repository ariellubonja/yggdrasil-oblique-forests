#!/usr/bin/env python3
# Rule F of select_spo_clearly_beats_aa.py on the trunk screen (kdd28_hao SPO vs kdd28_ydf AA), BH (a) over the trunk rows,
# (b) pooled with the existing kdd28_hao rows. Writes derived/{trunk_per_dataset.csv,trunk_report.txt}. Run from the repo root.
import csv, glob, math, os, re, statistics as st
from scipy import stats

H = os.path.dirname(os.path.abspath(__file__))
OLD = "benchmarks/results/kdd28/floats/accuracy/tab-accuracy-hao-vs-ydf-params-single-machine/derived/spo_clearly_beats_aa_per_dataset.csv"
AUC_FLOOR = 0.005
NAME = re.compile(r"trunk_(shift|mix)_n(\d+)_f(\d+)_k(\d+)_auc(\d+)_rho(\d+)$")
read = lambda p: list(csv.DictReader(open(p, newline="")))


def tstats(x):  # copied from select_spo_clearly_beats_aa.py: mean, sd, t, one-sided p (H1 mean > 0)
    n = len(x); m = st.mean(x)
    if n < 2: return m, None, None, None
    sd = st.stdev(x); se = sd / math.sqrt(n)
    if se == 0 and m == 0: return m, sd, 0.0, 1.0
    t = m / se if se > 0 else math.copysign(math.inf, m)
    return m, sd, t, float(stats.t.sf(t, n - 1))


def bh(items, q=0.05):  # same step-up as select_spo_clearly_beats_aa.py
    ps = sorted((p, k) for p, k in items if p is not None); n = len(ps)
    kmax = max([i + 1 for i, (p, _) in enumerate(ps) if p <= q * (i + 1) / n], default=0)
    return {k for _, k in ps[:kmax]}, n, (ps[kmax - 1][0] if kmax else None)


def load(arm, prof):
    out, sha, secs = {}, {}, {}
    for p in glob.glob(f"{H}/{prof}/seed*/distacc_stamped/results.csv"):
        for r in read(p):
            if r["rc"] == "0" and r["arm"] == arm:
                out.setdefault(r["dataset"], {}).setdefault(int(r["seed"]), {})[r["fold"]] = float(r["test_auc"])
                sha[r["dataset"], r["fold"]] = r["data_sha"]
                secs.setdefault(r["dataset"], []).append(float(r["train_s"]))
    return out, sha, secs


spo, sha_s, t_spo = load("single:SPO", "kdd28_hao")
aa, sha_a, t_aa = load("single:AA", "kdd28_ydf")
assert all(sha_s[k] == sha_a[k] for k in sha_s if k in sha_a), "SPO and AA cells read different fold files"
rows = []
for d in sorted(spo):
    seeds = sorted(s for s in spo[d] if s in aa.get(d, {}))
    folds = sorted(spo[d][seeds[0]])
    assert all(sorted(spo[d][s]) == folds == sorted(aa[d][s]) for s in seeds), d
    eff = {s: {f: spo[d][s][f] - aa[d][s][f] for f in folds} for s in seeds}
    es = [st.mean(eff[s].values()) for s in seeds]
    m, fsd, t, p = tstats([st.mean(eff[s][f] for s in seeds) for f in folds])
    sim, n, F, k, auc, rho = NAME.match(d).groups()
    r = dict(dataset=d, sim=sim, rows=int(n), cols=int(F), k=int(k), bayes_auc=int(auc) / 100, rho=int(rho) / 10,
             folds=len(folds), seeds=len(seeds), aa_mean=st.mean(st.mean(aa[d][s].values()) for s in seeds),
             spo_mean=st.mean(st.mean(spo[d][s].values()) for s in seeds), effect=m, effect_seed1=es[0],
             se=fsd / math.sqrt(len(folds)), p_rule=p, seed_min=min(es), seeds_pos=sum(e > 0 for e in es),
             cells_won=sum(v > 0 for s in seeds for v in eff[s].values()), cells=len(folds) * len(seeds),
             aa_train_s=st.mean(t_aa[d]), spo_train_s=st.mean(t_spo[d]))
    r["valid"] = min(r["aa_mean"], r["spo_mean"]) > 0.5
    rows.append(r)

old = [r for r in read(OLD) if r["profile"] == "kdd28_hao"]
for r in old:
    r["p_rule"] = float(r["p_rule"]) if r["p_rule"] else None
ok_old, _, _ = bh([(r["p_rule"], ("old", r["suite"], r["dataset"])) for r in old])
assert all((("old", r["suite"], r["dataset"]) in ok_old) == (r["bh"] == "True") for r in old), "BH reimplementation drift"
ok_a, n_a, pmax_a = bh([(r["p_rule"], r["dataset"]) for r in rows])
ok_b, n_b, pmax_b = bh([(r["p_rule"], r["dataset"]) for r in rows] + [(r["p_rule"], ("old", r["suite"], r["dataset"])) for r in old])
for r in rows:
    gate = r["valid"] and r["effect"] > AUC_FLOOR and r["seeds_pos"] == r["seeds"]
    r["bh_a"], r["bh_b"] = r["dataset"] in ok_a, r["dataset"] in ok_b
    r["pass_a"], r["pass_b"] = gate and r["bh_a"], gate and r["bh_b"]

os.makedirs(os.path.join(H, "derived"), exist_ok=True)
with open(os.path.join(H, "derived", "trunk_per_dataset.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
    for r in rows: w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
L = ["Rule F (effect > 0.005 AUC AND BH q=0.05 on the one-sided paired t over seed-averaged folds AND effect > 0 in every seed).",
     "SPO = kdd28_hao.distacc_single_spo, AA = kdd28_ydf.distacc_single_aa.",
     f"(a) BH over {n_a} trunk rows: largest accepted p {pmax_a}. (b) pooled with {len(old)} existing kdd28_hao rows ({n_b}): {pmax_b}.",
     f"Passers: (a) {sum(r['pass_a'] for r in rows)}/{len(rows)}, (b) {sum(r['pass_b'] for r in rows)}/{len(rows)}.", "",
     f"{'dataset':42s} s {'AA':>7s} {'SPO':>7s} {'effect':>8s} {'SE':>7s} {'p':>8s} pos  (a) (b) {'AA_s':>7s} {'SPO_s':>7s}"]
for r in sorted(rows, key=lambda r: -r["effect"]):
    L.append(f"{r['dataset']:42s} {r['seeds']} {r['aa_mean']:7.4f} {r['spo_mean']:7.4f} {r['effect']:+8.4f} {r['se']:7.4f} {r['p_rule']:8.2g} "
             f"{r['seeds_pos']}/{r['seeds']}  {'Y' if r['pass_a'] else '-'}   {'Y' if r['pass_b'] else '-'}  {r['aa_train_s']:7.1f} {r['spo_train_s']:7.1f}")
# one-knob views: mean seed-1 effect and pass count per value, within the block that varies it
L += ["", "Mean effect and rule-F (a) passes per knob value (all trunks with that value):"]
for knob in ("sim", "rows", "cols", "k", "bayes_auc", "rho"):
    vals = sorted({r[knob] for r in rows})
    L.append(f"  {knob:9s} " + "  ".join(
        f"{v}: {st.mean(r['effect'] for r in rows if r[knob] == v):+.4f} ({sum(r['pass_a'] for r in rows if r[knob] == v)}/"
        f"{sum(r[knob] == v for r in rows)})" for v in vals))
open(os.path.join(H, "derived", "trunk_report.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
