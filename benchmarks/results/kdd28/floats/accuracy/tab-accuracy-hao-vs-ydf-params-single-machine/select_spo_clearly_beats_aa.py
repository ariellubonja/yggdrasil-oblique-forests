# Selection threshold for "single SPO-GBT beats AA-GBT" across binary AUC (SPAA eval, wide, CC18, multiclass as
# majority-vs-rest) and regression RMSE (KDD28). Reads existing results only. Run from the repo root.
# Writes derived/spo_clearly_beats_aa_{report.txt,per_dataset.csv} (2026-11-20 Fable-high analysis, tighter tolerance).
import csv, itertools, math, os, statistics as st
from scipy import stats

H = os.path.join(os.path.dirname(os.path.abspath(__file__)), "derived")
R = "benchmarks/results/kdd28/runs"
SELECTED = "benchmarks/results/kdd28/floats/accuracy/tab-accuracy-where-spo-outperforms-aa/derived/spo_advantage_datasets.txt"
SEEDS = (1, 2, 3, 4, 5)
AUC_FLOOR = 0.005   # absolute test-AUC edge (the paper's current cutoff)
REG_FLOOR = 0.01    # relative RMSE reduction (AA - SPO) / AA, calibrated below against AUC_FLOOR
read = lambda p: list(csv.DictReader(open(p, newline="")))

def load(path, arm, metric, seed=None):
    # {fold: value} per dataset and {(dataset, fold): data_sha}. rc == "0" only, last row wins.
    val, sha = {}, {}
    for r in read(path):
        if r["rc"] == "0" and r["arm"] == arm and (seed is None or r["seed"] == str(seed)):
            val.setdefault(r["dataset"], {})[r["fold"]] = float(r[metric])
            sha[r["dataset"], r["fold"]] = r["data_sha"]
    return val, sha

X, C = f"{R}/2026-11-20_regression_multiclass_tabarena_tabred", f"{R}/2026-11-20_cc18_seeds"
HS, HW = f"{R}/2026-11-06_hao_seeds/distacc/results.csv", f"{R}/2026-11-06_hao_seeds/wide_eval/distacc/results.csv"
ydf_spaa = lambda s: f"{R}/2026-11-03_distacc/results.csv" if s == 1 else f"{R}/2026-11-20_ydf_seeds/seed{s}/distacc/results.csv"
# (profile, suite) -> (metric, spo_arm, aa_arm, {seed: (spo_path, aa_path)}). Hao AA of the extra sets = the YDF AA run.
SRC = {
    ("kdd28_ydf", "spaa"): ("test_auc", "single:SPO", "single:AA", {s: (ydf_spaa(s),) * 2 for s in SEEDS}),
    ("kdd28_ydf", "wide"): ("test_auc", "single:SPO", "single:AA", {1: (f"{R}/2026-11-04_wide_eval/distacc/results.csv",) * 2}),
    ("kdd28_ydf", "cc18"): ("test_auc", "cc18_spo", "cc18_aa", {s: (f"{C}/kdd28_ydf/seed{s}/cc18/results.csv",) * 2 for s in SEEDS}),
    ("kdd28_ydf", "multiclass"): ("test_auc", "single:SPO", "single:AA", {s: (f"{X}/kdd28_ydf/seed{s}/distacc/results.csv",) * 2 for s in SEEDS}),
    ("kdd28_ydf", "regression"): ("test_rmse", "single:SPO", "single:AA", {s: (f"{X}/kdd28_ydf/seed{s}/reg/results.csv",) * 2 for s in SEEDS}),
    ("kdd28_hao", "spaa"): ("test_auc", "single:SPO", "single:AA", {s: (HS, HS) for s in SEEDS}),
    ("kdd28_hao", "wide"): ("test_auc", "single:SPO", "single:AA", {s: (HW, HW) for s in SEEDS}),
    ("kdd28_hao", "cc18"): ("test_auc", "cc18_spo", "cc18_aa", {s: (f"{C}/kdd28_hao/seed{s}/cc18/results.csv",) * 2 for s in SEEDS}),
    ("kdd28_hao", "multiclass"): ("test_auc", "single:SPO", "single:AA",
                                  {s: (f"{X}/kdd28_hao/seed{s}/distacc/results.csv", f"{X}/kdd28_ydf/seed{s}/distacc/results.csv") for s in SEEDS}),
    ("kdd28_hao", "regression"): ("test_rmse", "single:SPO", "single:AA",
                                  {s: (f"{X}/kdd28_hao/seed{s}/reg/results.csv", f"{X}/kdd28_ydf/seed{s}/reg/results.csv") for s in SEEDS}),
}
MULTI_SEED_FILE = {HS, HW}

def tstats(x):
    # mean, sd, t, one-sided p (H1: mean > 0), mirrored p (H1: mean < 0) of a list. SE = sd / sqrt(len)
    n = len(x); m = st.mean(x)
    if n < 2: return m, None, None, None, None
    sd = st.stdev(x); se = sd / math.sqrt(n)
    if se == 0 and m == 0: return m, sd, 0.0, 1.0, 1.0   # all-equal arms (tic-tac-toe): no evidence either way
    t = m / se if se > 0 else math.copysign(math.inf, m)
    return m, sd, t, float(stats.t.sf(t, n - 1)), float(stats.t.sf(-t, n - 1))

out = []
def P(*a): out.append(" ".join(str(v) for v in a))
rows, sha_report, shas = [], [], {}
for (prof, suite), (metric, spo_arm, aa_arm, paths) in SRC.items():
    spo, aa, sha = {}, {}, {}
    for s, (ps, pa) in paths.items():
        spo[s], sha[s] = load(ps, spo_arm, metric, s if ps in MULTI_SEED_FILE else None)
        aa[s], sha_aa = load(pa, aa_arm, metric, s if pa in MULTI_SEED_FILE else None)
        assert spo[s].keys() == aa[s].keys(), (prof, suite, s)
        assert all(sha[s][k] == sha_aa[k] for k in sha[s]), (prof, suite, s)  # SPO and AA cells read the same fold files
    s0 = min(paths)
    mism = sum(1 for s in paths for k, v in sha[s0].items() if sha[s].get(k) != v) + sum(len(sha[s]) != len(sha[s0]) for s in paths)
    sha_report.append((prof, suite, len(paths), len(spo[s0]), len(sha[s0]), mism))
    shas[prof, suite] = sha[s0]
    reg = metric == "test_rmse"
    floor = REG_FLOOR if reg else AUC_FLOOR
    for d in sorted(spo[s0], key=str.lower):
        folds = sorted(spo[s0][d])
        assert all(sorted(spo[s][d]) == folds and sorted(aa[s][d]) == folds for s in paths), (prof, suite, d)
        # effect per cell: AUC edge SPO - AA, or relative RMSE reduction (AA - SPO) / AA. Positive = SPO better.
        eff = {s: {f: ((aa[s][d][f] - spo[s][d][f]) / aa[s][d][f] if reg else spo[s][d][f] - aa[s][d][f]) for f in folds} for s in paths}
        es = [st.mean(eff[s].values()) for s in paths]; k, ns = len(folds), len(paths)
        fold_avg = [st.mean(eff[s][f] for s in paths) for f in folds]  # seed-averaged paired fold effects
        m, fsd, t, p, pm = tstats(fold_avg)
        nb = math.sqrt(1 / k + 1 / (k - 1)) / math.sqrt(1 / k) if k > 1 else None  # Nadeau-Bengio SE inflation for k-fold CV
        _, ssd, ts, psd, psm = tstats(es)
        row = dict(profile=prof, suite=suite, metric="rel_rmse_red" if reg else "auc_edge", dataset=d, folds=k, seeds=ns,
                   aa_mean=st.mean(st.mean(aa[s][d].values()) for s in paths), spo_mean=st.mean(st.mean(spo[s][d].values()) for s in paths),
                   effect=m, effect_seed1=es[0], floor=floor,
                   seed_sd=ssd, seed_se=(ssd / math.sqrt(ns) if ns > 1 else None), seed_min=min(es), seed_max=max(es),
                   seeds_over=sum(e > floor for e in es), seeds_pos=sum(e > 0 for e in es), seeds_under=sum(e < -floor for e in es),
                   fold_sd=fsd, se=(fsd / math.sqrt(k) if k > 1 else None), se_nb=(fsd / math.sqrt(k) * nb if k > 1 else None),
                   t=t, p=p, t_nb=(t / nb if k > 1 else None), p_nb=(float(stats.t.sf(t / nb, k - 1)) if k > 1 else None),
                   t_seed=ts, p_seed=psd, cells_won=sum(v > 0 for s in paths for v in eff[s].values()), cells=k * ns,
                   aa_seed_sd=(st.stdev(st.mean(aa[s][d].values()) for s in paths) if ns > 1 else None),
                   spo_seed_sd=(st.stdev(st.mean(spo[s][d].values()) for s in paths) if ns > 1 else None),
                   spo_seed_means=[st.mean(spo[s][d].values()) for s in paths], aa_seed_means=[st.mean(aa[s][d].values()) for s in paths],
                   seed_effects=es)
        # noise SE the rule uses: folds when k > 1, else seeds (training randomness only), else none
        row["noise"] = "folds" if k > 1 else ("seeds" if ns > 1 else "none")
        row["se_rule"] = row["se"] if k > 1 else row["seed_se"]
        row["p_rule"] = p if k > 1 else psd
        row["p_mirror"] = pm if k > 1 else psm
        row["valid"] = reg or min(row["aa_mean"], row["spo_mean"]) > 0.5   # AUC <= .5 = no better than chance: edge not interpretable
        rows.append(row)

sel_now = [l.split("\t")[0] for l in open(SELECTED).read().splitlines() if l]
names = {}
for r in rows: names.setdefault(r["profile"], {}).setdefault(r["dataset"], []).append(r["suite"])
for r in rows:
    r["also_in"] = ",".join(s for s in names[r["profile"]][r["dataset"]] if s != r["suite"])
    r["in_current_list"] = r["dataset"] in sel_now and r["suite"] in ("spaa", "wide")
    r["label"] = f"{r['dataset']}[{r['suite']}]"

def bh(rs, key, q=0.05):
    # Benjamini-Hochberg step-up over the rows that have a p-value. Returns the ids that pass and the family size.
    ps = sorted((r[key], id(r)) for r in rs if r[key] is not None); n = len(ps)
    kmax = max([i + 1 for i, (p, _) in enumerate(ps) if p <= q * (i + 1) / n], default=0)
    return {i for _, i in ps[:kmax]}, n, (ps[kmax - 1][0] if kmax else None)
BHINFO = {}
for prof in ("kdd28_ydf", "kdd28_hao"):
    rs = [r for r in rows if r["profile"] == prof]
    for r in rs: r["p_fold"] = r["p_rule"] if r["folds"] > 1 else None
    for key, name, q in (("p_rule", "bh", 0.05), ("p_rule", "bh10", 0.10), ("p_nb", "bh_nb", 0.05), ("p_mirror", "bh_mirror", 0.05), ("p_fold", "bh_fold", 0.05)):
        ok, n, pmax = bh(rs, key, q)
        BHINFO[prof, name] = (n, len(ok), pmax)
        for r in rs: r[name] = id(r) in ok

# ---- rules. All use the 5-seed mean effect where 5 seeds exist (YDF wide: seed 1 only).
def floor_ok(r): return r["valid"] and r["effect"] > r["floor"]
def guard(r, mult=2.0, key="se_rule"): return r[key] is not None and r["effect"] > mult * r[key]
allpos = lambda r: r["seeds_pos"] == r["seeds"]
RULES = {
    "A  floor only (5-seed mean)": floor_ok,
    "B  floor AND > 2 SE (folds, or seeds on 1-split sets)": lambda r: floor_ok(r) and guard(r),
    "C  B AND effect > 0 in every seed": lambda r: floor_ok(r) and guard(r) and allpos(r),
    "D  C AND effect > floor in >= 4/5 seeds (needs 5 seeds)": lambda r: floor_ok(r) and guard(r) and r["seeds"] == 5 and r["seeds_over"] >= 4,
    "E  floor AND > 2 SE Nadeau-Bengio (multi-fold only)": lambda r: floor_ok(r) and guard(r, key="se_nb"),
    "F  floor AND BH q=0.05 on paired t AND effect > 0 in every seed": lambda r: floor_ok(r) and r["bh"] and allpos(r),
    "F' as F with BH q=0.10": lambda r: floor_ok(r) and r["bh10"] and allpos(r),
    "G  floor AND BH q=0.05 on Nadeau-Bengio t (multi-fold only)": lambda r: floor_ok(r) and r["bh_nb"],
    "H  floor AND one-sided p < 0.05 (uncorrected)": lambda r: floor_ok(r) and r["p_rule"] is not None and r["p_rule"] < 0.05,
    "I  effect > floor in every seed (no fold test)": lambda r: r["valid"] and r["seeds"] == 5 and r["seeds_over"] == 5,
}
REC, ALT = "F  floor AND BH q=0.05 on paired t AND effect > 0 in every seed", "C  B AND effect > 0 in every seed"
def mirror(r):  # the recommended rule with the sign flipped: AA beats SPO
    return r["valid"] and -r["effect"] > r["floor"] and r["bh_mirror"] and r["seeds_pos"] == 0
def mirror_alt(r):  # the alternative rule with the sign flipped
    return r["valid"] and -r["effect"] > r["floor"] and r["se_rule"] is not None and -r["effect"] > 2 * r["se_rule"] and r["seeds_pos"] == 0
for r in rows:
    r["pass_rec"], r["pass_alt"], r["aa_wins_rec"], r["aa_wins_alt"] = RULES[REC](r), RULES[ALT](r), mirror(r), mirror_alt(r)

f4 = lambda v, w=7: " " * (w - 1) + "-" if v is None else f"{v:+.4f}"
u4 = lambda v: "   -  " if v is None else f"{v:.4f}"
u2 = lambda v: "   - " if v is None or math.isinf(v) else f"{v:5.1f}"
SUITES = ("spaa", "wide", "cc18", "multiclass", "regression")
by = lambda prof, suite=None: [r for r in rows if r["profile"] == prof and (suite is None or r["suite"] == suite)]
lab = lambda rs: ", ".join(sorted((r["label"] for r in rs), key=str.lower)) or "-"
Y = by("kdd28_ydf")

body = out; out = []   # sections 1-6 are built first, the summary (section 0) is written last and printed first
P("1. Inputs and fold-file consistency (data_sha per dataset x fold, every seed vs the first seed, and SPO vs AA cells)")
for prof, suite, ns, nd, ncell, mism in sha_report:
    P(f"  {prof:9s} {suite:10s} seeds {ns}  datasets {nd:2d}  dataset-folds {ncell:3d}  data_sha mismatches {mism}")
cross = [(s, sum(1 for k, v in shas["kdd28_ydf", s].items() if shas["kdd28_hao", s].get(k) != v)) for s in SUITES]
P("  kdd28_hao vs kdd28_ydf fold files (data_sha mismatches): " + ", ".join(f"{s} {n}" for s, n in cross))
uniq = {p: len(names[p]) for p in names}
overlap = sorted(d for d, s in names["kdd28_ydf"].items() if len(s) > 1)
P(f"  kdd28_ydf rows {len(Y)} over {uniq['kdd28_ydf']} dataset names. In both SPAA eval (5 folds) and CC18 (10 folds, other fold files): " + ", ".join(overlap))
P("  YDF wide: seed 1 only (runs/2026-11-04_wide_eval). Epsilon: 1 split and 1 seed under kdd28_ydf, so no noise estimate from these cells")
P("  Hao seed-1 single:SPO rows of the SPAA and wide sets are the adopted harness_exact300 rows (exponent 1, max 1000, density 1.5, the kdd28_hao values)")

P("\n2. Calibrating the regression floor against the AUC floor of 0.005 (kdd28_ydf, 5-seed sets)")
auc5 = [r for r in Y if r["metric"] == "auc_edge" and r["seeds"] == 5]
regall = [r for r in Y if r["metric"] == "rel_rmse_red"]
reg5 = [r for r in regall if r["dataset"] != "sberbank-housing"]
med = lambda rs, k: st.median(r[k] for r in rs if r[k] is not None)
P(f"  AUC sets n={len(auc5)}: median seed SD of the edge {med(auc5, 'seed_sd'):.4f}, median fold SE {med(auc5, 'se'):.4f}, median NB SE {med(auc5, 'se_nb'):.4f}")
P(f"  regression n={len(reg5)} (sberbank-housing left out, see 6): median seed SD of the relative reduction {med(reg5, 'seed_sd'):.4f}, "
  f"median fold SE {med(reg5, 'se'):.4f}, median NB SE {med(reg5, 'se_nb'):.4f}. With sberbank-housing (n=18): seed SD {med(regall, 'seed_sd'):.4f}")
ra, rr = AUC_FLOOR / med(auc5, "seed_sd"), AUC_FLOOR / med(auc5, "se")
m_sd, m_se = ra * med(reg5, "seed_sd"), rr * med(reg5, "se")
P(f"  0.005 AUC = {ra:.1f} x median seed SD = {rr:.1f} x median fold SE. The same multiples on regression: {m_sd:.4f} (seed SD), {m_se:.4f} (fold SE)")
# same-method null: |SPO seed s - SPO seed s'| of the fold-mean metric (relative to the AA mean for RMSE)
def null(rs, rel):
    return sorted(abs(a - b) / (r["aa_mean"] if rel else 1) for r in rs for a, b in itertools.combinations(r["spo_seed_means"], 2))
na, nr = null(auc5, False), null(reg5, True)
qa = sum(v <= AUC_FLOOR for v in na) / len(na)
qr = nr[min(len(nr) - 1, int(qa * len(nr)))]
qreg = sum(v <= REG_FLOOR for v in nr) / len(nr)
P(f"  same-method null |SPO seed s - SPO seed s'| of the fold-mean metric: AUC {len(na)} seed pairs, {qa:.1%} are <= 0.005. The same quantile of the "
  f"relative RMSE null ({len(nr)} pairs) is {qr:.4f}. Share of RMSE null pairs <= {REG_FLOOR}: {qreg:.1%}")
rel_auc = AUC_FLOOR / st.median(1 - r["aa_mean"] for r in auc5)
P(f"  relative-error view (not matched): 0.005 AUC = {rel_auc:.1%} of the median AUC error (1 - AA AUC). 1 % RMSE = about 2 % MSE. "
  "The floors are matched on noise, not on relative error")
P("  regression floor sweep (floor only, 5-seed mean, of 17): " + ", ".join(
    f"> {th:.4f}: {sum(r['effect'] > th for r in reg5)}" for th in (0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.05)))
P(f"  chosen: relative RMSE reduction > {REG_FLOOR} (1 %). Any floor from 0.0075 to 0.015 selects the same regression sets")

for prof in ("kdd28_ydf", "kdd28_hao"):
    P(f"\n3. Per-dataset table, {prof} (effect = {'5-seed' if prof == 'kdd28_hao' else '5-seed (wide: seed 1)'} mean. AUC edge SPO-AA, or relative RMSE reduction. "
      "SE over seed-averaged folds, '~' = over seeds on a 1-split set. p = one-sided paired t. REC/ALT = passes the recommended/alternative rule, aa = AA wins by the mirrored rule)")
    for suite in SUITES:
        P(f" [{suite}]")
        P(f"  {'dataset':38s} k  s {'AA':>9s} {'SPO':>9s} {'effect':>7s} {'seed1':>7s} {'SE':>7s} {'eff/SE':>6s} {'SE_NB':>6s} {'p':>6s} {'seedSD':>6s} {'seed range':>17s} >fl >0  cells REC ALT")
        for r in sorted(by(prof, suite), key=lambda r: -r["effect"]):
            se = (u4(r["se_rule"]) + ("~" if r["noise"] == "seeds" else " "))
            ratio = None if not r["se_rule"] else r["effect"] / r["se_rule"]
            mv = lambda v: f"{v:9.4f}" if v < 100 else f"{v:9.1f}"
            P(f"  {r['dataset'][:38]:38s} {r['folds']:2d} {r['seeds']} {mv(r['aa_mean'])} {mv(r['spo_mean'])} {r['effect']:+.4f} {r['effect_seed1']:+.4f} "
              f"{se} {u2(ratio)} {u4(r['se_nb'])} {('  -   ' if r['p_rule'] is None else format(r['p_rule'], '6.4f'))} {u4(r['seed_sd'])} "
              f"[{r['seed_min']:+.4f},{r['seed_max']:+.4f}] {r['seeds_over']}/{r['seeds']} {r['seeds_pos']}/{r['seeds']} {r['cells_won']:2d}/{r['cells']:2d} "
              f"{'YES' if r['pass_rec'] else ('aa ' if r['aa_wins_rec'] else '   ')} {'YES' if r['pass_alt'] else '   '}{'' if r['valid'] else '  INVALID (AUC <= .5)'}")

for prof in ("kdd28_ydf", "kdd28_hao"):
    rs = by(prof)
    n, nd, pmax = BHINFO[prof, "bh"]; nn, ndn, pmaxn = BHINFO[prof, "bh_nb"]
    P(f"\n4. Rule comparison, {prof} ({len(rs)} dataset rows)")
    P(f"  BH q=0.05 family: {n} rows with a p-value, {nd} rejected, largest rejected p {pmax:.4f} (before the floor). "
      f"Nadeau-Bengio family: {nn} multi-fold rows, {ndn} rejected, largest rejected p {pmaxn:.4f}")
    P(f"  {'rule':66s} tot " + " ".join(f"{s[:5]:>5s}" for s in SUITES))
    for name, fn in RULES.items():
        keep = [r for r in rs if fn(r)]
        P(f"  {name:66s} {len(keep):3d} " + " ".join(f"{sum(r['suite'] == s for r in keep):5d}" for s in SUITES)
          + ("   <- recommended" if name == REC else "   <- alternative" if name == ALT else ""))
    base = {r["label"] for r in rs if RULES[REC](r)}
    for name, fn in RULES.items():
        if name == REC: continue
        keep = {r["label"] for r in rs if fn(r)}
        P(f"  vs recommended, {name[:2].strip()}: gains {', '.join(sorted(keep - base, key=str.lower)) or '-'}. Loses {', '.join(sorted(base - keep, key=str.lower)) or '-'}")
    P(f"  mirror of the recommended rule (AA beats SPO): {sum(r['aa_wins_rec'] for r in rs)} rows: " + lab([r for r in rs if r["aa_wins_rec"]]))
    P(f"  mirror of the alternative rule (AA beats SPO): {sum(r['aa_wins_alt'] for r in rs)} rows: " + lab([r for r in rs if r["aa_wins_alt"]]))
    ff = [r for r in rs if floor_ok(r) and r["bh_fold"] and allpos(r)]
    P(f"  BH over the {BHINFO[prof, 'bh_fold'][0]} multi-fold rows only (seed-only p-values of 1-split sets left out of the family): {len(ff)} rows pass. "
      f"Differences to the recommended rule: {lab([r for r in rs if (r in ff) != r['pass_rec']])}")
    P(f"  sign of the effect over all rows: > 0 in {sum(r['effect'] > 0 for r in rs)}, < 0 in {sum(r['effect'] < 0 for r in rs)}, = 0 in {sum(r['effect'] == 0 for r in rs)}. "
      f"Beyond the floor: SPO {sum(floor_ok(r) for r in rs)}, AA {sum(r['valid'] and -r['effect'] > r['floor'] for r in rs)}")
    P("  floor sweep under the recommended guards (AUC floor / regression floor -> rows passing): " + ", ".join(
        f"{fa}/{fr}: {sum(r['valid'] and r['effect'] > (fr if r['metric'] == 'rel_rmse_red' else fa) and r['bh'] and allpos(r) for r in rs)}"
        for fa, fr in ((0.0, 0.0), (0.0025, 0.005), (0.004, 0.0075), (0.005, 0.01), (0.0075, 0.015), (0.01, 0.02))))
    s1 = [r for r in rs if r["valid"] and r["effect_seed1"] > r["floor"]]
    P(f"  seed-1-only selection (floor on the seed-1 effect, the paper's current rule extended to all suites): {len(s1)} rows. "
      f"Of these, fail the recommended rule: {lab([r for r in s1 if not r['pass_rec']])}")
    # winner's curse: seed-1-selected 5-seed rows, seed-1 effect vs seeds 2-5, in floor units (pools AUC and RMSE)
    for nm, grp in (("pass the recommended rule", [r for r in s1 if r["seeds"] == 5 and r["pass_rec"]]), ("fail it", [r for r in s1 if r["seeds"] == 5 and not r["pass_rec"]])):
        if grp: P(f"    seed-1-selected rows that {nm} (n={len(grp)}): mean effect in floor units, seed 1 {st.mean(r['seed_effects'][0] / r['floor'] for r in grp):.2f}, "
                  f"seeds 2-5 {st.mean(st.mean(r['seed_effects'][1:]) / r['floor'] for r in grp):.2f}")

for prof in ("kdd28_ydf", "kdd28_hao"):
    P(f"\n5. Datasets passing, {prof}. Recommended = {REC[3:]}. '+alt' rows pass only the alternative ({ALT[3:]} with B = floor AND > 2 SE)")
    for suite in SUITES:
        keep = sorted((r for r in by(prof, suite) if r["pass_rec"] or r["pass_alt"]), key=lambda r: (not r["pass_rec"], -r["effect"]))
        P(f" [{suite}] recommended {sum(r['pass_rec'] for r in keep)}, alternative {sum(r['pass_alt'] for r in keep)}, of {len(by(prof, suite))}")
        for r in keep:
            P(f"  {'    ' if r['pass_rec'] else '+alt'} {r['dataset']:38s} effect {r['effect']:+.4f}  SE {u4(r['se_rule'])}{'~' if r['noise'] == 'seeds' else ' '} ({r['effect'] / r['se_rule']:5.1f} SE, p {r['p_rule']:.4f})  "
              f"seeds > floor {r['seeds_over']}/{r['seeds']}, > 0 {r['seeds_pos']}/{r['seeds']}  cells won {r['cells_won']}/{r['cells']}"
              f"{'  [also in ' + r['also_in'] + ']' if r['also_in'] else ''}{'  [in current list]' if r['in_current_list'] else ''}")
    for nm, key in (("recommended", "pass_rec"), ("alternative", "pass_alt")):
        u = sorted({r["dataset"] for r in by(prof) if r[key]}, key=str.lower)
        P(f" unique dataset names passing the {nm} rule: {len(u)} ({sum(r[key] for r in by(prof))} rows): {', '.join(u)}")
        if prof == "kdd28_ydf":
            P(f"   current list (11) vs {nm}: not passing in the SPAA/wide rows: " + (", ".join(sorted((d for d in sel_now if not any(r[key] and r["dataset"] == d and r["suite"] in ("spaa", "wide") for r in Y)), key=str.lower)) or "-"))
    P(" SPAA/CC18 overlap, both evaluations: " + ", ".join(
        f"{d} spaa {a['effect']:+.4f}{'*' if a['pass_rec'] else ''} / cc18 {b['effect']:+.4f}{'*' if b['pass_rec'] else ''}"
        for d in sorted(names[prof]) if len(names[prof][d]) > 1
        for a in by(prof, "spaa") if a["dataset"] == d for b in by(prof, "cc18") if b["dataset"] == d) + "  (* = passes the recommended rule)")

P("\n6. Anomalies")
for prof in ("kdd28_ydf", "kdd28_hao"):
    for r in by(prof):
        if r["dataset"] in ("hiva_agnostic", "sberbank-housing"):
            P(f"  {prof} {r['dataset']}: AA per seed " + " ".join(f"{v:.4f}" for v in r["aa_seed_means"]) + " | SPO per seed "
              + " ".join(f"{v:.4f}" for v in r["spo_seed_means"]) + f" | effect {r['effect']:+.4f}, seed SD {u4(r['seed_sd'])}, effect/SE {r['effect'] / r['se_rule']:.1f}, folds {r['folds']}")
inv = [f"{r['profile']} {r['label']}" for r in rows if not r["valid"]]
P(f"  rows with a mean AUC <= 0.5 in either arm (an AUC edge is not interpretable there): {', '.join(inv) or 'none'}")
low = [f"{r['label']} AA {r['aa_mean']:.3f}" for r in Y if r["metric"] == "auc_edge" and r["valid"] and min(r["aa_mean"], r["spo_mean"]) < 0.56]
P(f"  kdd28_ydf valid AUC rows with an arm below 0.56 (near chance): {', '.join(low) or 'none'}")
for prof in ("kdd28_ydf", "kdd28_hao"):
    rg = by(prof, "regression")
    worst = sorted(rg, key=lambda r: -r["spo_seed_sd"] / r["aa_mean"])[:3]
    P(f"  {prof} regression, SPO seed SD of the mean RMSE relative to the AA RMSE, top 3: " + ", ".join(f"{r['dataset']} {r['spo_seed_sd'] / r['aa_mean']:.3f}" for r in worst)
      + f". Median over the 18: {st.median(r['spo_seed_sd'] / r['aa_mean'] for r in rg):.4f}. SPO/AA RMSE ratio of sberbank-housing: "
      + " ".join(f"{a / b:.2f}" for r in rg if r["dataset"] == "sberbank-housing" for a, b in zip(r["spo_seed_means"], r["aa_seed_means"])))
    one = [r for r in by(prof) if r["folds"] == 1]
    P(f"  {prof} 1-split sets (no fold SE): " + ", ".join(f"{r['label']} {r['effect']:+.4f} ({r['seeds']} seed{'s' if r['seeds'] > 1 else ''})" for r in one))
big = sorted((r for r in Y if r["seed_sd"] is not None), key=lambda r: -r["seed_sd"] / r["floor"])[:6]
P("  kdd28_ydf largest seed SD of the effect, in floor units: " + ", ".join(f"{r['label']} {r['seed_sd'] / r['floor']:.1f}" for r in big))
# epsilon under kdd28_ydf: 1 shipped split, 1 seed. Other evidence: distributed arms, Hao seeds, Hanley-McNeil bound
ep = {r["arm"]: float(r["test_auc"]) for r in read(f"{R}/2026-11-04_wide_eval/distacc/results.csv") if r["dataset"] == "epsilon" and r["rc"] == "0"}
pairs = [(w, ep[f"{w}:SPO"] - ep[f"{w}:AA"]) for w in ("single", "1", "4", "8", "16", "32") if f"{w}:SPO" in ep and f"{w}:AA" in ep]
a, npos = ep["single:AA"], 50000   # 100k test rows, balanced classes assumed (hypothesis, class counts not read)
q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
hm = math.sqrt((a * (1 - a) + (npos - 1) * (q1 - a * a) + (npos - 1) * (q2 - a * a)) / (npos * npos)) * math.sqrt(2)
eh = next(r for r in by("kdd28_hao", "wide") if r["dataset"] == "epsilon")
P(f"  epsilon, kdd28_ydf: edge {pairs[0][1]:+.4f} on 1 split and 1 seed. SPO-AA by worker count (other partitions, same test set): " + ", ".join(f"{w} {e:+.4f}" for w, e in pairs)
  + f". Hanley-McNeil unpaired SE of an AUC difference at {a:.4f} with 50k/50k rows: {hm:.5f} (upper bound for the paired SE), so the edge is {pairs[0][1] / hm:.1f} x that bound. "
  f"kdd28_hao 5 seeds: {eh['effect']:+.4f}, seed SD {eh['seed_sd']:.5f}")

# ---- section 0: recommendation, written from the numbers above
rec, alt = [r for r in Y if r["pass_rec"]], [r for r in Y if r["pass_alt"]]
cnt = lambda rs: ", ".join(f"{s} {sum(r['suite'] == s for r in rs)}" for s in SUITES)
nb_keep = [r for r in Y if RULES["G  floor AND BH q=0.05 on Nadeau-Bengio t (multi-fold only)"](r)]
aa_w = [r for r in Y if r["aa_wins_rec"]]
n, nd, pmax = BHINFO["kdd28_ydf", "bh"]
S = []
S.append("0. Recommendation (kdd28_ydf is the primary profile. Details and evidence in sections 1-6)")
S.append("  Effect: binary AUC sets use the test-AUC edge SPO - AA. Regression sets use the relative RMSE reduction (AA - SPO) / AA per fold.")
S.append("  Both are averaged over folds and over the 5 seeds (YDF wide sets have seed 1 only).")
S.append(f"  RECOMMENDED RULE: 5-seed mean effect > 0.005 AUC (regression: > 1 % relative RMSE reduction), AND the one-sided paired t-test over the")
S.append(f"    seed-averaged fold differences is significant after Benjamini-Hochberg at q = 0.05 over all {n} dataset rows, AND the effect is > 0 in every seed.")
S.append(f"    Passing rows: {len(rec)} of {len(Y)} ({cnt(rec)}), {len({r['dataset'] for r in rec})} unique dataset names. Largest p-value BH accepts: {pmax:.4f}.")
S.append(f"    The same rule mirrored (AA beats SPO) passes {len(aa_w)} rows: {lab(aa_w)}.")
S.append(f"    For scale: the floor alone on 5-seed means passes {sum(floor_ok(r) for r in Y)} rows, the floor on seed 1 alone (the current rule, all suites) {sum(r['valid'] and r['effect_seed1'] > r['floor'] for r in Y)}.")
S.append(f"  ALTERNATIVE (the v1 rule extended): replace the BH test by effect > 2 SE over folds. Passing rows: {len(alt)} ({cnt(alt)}).")
S.append(f"    Gains over the recommended rule: {lab([r for r in alt if not r['pass_rec']])}. Loses: {lab([r for r in rec if not r['pass_alt']])}.")
S.append(f"    Cost: no multiple-comparison control. 2 SE with 5 folds is a one-sided p of {float(stats.t.sf(2, 4)):.3f}, and {len(Y)} rows are tested. Mirrored, it passes {sum(r['aa_wins_alt'] for r in Y)} AA wins.")
S.append(f"    Middle option: BH at q = 0.10 passes {sum(RULES[chr(70) + chr(39) + ' as F with BH q=0.10'](r) for r in Y)} rows (adds {lab([r for r in Y if RULES[chr(70) + chr(39) + ' as F with BH q=0.10'](r) and not r['pass_rec']])}).")
S.append(f"  STRICT VARIANT (not recommended, for reference): BH on the Nadeau-Bengio corrected t keeps {len(nb_keep)} rows ({cnt(nb_keep)}).")
S.append(f"    It loses {lab([r for r in rec if r not in nb_keep])}.")
S.append("    The plain k-fold paired t is known to be liberal because training folds overlap (Dietterich 1998, Nadeau and Bengio 2003). Its size on these data is not measured here.")
S.append(f"  Why 1 % for regression: 0.005 AUC is {ra:.1f} x the median seed SD of the edge and {rr:.1f} x the median fold SE on the {len(auc5)} five-seed AUC rows.")
S.append(f"    The same multiples of the regression medians are {m_sd:.4f} and {m_se:.4f}. A reseed of SPO alone moves the fold-mean AUC by more than 0.005 in {1 - qa:.1%} of seed pairs,")
S.append(f"    and moves the relative RMSE by more than 1 % in {1 - qreg:.1%} of seed pairs. Floors from 0.75 % to 1.5 % select the same regression sets.")
S.append("  Rows the rule cannot test or that need a footnote:")
S.append(f"    epsilon (kdd28_ydf): 1 shipped split and 1 seed, so no p-value. Edge {pairs[0][1]:+.4f} = {pairs[0][1] / hm:.1f} x the Hanley-McNeil unpaired SE bound. Needs seeds 2-5 to enter by the rule.")
S.append("    philippine (kdd28_ydf): passes on seed 1 with 5 folds. Seed agreement is untested under kdd28_ydf (5/5 seeds above the floor under kdd28_hao).")
ov = {d: (next(r for r in by("kdd28_ydf", "spaa") if r["dataset"] == d), next(r for r in by("kdd28_ydf", "cc18") if r["dataset"] == d)) for d in overlap}
split = [d for d, (a_, b_) in ov.items() if a_["pass_rec"] != b_["pass_rec"]]
S.append("    Datasets evaluated twice (SPAA eval 5 folds and CC18 10 folds) that pass in only one: " + ", ".join(
    f"{d} (spaa {ov[d][0]['effect']:+.4f}, p {ov[d][0]['p_rule']:.3f}. cc18 {ov[d][1]['effect']:+.4f}, p {ov[d][1]['p_rule']:.3f})" for d in split) + ".")
S.append("    Bioresponse is the weakest pass: its SPAA-eval edge is below the floor and 3/5 CC18 seeds are above it. Suggest dropping it or showing the CC18 number only.")
S.append("  Anomalies: hiva_agnostic is excluded (AUC < 0.5 in both arms, and its edge is 1.0 SE). sberbank-housing stays in the universe as an AA win (SPO RMSE 3.1 to 4.3 x AA).")
S.append("  kdd28_hao (secondary): " + f"{sum(r['pass_rec'] for r in by('kdd28_hao'))} rows pass the recommended rule ({cnt([r for r in by('kdd28_hao') if r['pass_rec']])}), "
         f"{sum(r['pass_alt'] for r in by('kdd28_hao'))} the alternative. Lists in section 5.")
S.append("")
out = S + out

open(os.path.join(H, "spo_clearly_beats_aa_report.txt"), "w").write("\n".join(out) + "\n")
cols = ["profile", "suite", "metric", "dataset", "folds", "seeds", "aa_mean", "spo_mean", "effect", "effect_seed1", "floor", "valid", "noise", "se_rule",
        "fold_sd", "se", "se_nb", "t", "p", "t_nb", "p_nb", "seed_sd", "seed_se", "t_seed", "p_seed", "p_rule", "seed_min", "seed_max", "seeds_over", "seeds_pos",
        "seeds_under", "cells_won", "cells", "aa_seed_sd", "spo_seed_sd", "bh", "bh10", "bh_nb", "pass_rec", "pass_alt", "aa_wins_rec", "aa_wins_alt", "also_in", "in_current_list"]
with open(os.path.join(H, "spo_clearly_beats_aa_per_dataset.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", restval=""); w.writeheader()
    for r in rows: w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else ("" if v is None else v)) for k, v in r.items()})
print("\n".join(out))
