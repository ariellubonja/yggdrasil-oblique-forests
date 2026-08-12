#!/usr/bin/env python3
"""Statistical comparison of split finders from accuracy.sh CV result CSVs.

Purpose: support the paper claim that Random Histogramming (scalar or
vectorized -- bit-identical implementations) has no meaningful accuracy
penalty vs Exact split finding, for SPO-RF and SPO-GBT.

Input: any number of accuracy_*.csv files produced by accuracy.sh /
parse_log_to_csv.py (main = held-out accuracy; side files *_auc.csv /
*_logloss.csv are auto-typed by filename). Provenance headers are skipped;
the model seed is recovered from the EXTRA_TRAIN_ARGS provenance line
(--seed=N, default 1), so multi-seed replicates live in separate CSVs.

Method (the unit of analysis is the DATASET, folds within a dataset are
correlated and never treated as independent across datasets):
  1. Per dataset: paired per-(seed,fold) deltas comparator-vs-Exact, mean
     delta, 95% t CI, two-sided Wilcoxon signed-rank (Holm-corrected across
     datasets).
  2. Across datasets: per-dataset mean deltas D_1..D_n ->
     mean/median, 95% t CI, 95% percentile bootstrap CI, Wilcoxon
     signed-rank, paired t, win/tie/loss, and TOST equivalence at a margin
     grid (reports the smallest margin at which p_TOST < 0.05).
  3. Friedman test + average ranks when >= 3 arms are present.
  4. Seed yardstick (needs >= 2 seeds of the Exact arm): compares the
     method effect |Random - Exact| at a fixed seed against the spread of
     Exact across seeds -- "is switching the split finder distinguishable
     from switching the RNG seed?".

Usage: accuracy_stats.py --out_dir DIR csv [csv ...]
Writes: report.md, tidy.csv, per-family LaTeX tables into DIR.
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

BOOT_N = 20000
BOOT_SEED = 20260811
# Equivalence margins tried in ascending order (absolute units of the
# metric; 0.005 accuracy = half a percentage point).
TOST_MARGINS = [0.001, 0.0025, 0.005, 0.01, 0.02]
# |delta| below this counts as a practical tie in win/tie/loss.
PRACTICAL_TIE = 0.001

SEED_RE = re.compile(r'--seed[ =](\d+)')
BINS_RE = re.compile(r'--histogram_num_bins[ =](\d+)')
# The canonical bin count: arms at this count keep their plain method name;
# other counts get a _bins<N> suffix (the bin-count ablation).
CANONICAL_BINS = 64


def metric_of(path):
    base = os.path.basename(path)
    if base.endswith('_auc.csv'):
        return 'auc'
    if base.endswith('_logloss.csv'):
        return 'logloss'
    return 'accuracy'


def load_csv(path):
    """Returns (tidy DataFrame, seed) for one result CSV."""
    seed = 1
    bins = None
    header_idx = None
    with open(path) as f:
        for i, line in enumerate(f):
            m = SEED_RE.search(line)
            if m and header_idx is None:
                seed = int(m.group(1))
            m = BINS_RE.search(line)
            if m and header_idx is None:
                bins = int(m.group(1))
            if line.startswith('dataset,'):
                header_idx = i
                break
    if header_idx is None:
        raise ValueError(f"{path}: no 'dataset,' header line found")
    df = pd.read_csv(path, skiprows=header_idx)
    fold_cols = [c for c in df.columns if c.startswith('fold_')]
    long = df.melt(id_vars=['dataset', 'algorithm'], value_vars=fold_cols,
                   var_name='fold', value_name='value')
    long['fold'] = long['fold'].str.replace('fold_', '').astype(int)
    long = long.dropna(subset=['value'])
    long['value'] = long['value'].astype(float)
    long['seed'] = seed
    long['metric'] = metric_of(path)
    long['source'] = os.path.basename(path)
    # Non-canonical bin counts (the ablation) get their own method label so
    # they don't merge with the canonical 64-bin arm of the same name.
    if bins is not None and bins != CANONICAL_BINS:
        long['algorithm'] = long['algorithm'] + f'_bins{bins}'
    return long


def family_of(algo):
    if algo.startswith('SPO-GBT'):
        return 'SPO-GBT'
    if algo.startswith('SPORF'):
        return 'SPO-RF'
    return algo.split('_')[0]


def method_of(algo):
    m = re.sub(r'^(SPO-GBT|SPORF)_', '', algo)
    return m


def paired_frame(sub, base_method, comp_method):
    """Aligns comparator vs baseline on (dataset, seed, fold). Returns df
    with columns dataset, seed, fold, base, comp, delta (comp - base)."""
    b = sub[sub.method == base_method][['dataset', 'seed', 'fold', 'value']]
    c = sub[sub.method == comp_method][['dataset', 'seed', 'fold', 'value']]
    m = b.merge(c, on=['dataset', 'seed', 'fold'], suffixes=('_base', '_comp'))
    m['delta'] = m['value_comp'] - m['value_base']
    return m


def t_ci(x, conf=0.95):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return (np.nan, np.nan)
    se = x.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.5 + conf / 2, n - 1)
    return (x.mean() - tcrit * se, x.mean() + tcrit * se)


def boot_ci(x, conf=0.95):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(BOOT_SEED)
    means = rng.choice(x, size=(BOOT_N, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * (0.5 - conf / 2),
                                   100 * (0.5 + conf / 2)])
    return (lo, hi)


def wilcoxon_p(deltas):
    d = np.asarray(deltas, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 1.0
    try:
        return stats.wilcoxon(d, alternative='two-sided').pvalue
    except ValueError:
        return 1.0


def tost_p(deltas, margin):
    """Two one-sided t-tests for |mean delta| < margin. Returns max p."""
    d = np.asarray(deltas, dtype=float)
    n = len(d)
    if n < 3:
        return np.nan
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return 0.0 if abs(d.mean()) < margin else 1.0
    df = n - 1
    p_lower = 1 - stats.t.cdf((d.mean() + margin) / se, df)
    p_upper = stats.t.cdf((d.mean() - margin) / se, df)
    return max(p_lower, p_upper)


def noninferiority_p(deltas, margin):
    """One-sided t-test of H0: mean delta <= -margin vs H1: mean > -margin.
    This is the paper's actual claim ('no meaningful penalty'): a metric
    IMPROVEMENT never counts against equivalence. Returns the p-value."""
    d = np.asarray(deltas, dtype=float)
    n = len(d)
    if n < 3:
        return np.nan
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return 0.0 if d.mean() > -margin else 1.0
    return 1 - stats.t.cdf((d.mean() + margin) / se, n - 1)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def fmt(x, digits=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return 'n/a'
    return f"{x:.{digits}f}"


def fmt_p(p):
    if not np.isfinite(p):
        return 'n/a'
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def per_dataset_table(sub, base_method, comp_methods):
    """Rows: one per dataset with means per arm and paired delta stats for
    each comparator."""
    rows = []
    for ds, g in sub.groupby('dataset'):
        row = {'dataset': ds}
        gb = g[g.method == base_method]
        row[f'{base_method}_mean'] = gb.value.mean()
        row[f'{base_method}_std'] = gb.value.std(ddof=1)
        row['n_samples'] = len(gb)
        for cm in comp_methods:
            m = paired_frame(g, base_method, cm)
            if m.empty:
                continue
            row[f'{cm}_mean'] = m.value_comp.mean()
            row[f'{cm}_std'] = m.value_comp.std(ddof=1)
            lo, hi = t_ci(m.delta)
            row[f'{cm}_delta'] = m.delta.mean()
            row[f'{cm}_delta_lo'] = lo
            row[f'{cm}_delta_hi'] = hi
            row[f'{cm}_p'] = wilcoxon_p(m.delta)
        rows.append(row)
    df = pd.DataFrame(rows)
    for cm in comp_methods:
        pcol = f'{cm}_p'
        if pcol in df.columns:
            mask = df[pcol].notna()
            df.loc[mask, f'{cm}_p_holm'] = holm(df.loc[mask, pcol].values)
    return df


def aggregate_stats(ds_table, cm, higher_is_better=True):
    """Across-dataset stats on per-dataset mean deltas for comparator cm.
    Deltas are always comp - base; for lower-is-better metrics (logloss)
    the non-inferiority test is run on the negated deltas so that 'worse'
    is always the guarded direction."""
    col = f'{cm}_delta'
    if col not in ds_table.columns:
        return None
    d = ds_table[col].dropna().values
    if len(d) == 0:
        return None
    d_noninf = d if higher_is_better else -d
    out = {'n_datasets': len(d),
           'mean': d.mean(), 'median': np.median(d),
           'min': d.min(), 'max': d.max(),
           't_ci': t_ci(d), 'boot_ci': boot_ci(d),
           'wilcoxon_p': wilcoxon_p(d),
           't_p': stats.ttest_1samp(d, 0).pvalue if len(d) > 2 else np.nan,
           'win': int((d > PRACTICAL_TIE).sum()),
           'tie': int((np.abs(d) <= PRACTICAL_TIE).sum()),
           'loss': int((d < -PRACTICAL_TIE).sum()),
           'win_raw': int((d > 0).sum()),
           'loss_raw': int((d < 0).sum()),
           'tost': {m: tost_p(d, m) for m in TOST_MARGINS},
           'noninf': {m: noninferiority_p(d_noninf, m) for m in TOST_MARGINS}}
    out['tost_margin'] = next((m for m in TOST_MARGINS
                               if np.isfinite(out['tost'][m])
                               and out['tost'][m] < 0.05), None)
    out['noninf_margin'] = next((m for m in TOST_MARGINS
                                 if np.isfinite(out['noninf'][m])
                                 and out['noninf'][m] < 0.05), None)
    # Leave-one-out sensitivity: drop the single largest-|delta| dataset and
    # recompute the headline numbers, so no aggregate claim hinges on one
    # outlier (e.g. a dataset outside the harness' no-missing-values
    # envelope where the baseline itself degenerates).
    if len(d) > 3:
        worst = np.argmax(np.abs(d))
        d_loo = np.delete(d, worst)
        d_loo_ni = d_loo if higher_is_better else -d_loo
        names = ds_table[col].dropna()
        out['loo_dropped'] = str(ds_table.loc[names.index[worst], 'dataset'])
        out['loo_mean'] = d_loo.mean()
        out['loo_t_ci'] = t_ci(d_loo)
        out['loo_tost_margin'] = next(
            (m for m in TOST_MARGINS if tost_p(d_loo, m) < 0.05), None)
        out['loo_noninf_margin'] = next(
            (m for m in TOST_MARGINS
             if noninferiority_p(d_loo_ni, m) < 0.05), None)
    return out


def friedman_ranks(sub, methods):
    """Friedman test + average ranks over per-dataset CV means (higher =
    better rank 1). Uses only datasets with all arms present."""
    means = (sub.groupby(['dataset', 'method']).value.mean()
             .unstack('method'))
    means = means.dropna(subset=[m for m in methods if m in means.columns])
    methods = [m for m in methods if m in means.columns]
    if len(methods) < 3 or len(means) < 3:
        return None
    mat = means[methods].values
    stat, p = stats.friedmanchisquare(*[mat[:, j]
                                        for j in range(mat.shape[1])])
    # rank 1 = best (ties share average rank). Lower-is-better metrics are
    # pre-negated by the caller.
    ranks = np.vstack([stats.rankdata(-mat[i]) for i in range(len(mat))])
    return {'n_datasets': len(means), 'chi2': stat, 'p': p,
            'avg_ranks': dict(zip(methods, ranks.mean(axis=0)))}


def seed_yardstick(sub, base_method, comp_method):
    """Method effect vs seed effect. Needs >= 2 seeds on the base arm.
    seed_spread(ds) = std over seeds of the CV-mean of the base arm.
    method_gap(ds)  = |mean over seeds of (comp - base) CV-mean gap|."""
    b = sub[sub.method == base_method]
    if b.seed.nunique() < 2:
        return None
    base_seed_means = b.groupby(['dataset', 'seed']).value.mean()
    spread = base_seed_means.groupby('dataset').std(ddof=1)
    m = paired_frame(sub, base_method, comp_method)
    gap = m.groupby('dataset').delta.mean().abs()
    j = pd.concat([spread.rename('seed_spread'), gap.rename('method_gap')],
                  axis=1).dropna()
    if j.empty:
        return None
    with np.errstate(divide='ignore'):
        ratio = j.method_gap / j.seed_spread.replace(0, np.nan)
    return {'n_datasets': len(j),
            'median_seed_spread': j.seed_spread.median(),
            'median_method_gap': j.method_gap.median(),
            'median_ratio': ratio.median(),
            'frac_gap_within_1_spread':
                float((j.method_gap <= j.seed_spread).mean())}


def latex_table(ds_table, base_method, comp_methods, family, metric, path):
    """Per-dataset LaTeX table: baseline mean+-std, each comparator
    mean+-std and paired delta with 95% CI."""
    lines = [
        r'% Auto-generated by accuracy_stats.py -- do not edit by hand.',
        r'\begin{tabular}{l' + 'c' * (1 + 2 * len(comp_methods)) + '}',
        r'\toprule',
    ]
    heads = ['Dataset', 'Exact']
    for cm in comp_methods:
        heads += [cm.replace('_', ' '), r'$\Delta$ [95\% CI]']
    lines.append(' & '.join(heads) + r' \\')
    lines.append(r'\midrule')
    for _, r in ds_table.sort_values('dataset').iterrows():
        name = r.dataset.replace('task_', '').split('_', 1)[-1]
        name = name.replace('_', r'\_')
        cells = [name,
                 f"{r[f'{base_method}_mean']:.4f}"
                 rf" $\pm$ {r[f'{base_method}_std']:.4f}"]
        for cm in comp_methods:
            if f'{cm}_mean' not in r or pd.isna(r.get(f'{cm}_mean')):
                cells += ['--', '--']
                continue
            cells.append(f"{r[f'{cm}_mean']:.4f}"
                         rf" $\pm$ {r[f'{cm}_std']:.4f}")
            cells.append(f"{r[f'{cm}_delta']:+.4f}"
                         f" [{r[f'{cm}_delta_lo']:+.4f},"
                         f" {r[f'{cm}_delta_hi']:+.4f}]")
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('csvs', nargs='+')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tidy = pd.concat([load_csv(p) for p in args.csvs], ignore_index=True)
    # Bit-identical replicates (e.g. the scalar-build verification sweep)
    # would double-count samples of the same arm; keep the first source.
    tidy = tidy.drop_duplicates(
        subset=['algorithm', 'metric', 'dataset', 'seed', 'fold'],
        keep='first')
    tidy['family'] = tidy.algorithm.map(family_of)
    tidy['method'] = tidy.algorithm.map(method_of)
    tidy.to_csv(os.path.join(args.out_dir, 'tidy.csv'), index=False)

    rep = []
    rep.append('# Split-finder accuracy comparison: Random Histogram vs Exact')
    rep.append('')
    rep.append(f'Inputs: {len(args.csvs)} CSVs, '
               f'{tidy.dataset.nunique()} datasets, '
               f'seeds: {sorted(tidy.seed.unique().tolist())}. '
               f'Unit of analysis for all aggregate claims: the dataset '
               f'(per-dataset mean of paired per-(seed,fold) deltas).')
    rep.append('')

    for family in sorted(tidy.family.unique()):
        for metric in ['accuracy', 'auc', 'logloss']:
            sub = tidy[(tidy.family == family) & (tidy.metric == metric)]
            if sub.empty:
                continue
            methods = sorted(sub.method.unique())
            base = next((m for m in methods if m == 'Exact'), None)
            if base is None:
                continue
            comps = [m for m in methods if m != base]
            if not comps:
                continue
            higher_is_better = metric != 'logloss'

            rep.append(f'## {family} -- {metric}')
            rep.append('')
            ds_table = per_dataset_table(sub, base, comps)
            ds_table.to_csv(os.path.join(
                args.out_dir,
                f'per_dataset_{family}_{metric}.csv'.replace('/', '_')),
                index=False)
            latex_table(ds_table, base, comps, family, metric,
                        os.path.join(args.out_dir,
                                     f'table_{family}_{metric}.tex'
                                     .replace('/', '_')))

            for cm in comps:
                agg = aggregate_stats(ds_table, cm, higher_is_better)
                if agg is None:
                    continue
                sign_note = ('(positive delta = comparator better)'
                             if higher_is_better else
                             '(negative delta = comparator better)')
                lo_t, hi_t = agg['t_ci']
                lo_b, hi_b = agg['boot_ci']
                rep.append(f'### {cm} vs {base} {sign_note}')
                rep.append('')
                rep.append(f'- Datasets: {agg["n_datasets"]}')
                rep.append(f'- Mean per-dataset delta: {agg["mean"]:+.5f} '
                           f'(median {agg["median"]:+.5f}, range '
                           f'[{agg["min"]:+.5f}, {agg["max"]:+.5f}])')
                rep.append(f'- 95% CI (t): [{lo_t:+.5f}, {hi_t:+.5f}]; '
                           f'95% CI (bootstrap): [{lo_b:+.5f}, {hi_b:+.5f}]')
                rep.append(f'- Wilcoxon signed-rank p = '
                           f'{fmt_p(agg["wilcoxon_p"])}; '
                           f'paired t p = {fmt_p(agg["t_p"])}')
                rep.append(f'- Win/tie/loss (|delta| <= {PRACTICAL_TIE} is '
                           f'a tie): {agg["win"]}/{agg["tie"]}/{agg["loss"]} '
                           f'(raw sign: {agg["win_raw"]} up, '
                           f'{agg["loss_raw"]} down)')
                tost_txt = ', '.join(f'{m}: {fmt_p(p)}'
                                     for m, p in agg['tost'].items())
                rep.append(f'- TOST equivalence p at margins {{{tost_txt}}}')
                if agg['tost_margin'] is not None:
                    rep.append(f'- **Equivalent (TOST) within +-'
                               f'{agg["tost_margin"]} at alpha=0.05**')
                else:
                    rep.append('- **Not TOST-equivalent at any tested '
                               'margin**')
                noninf_txt = ', '.join(f'{m}: {fmt_p(p)}'
                                       for m, p in agg['noninf'].items())
                rep.append(f'- Non-inferiority p at margins {{{noninf_txt}}}')
                if agg['noninf_margin'] is not None:
                    rep.append(f'- **Non-inferior (no penalty worse than '
                               f'{agg["noninf_margin"]}) at alpha=0.05**')
                else:
                    rep.append('- **Non-inferiority not shown at any '
                               'tested margin**')
                if 'loo_dropped' in agg:
                    llo, lhi = agg['loo_t_ci']
                    rep.append(f'- Leave-one-out (drop {agg["loo_dropped"]}):'
                               f' mean {agg["loo_mean"]:+.5f}, 95% CI '
                               f'[{llo:+.5f}, {lhi:+.5f}], TOST margin '
                               f'{agg["loo_tost_margin"]}, non-inf margin '
                               f'{agg["loo_noninf_margin"]}')
                # Per-dataset Holm-significant differences.
                pcol = f'{cm}_p_holm'
                if pcol in ds_table.columns:
                    sig = ds_table[ds_table[pcol] < 0.05]
                    if sig.empty:
                        rep.append('- Per-dataset Wilcoxon (Holm-corrected '
                                   'across datasets): no dataset with '
                                   'p < 0.05')
                    else:
                        det = '; '.join(
                            f'{r.dataset} (delta {r[f"{cm}_delta"]:+.4f}, '
                            f'p={fmt_p(r[pcol])})'
                            for _, r in sig.iterrows())
                        rep.append(f'- Per-dataset Wilcoxon (Holm): '
                                   f'{len(sig)} dataset(s) significant: '
                                   f'{det}')
                rep.append('')

                ys = seed_yardstick(sub, base, cm)
                if ys:
                    rep.append(f'- Seed yardstick ({ys["n_datasets"]} '
                               f'datasets): median |method gap| '
                               f'{ys["median_method_gap"]:.5f} vs median '
                               f'Exact seed spread '
                               f'{ys["median_seed_spread"]:.5f} '
                               f'(ratio {ys["median_ratio"]:.2f}); method '
                               f'gap within 1 seed-spread on '
                               f'{100 * ys["frac_gap_within_1_spread"]:.0f}%'
                               f' of datasets')
                    rep.append('')

            fr_sub = sub.copy()
            if not higher_is_better:
                fr_sub = fr_sub.assign(value=-fr_sub.value)
            fr = friedman_ranks(fr_sub, methods)
            if fr:
                ranks = ', '.join(f'{m}: {r:.2f}'
                                  for m, r in fr['avg_ranks'].items())
                rep.append(f'- Friedman over {fr["n_datasets"]} datasets: '
                           f'chi2 = {fr["chi2"]:.2f}, p = {fmt_p(fr["p"])}; '
                           f'average ranks (1 = best): {ranks}')
                rep.append('')

    report_path = os.path.join(args.out_dir, 'report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(rep) + '\n')
    print(f'Wrote {report_path}')


if __name__ == '__main__':
    sys.exit(main())
