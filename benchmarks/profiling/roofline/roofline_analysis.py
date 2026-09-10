#!/usr/bin/env python3
"""Roofline figures + kernel time shares from Intel Advisor results (survey + trip counts/FLOP).

Per arm it needs three CSV exports of one Advisor project:
  <p>.topdown.csv    advisor --report=top-down --format=csv --show-all-columns      (call tree, times only)
  <p>.functions.csv  advisor --report=survey --show-functions --format=csv --show-all-columns
                     (every node: loops, functions, inlined functions; self FLOP / bytes / elapsed)
  <p>.roofs.csv      advisor --report=roofs --format=csv

Why the call tree: Advisor books samples on the innermost node, and that node is often an inlined callee
(std::isnan, vector::operator[], AttributeValue ...) rather than the loop.  A loop-only self-time sum therefore
under-counts kernels whose body is mostly inlined calls (May-2025 Evaluate: 87% of its time sits on the
inlined std::isnan).  Here every node's self time is charged to the nearest enclosing kernel function, which is
exactly the chrono-scope definition (Evaluate = ApplyProjection, the two split finders, SampleProjection), and
shares are over the tree-training subtree (ThreadPool::ThreadLoop), i.e. the training block only.

The ApplyProjection dot is the whole Evaluate subtree (sum of self FLOP / bytes of its nodes, elapsed time of the
function); other dots are Advisor's per-loop self metrics (what its own roofline chart shows).

usage: roofline_analysis.py --out X.png --level L1|DRAM --ops mixed|float --roofs R.csv [--title T]
                            "label=<project prefix>" ...   (prefix = path without .topdown.csv)
Writes <out>.<label>.top.csv (dots) and appends a row per arm to --summary CSV if given.
"""
import argparse, os, re, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

CATS = ['ApplyProjection', 'Split search (histogram)', 'Split search (sort/scan)', 'Projection sampling', 'Other']
CAT_COL = {'ApplyProjection': '#2a78d6', 'Split search (histogram)': '#eb6834', 'Split search (sort/scan)': '#1baf7a',
           'Projection sampling': '#eda100', 'Other': '#8a8a8a'}
# kernel functions: nearest enclosing one wins
KERNELS = [('ApplyProjection', 'ProjectionEvaluator::Evaluate', None),
           # May-2025 binary built before Evaluate was marked noinline: the inlined row loop shows up as a loop of the
           # driver at 'oblique.cc' with no line number (its inner item loop is at oblique.cc:1044, inside Evaluate)
           ('ApplyProjection', '[loop in yggdrasil_decision_forests::model::decision_tree::FindBestConditionSparseObliqueTemplate', r'^oblique\.cc$'),
           ('Split search (histogram)', 'FindSplitLabelClassificationFeatureNumericalHistogram', None),
           ('Split search (sort/scan)', 'FindSplitLabelClassificationFeatureNumericalCart', None),
           ('Projection sampling', 'internal::SampleProjection', None)]


def kernel_cat(node, srcloc):
    for c, k, locre in KERNELS:
        if k in node and (locre is None or re.match(locre, srcloc)): return c
    return None
# leaf functions whose parent frame Advisor sometimes loses (stack walk); charged by name
LEAF_FALLBACK = [('Projection sampling', r'mersenne_twister_engine|uniform_int_distribution|binomial_distribution|bernoulli_distribution|generate_canonical|uniform_real_distribution'),
                 ('Split search (sort/scan)', r'__introsort_loop|ExampleBucket|hwy::N_|vqsort'),
                 ('Split search (histogram)', r'GenHistogramBins|Histogram')]
STRIP = 'yggdrasil_decision_forests::model::decision_tree::'


def num(x):
    try: return float(str(x).replace('s', '').replace('<', '').strip())
    except Exception: return np.nan


def short(name):
    return re.sub(r'<[^<>]*(<[^<>]*>[^<>]*)*>', '<>', str(name).replace(STRIP, '').replace('yggdrasil_decision_forests::', ''))


def read_adv(path):
    df = pd.read_csv(path, skiprows=5)
    df['node'] = df['Function Call Sites and Loops'].astype(str).str.strip()
    df['srcloc'] = df['Source Location'].astype(str)
    for c in df.columns:
        if re.search(r'(Time|GFLOP|GINTOP|Giga OP|GB$|AI$|Intensity|OPS$)', c):
            df[c] = df[c].map(num)   # '46.951s' -> 46.951 (pandas 3 reads these as str dtype, not object)
    return df


def build_tree(topdown_csv):
    """Pre-order listing without depth markers -> parents from Total = Self + sum(children Total)."""
    df = read_adv(topdown_csv)
    df['self'] = df['Self Time'].fillna(0); df['total'] = df['Total Time'].fillna(0)
    parent, stack = [-1] * len(df), []
    for i, r in df.iterrows():
        while stack and stack[-1][1] <= 0.011: stack.pop()
        if stack:
            parent[i] = stack[-1][0]; stack[-1][1] -= r.total
        stack.append([i, r.total - r.self])
    df['parent'] = parent
    kids = df.groupby('parent').total.sum()
    resid = (df.total - df.self - df.index.map(lambda i: kids.get(i, 0.0))).abs().max()
    assert resid < 0.05, f'tree reconstruction failed for {topdown_csv}: residual {resid}'
    cat, kern = ['Other'] * len(df), [-1] * len(df)
    for i in range(len(df)):
        j = i
        while j >= 0:
            hit = kernel_cat(df.at[j, 'node'], df.at[j, 'srcloc'])
            if hit: cat[i], kern[i] = hit, j; break
            j = parent[j]
        if kern[i] < 0:
            for c, pat in LEAF_FALLBACK:
                if re.search(pat, df.at[i, 'node']): cat[i] = c; break
    df['cat'], df['kernel_node'] = cat, kern
    tl = df[df.node.str.contains('ThreadPool::ThreadLoop') & (df.Type == 'Function')].index
    in_train = [False] * len(df)
    if len(tl):
        tl = tl[0]
        for i in range(len(df)):
            j = i
            while j >= 0 and j != tl: j = parent[j]
            in_train[i] = (j == tl)
    df['in_train'] = in_train
    return df


def load_arm(prefix, roofs_csv=None):
    tree = build_tree(prefix + '.topdown.csv')
    fn = read_adv(prefix + '.functions.csv')
    fn = fn[~fn.node.str.startswith('[child]')]
    rp = roofs_csv or prefix + '.roofs.csv'
    roofs = {}
    if os.path.exists(rp):
        r = pd.read_csv(rp, skiprows=1); roofs = {row['Name']: float(row['Bandwidth']) / 1e9 for _, row in r.iterrows()}
    tr = tree[tree.in_train]; tot = tr.self.sum()
    shares = (tr.groupby('cat').self.sum() / tot * 100).to_dict()
    # --- ApplyProjection kernel = Evaluate subtree.  Which flat (name, loc) rows belong to it: those whose tree
    # self time is mostly inside the subtree.
    key = lambda d: d['node'].map(short) + '|' + d['srcloc']
    tree['key'] = key(tree); fn['key'] = key(fn)
    ap_nodes = tree[tree.cat == 'ApplyProjection']
    frac = (ap_nodes.groupby('key').self.sum() / tree.groupby('key').self.sum()).fillna(0)
    in_ap = fn.key.map(lambda k: frac.get(k, 0) >= 0.5)
    ap_rows = fn[in_ap]
    # kernel root rows: the Evaluate function (noinline builds) or the inlined row loop (old May-2025 binary)
    ev = fn[[kernel_cat(n, l) == 'ApplyProjection' and (t == 'Function' or 'Evaluate' not in n) for n, l, t in zip(fn.node, fn.srcloc, fn.Type)]]
    ap = dict(cpu_s=ap_nodes.self.sum(), elapsed_s=float(ev['Total Elapsed Time'].sum()) if len(ev) else np.nan,
              gflop=ap_rows['Self GFLOP'].sum(), gop=ap_rows['Self Giga OP'].sum(), l1_gb=ap_rows['Self Memory GB'].sum())
    for lvl, col in (('dram', 'Self DRAM GB'), ('l2', 'Self L2 GB'), ('l3', 'Self L3 GB')):
        ap[lvl + '_gb'] = ap_rows[col].sum() if col in ap_rows.columns else np.nan
    if 'Total GFLOP' in fn.columns and len(ev):   # results collected with --stacks carry inclusive metrics: cross-check
        ap['check_total_gflop'] = float(ev['Total GFLOP'].sum()); ap['check_total_l1_gb'] = float(ev['Total Memory GB'].sum())
    # loops for the dots (Advisor self metrics), category via the tree when the node is found there
    loops = fn[fn.node.str.startswith('[loop in')].copy()
    tcat = tree.groupby('key').apply(lambda g: g.loc[g.self.idxmax(), 'cat'] if len(g) else 'Other')
    loops['cat'] = loops.key.map(lambda k: tcat.get(k, 'Other'))
    return dict(tree=tree, fn=fn, loops=loops, roofs=roofs, shares=shares, train_cpu_s=tot, ap=ap)


def panel(ax, arm, ops, level, title, roofs_override=None):
    roofs = roofs_override or arm['roofs']
    opcol = 'Self GFLOP' if ops == 'float' else 'Self Giga OP'
    lvl = {'L1': 'Self Memory GB', 'L2': 'Self L2 GB', 'L3': 'Self L3 GB', 'DRAM': 'Self DRAM GB'}[level]
    d = arm['loops'].dropna(subset=[opcol, lvl, 'Self Elapsed Time']).copy()
    d = d[(d[opcol] > 0) & (d[lvl] > 0) & (d['Self Elapsed Time'] > 0)]
    d['ai'] = d[opcol] / d[lvl]; d['perf'] = d[opcol] / d['Self Elapsed Time']
    tmax = d['Self Time'].max()
    d = d[d['Self Time'] >= 0.01 * tmax].sort_values('Self Time', ascending=False)
    comp = [('SP Vector FMA Peak', roofs.get('SP Vector FMA Peak')), ('Scalar Add Peak', roofs.get('Scalar Add Peak'))]
    bw = [(f'{level} BW', roofs.get(f'{level} Bandwidth'))] + ([('L1 BW', roofs.get('L1 Bandwidth'))] if level != 'L1' else [])
    xs = np.logspace(-3, 5, 300); top = max(v for _, v in comp if v)
    for name, v in comp:
        ax.axhline(v, color='#bbb', lw=1); ax.text(8e4 if level == 'DRAM' else 80, v * 1.15, f'{name} {v:,.0f}', ha='right', fontsize=7, color='#666')
    for name, v in bw:
        if not v: continue
        y = v * xs; m = y < top * 1.5; ax.plot(xs[m], y[m], color='#bbb', lw=1)
        xi = xs[m][len(xs[m]) // 3]; ax.text(xi, v * xi * 1.3, f'{name} {v:,.0f} GB/s', rotation=40, fontsize=7, color='#666', rotation_mode='anchor')
    sh = arm['shares']
    for cat in CATS:
        sub = d[d.cat == cat]
        lab = f'{cat} ({sh.get(cat, 0):.0f}% of training CPU time)'
        if sub.empty: ax.scatter([], [], color=CAT_COL[cat], label=lab); continue
        ax.scatter(sub.ai, sub.perf, s=20 + 400 * sub['Self Time'] / tmax, color=CAT_COL[cat], alpha=.75, edgecolor='white', lw=1, label=lab)
    # the ApplyProjection kernel (whole Evaluate subtree)
    ap = arm['ap']; gb = ap['l1_gb'] if level == 'L1' else ap[level.lower() + '_gb']
    o = ap['gflop'] if ops == 'float' else ap['gop']
    if gb and gb > 0 and o > 0 and ap['elapsed_s'] > 0:
        ax.scatter([o / gb], [o / ap['elapsed_s']], marker='*', s=260, color=CAT_COL['ApplyProjection'], edgecolor='black', lw=1, zorder=5,
                   label=f'ApplyProjection kernel (Evaluate incl. inlined callees): {o / ap["elapsed_s"]:.1f} G{"FLOP" if ops == "float" else "OP"}/s, AI {o / gb:.3f}')
    for _, row in d.head(4).iterrows():
        fnm = re.sub(r'^\[loop in ', '', short(row['node'])).rstrip(']')
        fpart, loc = fnm.rsplit(' at ', 1) if ' at ' in fnm else (fnm, '')
        f = re.sub(r'<.*', '', fpart).rstrip(':').split('::')[-1]
        f = {'Evaluate': 'AP', 'FindSplitLabelClassificationFeatureNumericalHistogram': 'Hist', 'FindSplitLabelClassificationFeatureNumericalCart': 'Cart',
             'Recurse': 'VQSort', '__introsort_loop': 'std::sort', 'SampleProjection': 'SampleProj', 'FillExampleBucketSet': 'FillBuckets'}.get(f, f[:16])
        ax.annotate(f + ' @ ' + loc, (row.ai, row.perf), xytext=(6, 4), textcoords='offset points', fontsize=6.5, color='#333')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(1e-3, 1e5 if level == 'DRAM' else 1e2); ax.set_ylim(1e-1, top * 3)
    ax.set_xlabel(f'Arithmetic intensity ({"FLOP" if ops == "float" else "FLOP+INTOP"} / byte, {level} traffic)')
    ax.set_ylabel(f'Performance (G{"FLOP" if ops == "float" else "OP"}/s, 48 threads)')
    ax.set_title(title, fontsize=10); ax.grid(True, which='major', color='#eee'); ax.legend(fontsize=6.5, loc='lower left', markerscale=0.6, framealpha=.9)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    return d


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', required=True); p.add_argument('--title', default=''); p.add_argument('--ops', default='mixed')
    p.add_argument('--level', default='L1'); p.add_argument('--roofs', default=None); p.add_argument('--summary', default=None); p.add_argument('--shape', default='')
    p.add_argument('arms', nargs='+'); a = p.parse_args()
    roofs_override = None
    if a.roofs:
        r = pd.read_csv(a.roofs, skiprows=1); roofs_override = {row['Name']: float(row['Bandwidth']) / 1e9 for _, row in r.iterrows()}
    fig, axes = plt.subplots(1, len(a.arms), figsize=(6 * len(a.arms), 5.4), squeeze=False)
    rows = []
    for ax, spec in zip(axes[0], a.arms):
        label, prefix = spec.split('=', 1); arm = load_arm(prefix)
        d = panel(ax, arm, a.ops, a.level, label, roofs_override)
        d[['node', 'srcloc', 'Self Time', 'Self Elapsed Time', 'ai', 'perf', 'cat']].head(15).to_csv(a.out.replace('.png', f'.{label}.top.csv'), index=False)
        ap, sh = arm['ap'], arm['shares']
        row = dict(shape=a.shape, arm=label, project=os.path.basename(prefix), train_cpu_s=round(arm['train_cpu_s'], 1),
                   AP_pct=round(sh.get('ApplyProjection', 0), 1), hist_pct=round(sh.get('Split search (histogram)', 0), 1),
                   sort_pct=round(sh.get('Split search (sort/scan)', 0), 1), sampling_pct=round(sh.get('Projection sampling', 0), 1), other_pct=round(sh.get('Other', 0), 1),
                   AP_cpu_s=round(ap['cpu_s'], 1), AP_elapsed_s=round(ap['elapsed_s'], 3), AP_GFLOP=round(ap['gflop'], 2), AP_GOP=round(ap['gop'], 2), AP_L1_GB=round(ap['l1_gb'], 1),
                   AP_GFLOPS=round(ap['gflop'] / ap['elapsed_s'], 2), AP_GOPS=round(ap['gop'] / ap['elapsed_s'], 2),
                   AP_L1_AI_float=round(ap['gflop'] / ap['l1_gb'], 4), AP_L1_AI_mixed=round(ap['gop'] / ap['l1_gb'], 4))
        if ap.get('dram_gb') and not np.isnan(ap['dram_gb']):
            row.update(AP_DRAM_GB=round(ap['dram_gb'], 1), AP_DRAM_AI_float=round(ap['gflop'] / ap['dram_gb'], 4), AP_DRAM_AI_mixed=round(ap['gop'] / ap['dram_gb'], 4))
        if 'check_total_gflop' in ap: row.update(check_total_GFLOP=round(ap['check_total_gflop'], 2), check_total_L1_GB=round(ap['check_total_l1_gb'], 1))
        rows.append(row); print(pd.Series(row).to_string(), '\n')
    fig.suptitle(a.title, fontsize=11); fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(a.out, dpi=160)
    if a.summary:
        t = pd.DataFrame(rows)
        if os.path.exists(a.summary):
            old = pd.read_csv(a.summary); old = old[~((old['shape'] == a.shape) & old['arm'].isin(t['arm']))]; t = pd.concat([old, t], ignore_index=True)
        t.to_csv(a.summary, index=False)
