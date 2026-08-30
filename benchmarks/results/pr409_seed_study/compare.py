import sys, csv, math, collections
orig_path, new_paths = sys.argv[1], sys.argv[2:]
def load(p):
    d = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            d[(r['compiler'], r['arm'], r['test'], r['seed'])] = float(r['metric'])
    return d
orig = load(orig_path)
new = {}
for p in new_paths: new.update(load(p))
bands = {'AbaloneSPO': (2.054, 0.01), 'AdultNWTAWeights': (0.8415, 0.012), 'SimPTELowerBound': (0.10889, 0.002)}
comps = sorted({k[0] for k in new}); tests = ['AbaloneSPO', 'AdultNWTAWeights', 'SimPTELowerBound']
print("== per-seed replication vs original matrix ==")
for c in comps:
    for a in ('base', 'head'):
        for t in tests:
            ks = [k for k in new if k[:3] == (c, a, t)]
            if not ks: continue
            common = [k for k in ks if k in orig]
            same = sum(1 for k in common if abs(new[k]-orig[k]) < 5e-7)
            maxd = max((abs(new[k]-orig[k]) for k in common), default=float('nan'))
            print(f"{c:4s} {a:4s} {t:17s} n={len(ks):3d} common={len(common):3d} identical={same:3d} max|Δ|={maxd:.6f}")
def stats(xs):
    n = len(xs); m = sum(xs)/n; v = sum((x-m)**2 for x in xs)/(n-1) if n > 1 else 0.0
    return n, m, math.sqrt(v)
print("\n== base vs head (new runs) ==")
print(f"{'comp':4s} {'test':17s} {'base mean±sd':>22s} {'head mean±sd':>22s} {'head-base':>10s} {'t':>6s}  base_out_of_band head_out_of_band")
for c in comps:
    for t in tests:
        b = [v for k, v in new.items() if k[:3] == (c, 'base', t)]
        h = [v for k, v in new.items() if k[:3] == (c, 'head', t)]
        if not b or not h: continue
        nb, mb, sb = stats(b); nh, mh, sh = stats(h)
        se = math.sqrt(sb**2/nb + sh**2/nh); tt = (mh-mb)/se if se > 0 else float('nan')
        ctr, w = bands[t]
        ob = sum(1 for x in b if abs(x-ctr) > w); oh = sum(1 for x in h if abs(x-ctr) > w)
        print(f"{c:4s} {t:17s} {mb:.5f}±{sb:.5f} (n={nb:3d}) {mh:.5f}±{sh:.5f} (n={nh:3d}) {mh-mb:+10.5f} {tt:+6.2f}  {ob:3d}/{nb}          {oh:3d}/{nh}")
print("\n== default-seed values (new | orig) ==")
for c in comps:
    for a in ('base', 'head'):
        for t in tests:
            k = (c, a, t, 'default')
            if k in new: print(f"{c:4s} {a:4s} {t:17s} {new[k]:.6f} | {orig.get(k, float('nan')):.6f}")
