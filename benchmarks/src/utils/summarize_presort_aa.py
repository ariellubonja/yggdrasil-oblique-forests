#!/usr/bin/env python3
"""Summarize a presort_aa_study.sh CSV: in_node vs presorted per (dataset, split).

Usage: summarize_presort_aa.py <presort_aa_<suffix>.csv> [--out summary.csv]
Prints a markdown table; --out writes the tidy summary CSV.
"""
import argparse, csv, io, sys

def load(path):
    lines = open(path).read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('dataset,'))
    return list(csv.DictReader(io.StringIO('\n'.join(lines[start:]))))

def f(x):
    try: return float(x)
    except (TypeError, ValueError): return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv'); ap.add_argument('--out')
    a = ap.parse_args()
    rows = load(a.csv)
    cells = {(r['dataset'], r['split_type'], r['strategy']): r for r in rows}
    keys = []
    for r in rows:
        k = (r['dataset'], r['split_type'])
        if k not in keys: keys.append(k)
    out = []
    hdr = ['dataset', 'split_type', 'in_node_s', 'presorted_s', 'delta_pct',
           'speedup_x', 'presort_build_s', 'presort_build_pct_of_presorted',
           'in_node_stddev_s', 'presorted_stddev_s']
    print('| ' + ' | '.join(hdr) + ' |'); print('|' + '---|' * len(hdr))
    for ds, sp in keys:
        i, p = cells.get((ds, sp, 'in_node')), cells.get((ds, sp, 'presorted'))
        ti = f(i['median_s']) if i else None
        tp = f(p['median_s']) if p else None
        pb = f(p['presort_s']) if p else None
        delta = (tp - ti) / ti * 100 if ti and tp else None
        sp_x = ti / tp if ti and tp else None
        share = pb / tp * 100 if pb is not None and tp else None
        rec = [ds, sp,
               f'{ti:.1f}' if ti else '--', f'{tp:.1f}' if tp else '--',
               f'{delta:+.1f}' if delta is not None else '--',
               f'{sp_x:.3f}' if sp_x else '--',
               f'{pb:.2f}' if pb is not None else '--',
               f'{share:.1f}' if share is not None else '--',
               i['stddev_s'] if i else '--', p['stddev_s'] if p else '--']
        out.append(rec)
        print('| ' + ' | '.join(rec) + ' |')
    if a.out:
        with open(a.out, 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(hdr); w.writerows(out)

if __name__ == '__main__':
    main()
