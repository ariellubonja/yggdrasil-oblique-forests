#!/usr/bin/env python3
"""Per-cell summary for the AA-vs-SPO cause study: tree time, per-layer FindSplits (median worker), and
per-node CPU use (mean busy vCPUs over the training plateau) from cpu_<role>.json written by train.py.
Usage: python analyze_aa_threading.py <cell_dir>... (e.g. .../barriers/vertex/cpu20261026_higgs/*)
"""
import csv, collections, glob, json, os, sys


def layers(cell):
    """Per-layer mean of the median worker's FindSplits time, and the share of the training window spent in FindSplits barriers."""
    f = os.path.join(cell, "barriers.csv")
    if not os.path.exists(f):
        return {}, None
    by, fs_barrier, all_barrier = collections.defaultdict(list), 0.0, 0.0
    for r in csv.DictReader(open(f)):
        if int(r["iter"]) < 0:
            continue
        all_barrier += float(r["barrier_s"])
        if r["stage"] == "FindSplits":
            by[int(r["layer"])].append(float(r["worker_med_s"]))
            fs_barrier += float(r["barrier_s"])
    return {l: sum(v) / len(v) for l, v in sorted(by.items())}, fs_barrier / all_barrier if all_barrier else None


def plateau(samples, ncpu, min_busy=0.08, min_len_s=20):
    """Longest run of samples with busy fraction > min_busy; returns (mean busy vCPUs, start_s, end_s)."""
    best, cur = [], []
    for s in samples:
        if s[1] > min_busy:
            cur.append(s)
        else:
            if len(cur) > len(best):
                best = cur
            cur = []
    if len(cur) > len(best):
        best = cur
    if not best or best[-1][0] - best[0][0] < min_len_s:
        return None
    return sum(s[1] for s in best) / len(best) * ncpu, best[0][0], best[-1][0]


def buckets(samples, ncpu, width=30):
    """Mean busy vCPUs per `width`-second bucket, as a compact string."""
    b = collections.defaultdict(list)
    for t, busy, _ in samples:
        b[int(t // width)].append(busy * ncpu)
    return " ".join(f"{sum(v) / len(v):.1f}" for _, v in sorted(b.items()))


for cell in sys.argv[1:]:
    ej = os.path.join(cell, "eval.json")
    if not os.path.exists(ej):
        print(f"{cell}: no eval.json"); continue
    e = json.load(open(ej))
    kw = {k: v for k, v in e.get("learner_kwargs", {}).items() if k not in ("split_axis", "num_trees")}
    print(f"\n== {os.path.basename(os.path.dirname(cell))}/{os.path.basename(cell)}  {e['machine_type']}  W={e['workers']}  "
          f"tree_training_s={e.get('tree_training_s')}  setup_s={e.get('setup_s')}  acc={e.get('accuracy')}  kwargs={kw}")
    ls, fs_share = layers(cell)
    if ls:
        print("   FindSplits median worker per layer (s): " + "  ".join(f"L{l}:{v:.2f}" for l, v in ls.items())
              + f"   | FindSplits share of barrier time: {fs_share:.2f}")
    for f in sorted(glob.glob(os.path.join(cell, "cpu_*.json"))):
        c = json.load(open(f))
        p = plateau(c["samples"], c["ncpu"])
        if p and fs_share and c["role"] != "manager":
            # busy vCPUs during FindSplits itself, if the worker idles in every other stage (upper bound)
            ptxt = f"plateau {p[0]:.1f}/{c['ncpu']} busy vCPUs (~{min(p[0] / fs_share, c['ncpu']):.1f} during FindSplits) over [{p[1]:.0f},{p[2]:.0f}] s"
        elif p:
            ptxt = f"plateau {p[0]:.1f}/{c['ncpu']} busy vCPUs over [{p[1]:.0f},{p[2]:.0f}] s"
        else:
            ptxt = "no plateau"
        print(f"   {c['role']:9} {ptxt} | 30 s buckets: {buckets(c['samples'], c['ncpu'])}")
