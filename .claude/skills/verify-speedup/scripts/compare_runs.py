#!/usr/bin/env python3
"""Compare two benchmark result CSVs (arm A = baseline, arm B = candidate) and
write a markdown verdict table.

Input CSV format (what benchmarks/evaluation/runtime.sh writes):
    ==== PROVENANCE ====      optional "key: value" header block, closed by a
    ...                       line of '=' signs
    ====================
    dataset,algorithm,median_s,stddev_s
Any CSV whose body has `dataset` and `median_s` (or `time_s`) columns works;
`algorithm` and `stddev_s` are optional.

Usage:
    compare_runs.py A.csv B.csv [--label-a base] [--label-b cand]
                    [--fail-below 15] [--star-at 20] [--out compare.md]
Speedup = A/B (>1 means B is faster); "time saved" = 1 - B/A. Exit code is
always 0: the verdict is in the table, the caller decides what to do with it.
"""
import argparse, csv, io, math, re
from pathlib import Path


def read_result_csv(path):
    text = Path(path).read_text(errors="replace")
    prov, body = {}, text
    if "==== PROVENANCE" in text:
        head, _, body = text.partition("\n====================\n")
        for line in head.splitlines():
            m = re.match(r"^\s*([A-Za-z_][\w ]*?):\s*(.*)$", line)
            if m:
                prov[m.group(1).strip()] = m.group(2).strip()
            for k, v in re.findall(r"([A-Z_]+):\s*(\S+)", line):  # packed keys
                prov.setdefault(k, v)
    rows = {}
    for r in csv.DictReader(io.StringIO(body.strip() + "\n")):
        if not r.get("dataset"):
            continue
        raw = r.get("median_s") or r.get("time_s") or r.get("median") or ""
        try:
            t = float(raw)
        except ValueError:
            t = None  # OOM / ERROR cells stay visible
        rows[r["dataset"]] = {"algorithm": r.get("algorithm", ""), "median_s": t,
                              "raw": raw, "stddev_s": r.get("stddev_s", "")}
    return prov, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--label-a", default="A (baseline)")
    ap.add_argument("--label-b", default="B (candidate)")
    ap.add_argument("--fail-below", type=float, default=15.0, help="%% time saved below which the change failed")
    ap.add_argument("--star-at", type=float, default=20.0, help="%% time saved at/above which the result is ★")
    ap.add_argument("--out", help="also write the markdown here")
    args = ap.parse_args()

    prov_a, a = read_result_csv(args.a)
    prov_b, b = read_result_csv(args.b)
    datasets = list(dict.fromkeys(list(a) + list(b)))

    L = [f"## A/B runtime: {args.label_a} vs {args.label_b}\n",
         "| dataset | algorithm | A median s | B median s | speedup A/B | time saved | verdict |",
         "|---|---|---:|---:|---:|---:|---|"]
    ratios = []
    for d in datasets:
        ra, rb = a.get(d), b.get(d)
        ta = ra["median_s"] if ra else None
        tb = rb["median_s"] if rb else None
        algo = (rb or ra or {}).get("algorithm", "")
        if ta is None or tb is None or tb == 0:
            L.append(f"| {d} | {algo} | {ra['raw'] if ra else '—'} | {rb['raw'] if rb else '—'} | n/a | n/a | missing / OOM / ERROR |")
            continue
        ratio, saved = ta / tb, (1 - tb / ta) * 100
        ratios.append(ratio)
        verdict = ("★ speedup" if saved >= args.star_at else
                   "speedup (below ★ bar)" if saved >= args.fail_below else
                   "no significant change" if saved > -args.fail_below else "SLOWDOWN")
        L.append(f"| {d} | {algo} | {ta:.2f} ± {ra['stddev_s'] or 'n/a'} | {tb:.2f} ± {rb['stddev_s'] or 'n/a'} | {ratio:.2f}× | {saved:+.1f} % | {verdict} |")
    if ratios:
        gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        L.append(f"\nGeometric-mean speedup over {len(ratios)} dataset(s): **{gm:.2f}×** ({(1 - 1 / gm) * 100:+.1f} % time saved). "
                 f"Gates: failed < {args.fail_below:g} %, ★ ≥ {args.star_at:g} %.")
    L += ["\n### Provenance\n", "| field | A | B |", "|---|---|---|"]
    for k in ("date_utc", "git_sha", "git_branch", "machine", "compiler", "EXTRA_BAZEL_CONFIGS",
              "EXTRA_TRAIN_ARGS", "NUM_TREES", "NUM_RUNS", "CSV_DATASETS", "TRUNK_DATASETS"):
        if k in prov_a or k in prov_b:
            L.append(f"| {k} | {prov_a.get(k, '—')} | {prov_b.get(k, '—')} |")
    bad = [k for k in ("machine", "compiler", "NUM_TREES", "NUM_RUNS", "EXTRA_TRAIN_ARGS")
           if k in prov_a and k in prov_b and prov_a[k] != prov_b[k]]
    if bad:
        L.append(f"\n**WARNING: arms differ in {', '.join(bad)}; this is not a clean A/B.**")
    md = "\n".join(L) + "\n"
    print(md)
    if args.out:
        Path(args.out).write_text(md)


if __name__ == "__main__":
    main()
