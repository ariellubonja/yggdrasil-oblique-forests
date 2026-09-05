#!/usr/bin/env python3
"""Find earlier result CSVs whose provenance matches a protocol, then either
(a) check that a new baseline run replicates them, or (b) estimate how long a
run will take before starting it.

Matching uses the provenance block runtime.sh prepends to every CSV: machine
model (substring), EXTRA_BAZEL_CONFIGS, EXTRA_TRAIN_ARGS (quotes stripped,
`--k=v` == `--k v`, order-insensitive) and NUM_TREES. The `algorithm` column
is deliberately NOT used: older parsers mislabelled it.

Usage:
  # replication: compare A.csv against the newest matching prior per dataset
  find_prior_baseline.py --results-dir benchmarks/results --like A.csv [--tolerance 3]
  # estimate: expected per-run seconds for a protocol you are about to run
  find_prior_baseline.py --results-dir benchmarks/results --estimate \
      --machine 8488C --configs "<none>" --args "--ensemble_method=Boosting" --trees 300
Exit codes: 0 = replicates (or estimate printed), 1 = drift beyond tolerance,
2 = no matching prior found.
"""
import argparse, csv, io, re, sys
from pathlib import Path


def norm_args(s):
    s = (s or "").strip()
    if s in ("", "<none>"):
        return ""
    s = re.sub(r"--(\w+)=", r"--\1 ", s.replace('"', "").replace("'", ""))
    return " ".join(sorted(re.findall(r"--\w+(?:\s+[^-\s][^\s]*(?:\s+[^-\s][^\s]*)*)?", s)))


def read(path):
    text = Path(path).read_text(errors="replace")
    if "==== PROVENANCE" not in text:
        return None, {}
    head, _, body = text.partition("\n====================\n")
    prov = {}
    for line in head.splitlines():
        m = re.match(r"^\s*([A-Za-z_][\w ]*?):\s*(.*)$", line)
        if m:
            prov[m.group(1).strip()] = m.group(2).strip()
        for k, v in re.findall(r"([A-Z_]+):\s*(\S+)", line):
            prov.setdefault(k, v)
    rows = {}
    try:
        for r in csv.DictReader(io.StringIO(body.strip() + "\n")):
            try:
                rows[r["dataset"]] = float(r["median_s"])
            except (KeyError, TypeError, ValueError):
                pass
    except csv.Error:
        pass
    return prov, rows


def matches(prov, machine, configs, args_, trees):
    if machine and machine not in prov.get("machine", ""):
        return False
    if (prov.get("EXTRA_BAZEL_CONFIGS") or "<none>") != (configs or "<none>"):
        return False
    if norm_args(prov.get("EXTRA_TRAIN_ARGS")) != norm_args(args_):
        return False
    if trees and str(prov.get("NUM_TREES", "")).split()[:1] != [str(trees)]:
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--like", help="new baseline CSV whose provenance defines the protocol")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--machine"); ap.add_argument("--configs"); ap.add_argument("--args"); ap.add_argument("--trees")
    ap.add_argument("--tolerance", type=float, default=3.0, help="max |drift| %% that still counts as replicating")
    ap.add_argument("--exclude", default="accuracy,per_function_timing,archive", help="comma-separated path substrings to skip")
    a = ap.parse_args()

    like_path, new_rows = None, {}
    if a.like:
        prov, new_rows = read(a.like)
        if prov is None:
            sys.exit("--like file has no provenance block")
        machine = re.sub(r"\s*\(nproc=.*\)", "", prov.get("machine", ""))
        configs, args_ = prov.get("EXTRA_BAZEL_CONFIGS", "<none>"), prov.get("EXTRA_TRAIN_ARGS", "<none>")
        trees = (prov.get("NUM_TREES", "").split() or [None])[0]
        like_path = Path(a.like).resolve()
    else:
        machine, configs, args_, trees = a.machine or "", a.configs or "<none>", a.args or "<none>", a.trees

    excl = [e for e in a.exclude.split(",") if e]
    cands = []
    for p in Path(a.results_dir).rglob("*.csv"):
        if any(e in str(p) for e in excl) or p.resolve() == like_path:
            continue
        prov, rows = read(p)
        if prov and rows and matches(prov, machine, configs, args_, trees):
            cands.append((prov.get("date_utc", ""), p, prov, rows))
    if not cands:
        print(f"No prior CSV under {a.results_dir} matches: machine~'{machine}' configs='{configs}' args='{args_}' trees={trees}")
        sys.exit(2)
    cands.sort(key=lambda c: c[0], reverse=True)

    print(f"Protocol: machine~'{machine}' configs='{configs}' args='{args_}' trees={trees}")
    print(f"{len(cands)} matching prior CSV(s), newest first:")
    for date, p, prov, _ in cands[:5]:
        print(f"  {date[:10]}  {prov.get('git_sha', '?'):<16} {p}")
    datasets = list(new_rows) or sorted({d for *_, rows in cands for d in rows})
    worst = 0.0
    print()
    print("| dataset | prior median s | prior date / sha | " + ("new median s | drift |" if new_rows else "expected per run |"))
    print("|---|---:|---|" + ("---:|---:|" if new_rows else "---|"))
    for d in datasets:
        prior = next(((date, prov, rows[d]) for date, p, prov, rows in cands if d in rows), None)
        if prior is None:
            print(f"| {d} | — | no prior | " + (f"{new_rows[d]:.2f} | n/a |" if new_rows else "unknown |"))
            continue
        date, prov, t0 = prior
        if new_rows:
            drift = (new_rows[d] / t0 - 1) * 100
            worst = max(worst, abs(drift))
            flag = "" if abs(drift) <= a.tolerance else " **DRIFT**"
            print(f"| {d} | {t0:.2f} | {date[:10]} / {prov.get('git_sha', '?')} | {new_rows[d]:.2f} | {drift:+.1f} %{flag} |")
        else:
            print(f"| {d} | {t0:.2f} | {date[:10]} / {prov.get('git_sha', '?')} | ≈ {t0 / 60:.0f} min |")
    if new_rows:
        ok = worst <= a.tolerance
        print(f"\nReplication: worst |drift| {worst:.1f} % vs tolerance {a.tolerance:g} % → "
              + ("REPLICATES" if ok else "DOES NOT REPLICATE (code, build or machine state changed)"))
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
