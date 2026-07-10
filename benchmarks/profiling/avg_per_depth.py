#!/usr/bin/env python3
"""Collapse parallel_chrono per-function-timing CSVs into per-depth averages.

The input CSV (as produced by parallel_chrono.py) has one repeated block of
columns per thread (blocks are separated by an empty column and each start
with "tree,depth,..."), and within each thread-block the rows for several
trees are stacked back-to-back (depth resets to 0 whenever a new tree
starts). The set of metric columns differs between COARSE and FINE runs, so
the column layout is read from the header instead of being hardcoded.

For every depth this script pools all (thread, tree) rows with that depth --
i.e. it averages across threads and across the multiple trees per thread at
once -- and writes one row per depth. Note: a depth is only averaged over the
trees that actually reached it, so the pool size shrinks at large depths.

Depth-0 rows carry only the total TreeTrain time; they are kept in the
output, and each file also gets a sanity check printed: mean TreeTrain per
tree vs. the mean of the per-depth NodeTrain sums (these should roughly
agree if the averaging is sound).

Provenance / freshness
----------------------
Each <name>_avg_per_depth.csv starts with a single "#"-comment metadata line
that fingerprints the raw source CSV it was derived from, e.g.

  #avg_per_depth_meta format=1 source=dfs_colmajor_multicore.csv sha256=<hex> \
      src_size=1481554 src_mtime=2026-07-07T13:38:00Z generated=2026-07-07T14:40:00Z

The authoritative change signal is the sha256 of the source's *bytes*; format
is bumped when this script's output schema changes. Because the first line is
a comment, read these with e.g. pandas' ``read_csv(path, comment="#")``.

Layout
------
parallel_chrono.py writes raw timing CSVs into a ``raw/`` subfolder of the
``<dataset_name>`` dir, e.g. ``.../<dataset_name>/raw/dfs_colmajor_multicore.csv``.
This script writes the per-depth average one level up, in ``<dataset_name>``
itself, reusing the source's name (``.../<dataset_name>/dfs_colmajor_multicore.csv``)
-- the raw/ vs. parent split is what distinguishes raw from average, so the
average no longer needs a suffix. A raw CSV that is *not* under a ``raw/`` dir
falls back to the legacy ``<name>_avg_per_depth.csv`` name in its own dir, so
the average can never overwrite the raw source.

Usage
-----
  avg_per_depth.py <csv|dir> [<csv|dir> ...]
      For each file argument: (re)writes the per-depth average for it
      (unconditionally, as before). For each directory argument: recursively
      scans it and (re)generates only the per-depth CSVs that are missing or
      whose source CSV has changed (see the fingerprint above).

  avg_per_depth.py --compare <csvA> <csvB> [-o out.csv]
      Also writes a long-format comparison of the metric columns the two
      files share: depth, metric, <A>, <B>, ratio B/A. Default output is
      compare_<stemA>_vs_<stemB>.csv next to <csvA>.

Scan-only flags: --force (regenerate every chrono CSV regardless of the
fingerprint), --dry-run (report what would be regenerated, write nothing),
-v/--verbose (also list CSVs skipped for not looking like chrono output).
"""
import argparse
import csv
import hashlib
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

# Bump when the per-depth output schema changes so existing files whose source
# is unchanged still get regenerated on the next scan.
FORMAT_VERSION = 1
AVG_SUFFIX = "_avg_per_depth.csv"
META_PREFIX = "#avg_per_depth_meta "
# csv.writer under open(newline="") emits CRLF; match it for the meta line.
LINE_TERM = "\r\n"


def norm(name):
    """Metric names carry leading dashes as nesting markers (e.g. -NodeTrain,
    ----ApplyProjection) and the nesting differs between files; strip them so
    the same function matches across files."""
    return name.lstrip("-")


def read_blocks(path):
    """Return (metric_names, samples) where samples is {depth: [row_dict, ...]}.

    Raises ValueError if the file has no per-thread "tree" header, i.e. it does
    not look like a parallel_chrono per-function-timing CSV."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))

    header_idx = next((i for i, r in enumerate(rows) if "tree" in r), None)
    if header_idx is None:
        raise ValueError("no 'tree' header row -- not a chrono timing CSV")
    header = rows[header_idx]
    block_starts = [i for i, v in enumerate(header) if v == "tree"]

    # Column names of one block: from "tree" up to the blank separator.
    b0 = block_starts[0]
    end = b0 + 1
    while end < len(header) and header[end] not in ("", "tree"):
        end += 1
    names = header[b0:end]
    depth_off = names.index("depth")
    metrics = [n for n in names if n not in ("tree", "depth")]
    offsets = [names.index(n) for n in metrics]

    samples = defaultdict(list)
    for row in rows[header_idx + 1:]:
        for b in block_starts:
            if b + len(names) > len(row) or row[b + depth_off] == "":
                continue
            depth = int(row[b + depth_off])
            samples[depth].append(
                [float(row[b + o]) if row[b + o] != "" else 0.0 for o in offsets])
    return metrics, samples


def read_labeled_cell(path, label):
    """Value of the cell immediately following the first `label` cell in the raw
    CSV's cmds section (e.g. label="Machine serial" -> the dmidecode serial), or
    "unknown" if absent. Spaces are collapsed to "_" because the meta line is a
    space-tokenized key=value string and the value must stay a single token."""
    try:
        with open(path, newline="") as f:
            for row in csv.reader(f):
                for i, cell in enumerate(row):
                    if cell == label and i + 1 < len(row) and row[i + 1]:
                        return "_".join(row[i + 1].split())
    except OSError:
        pass
    return "unknown"


def averages(metrics, samples):
    """Return {depth: (count, [mean per metric])}."""
    out = {}
    for depth, vals in sorted(samples.items()):
        n = len(vals)
        out[depth] = (n, [sum(v[i] for v in vals) / n for i in range(len(metrics))])
    return out


def sanity_check(path, metrics, samples):
    normed = [norm(m) for m in metrics]
    if "TreeTrain" not in normed or "NodeTrain" not in normed or 0 not in samples:
        return
    tt, nt = normed.index("TreeTrain"), normed.index("NodeTrain")
    n_trees = len(samples[0])
    mean_tree_train = sum(v[tt] for v in samples[0]) / n_trees
    total_node_train = sum(v[nt] for vals in samples.values() for v in vals)
    print(f"  sanity ({os.path.basename(path)}): mean TreeTrain/tree = "
          f"{mean_tree_train:.3f}s, mean sum(NodeTrain)/tree = "
          f"{total_node_train / n_trees:.3f}s over {n_trees} trees")


def stem(path):
    base = os.path.basename(path)
    while base.endswith(".csv"):
        base = base[:-4]
    return base


def avg_path_for(raw_path):
    """Path of the per-depth CSV that write_avg() produces for raw_path.

    New layout: raw at ``<ds>/raw/<name>.csv`` -> average at ``<ds>/<name>.csv``
    (the raw/ vs. parent split is what distinguishes them, so the average keeps
    the source's name -- no suffix). Legacy layout (a raw *not* inside a ``raw/``
    dir) falls back to the ``<name>_avg_per_depth.csv`` suffix in the same dir,
    so the average can never overwrite the raw source."""
    d = os.path.dirname(raw_path)
    if os.path.basename(d) == "raw":
        return os.path.join(os.path.dirname(d), os.path.basename(raw_path))
    return os.path.join(d, stem(raw_path) + AVG_SUFFIX)


# --- source fingerprint / provenance ---------------------------------------

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _iso(ts):
    return (datetime.fromtimestamp(ts, timezone.utc)
            .replace(microsecond=0).isoformat().replace("+00:00", "Z"))


def source_meta(path, sha=None, source_name=None):
    """Fingerprint dict for the raw source CSV at `path`. `source_name` is what
    to record as the source (e.g. 'raw/foo.csv' relative to the average's dir);
    defaults to the bare basename."""
    st = os.stat(path)
    return {
        "format": FORMAT_VERSION,
        "source": source_name or os.path.basename(path),
        "machine": read_labeled_cell(path, "Machine"),
        "machine_serial": read_labeled_cell(path, "Machine serial"),
        "sha256": sha or file_sha256(path),
        "src_size": st.st_size,
        "src_mtime": _iso(st.st_mtime),
        "generated": _iso(time.time()),
    }


def format_meta_line(meta):
    keys = ("format", "source", "machine", "machine_serial", "sha256", "src_size", "src_mtime", "generated")
    return META_PREFIX + " ".join(f"{k}={meta[k]}" for k in keys)


def parse_meta_line(line):
    """Inverse of format_meta_line; None if `line` is not a metadata line.

    Only sha256/format drive freshness, and those tokens are comma/space-free,
    so a source basename containing spaces (which these files never have) would
    at worst garble the informational `source` field -- never sha256/format."""
    if not line.startswith(META_PREFIX):
        return None
    out = {}
    for tok in line[len(META_PREFIX):].split():
        k, _, v = tok.partition("=")
        out[k] = v
    return out


def read_meta(avg_path):
    """Metadata dict recorded in an existing per-depth CSV, or None."""
    try:
        with open(avg_path, newline="") as f:
            first = f.readline()
    except OSError:
        return None
    return parse_meta_line(first.rstrip("\r\n"))


def staleness_reason(raw_path, avg_path, sha):
    """Why avg_path must be regenerated from raw_path, or None if up to date.
    `sha` is the current sha256 of raw_path (passed in to avoid re-hashing)."""
    if not os.path.exists(avg_path):
        return "missing"
    meta = read_meta(avg_path)
    if meta is None:
        return "no-meta"
    if meta.get("format") != str(FORMAT_VERSION):
        return "format-change"
    if meta.get("sha256") != sha:
        return "source-changed"
    return None


# --- conversion -------------------------------------------------------------

def write_avg(path, sha=None):
    metrics, samples = read_blocks(path)
    avg = averages(metrics, samples)
    out_path = avg_path_for(path)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    source_rel = os.path.relpath(path, out_dir)
    with open(out_path, "w", newline="") as f:
        f.write(format_meta_line(source_meta(path, sha, source_rel)) + LINE_TERM)
        w = csv.writer(f)
        w.writerow(["depth"] + metrics)
        for depth, (n, means) in avg.items():
            w.writerow([depth] + [f"{m:.2f}" for m in means])
    print(f"Wrote {out_path}")
    sanity_check(path, metrics, samples)
    return metrics, avg


def label(path):
    for part in path.split(os.sep):
        if part in ("FINE", "COARSE"):
            return f"{part}:{stem(path)}"
    return stem(path)


def write_compare(path_a, path_b, out_path):
    (metrics_a, avg_a), (metrics_b, avg_b) = write_avg(path_a), write_avg(path_b)
    normed_b = [norm(m) for m in metrics_b]
    common = [norm(m) for m in metrics_a if norm(m) in normed_b]
    ia = {m: [norm(x) for x in metrics_a].index(m) for m in common}
    ib = {m: normed_b.index(m) for m in common}
    la, lb = label(path_a), label(path_b)

    if out_path is None:
        out_path = os.path.join(os.path.dirname(path_a),
                                f"compare_{stem(path_a)}_vs_{stem(path_b)}.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["depth", "metric", la, lb, "ratio B/A"])
        for depth in sorted(set(avg_a) | set(avg_b)):
            for m in common:
                a = avg_a[depth][1][ia[m]] if depth in avg_a else ""
                b = avg_b[depth][1][ib[m]] if depth in avg_b else ""
                ratio = b / a if a != "" and b != "" and a != 0 else ""
                w.writerow([depth, m, a, b, ratio])
    print(f"Wrote {out_path}")

    # Per-metric totals over all depths, as a quick overall comparison.
    print(f"\n  totals over all depths ({la} vs {lb}):")
    for m in common:
        ta = sum(v[1][ia[m]] for v in avg_a.values())
        tb = sum(v[1][ib[m]] for v in avg_b.values())
        ratio = f"{tb / ta:6.3f}x" if ta else "   n/a"
        print(f"    {m:<20} {ta:12.4f} {tb:12.4f}  {ratio}")


# --- recursive scan ---------------------------------------------------------

def iter_raw_csvs(root):
    """Candidate raw chrono CSVs under root: *.csv that are neither our own
    per-depth outputs nor compare outputs. (Chrono-ness is confirmed later.)"""
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            if not name.endswith(".csv"):
                continue
            if name.endswith(AVG_SUFFIX) or name.startswith("compare_"):
                continue
            yield os.path.join(dirpath, name)


def looks_like_chrono(path, max_rows=25):
    """Cheap check (reads only the first rows) that `path` is a parallel_chrono
    timing CSV -- so the scan can skip datasets/other CSVs without loading them."""
    try:
        with open(path, newline="") as f:
            for i, row in enumerate(csv.reader(f)):
                if i >= max_rows:
                    break
                if "tree" in row:
                    return True
    except (OSError, csv.Error, UnicodeDecodeError):
        return False
    return False


def scan_dir(root, force=False, dry_run=False, verbose=False):
    converted = up_to_date = skipped = errors = 0
    for raw in iter_raw_csvs(root):
        # A per-depth average (which now carries no distinguishing suffix) is
        # recognised by its metadata line and is never itself a source.
        if read_meta(raw) is not None:
            continue
        if not looks_like_chrono(raw):
            skipped += 1
            if verbose:
                print(f"[skip non-chrono] {raw}")
            continue
        avg = avg_path_for(raw)
        try:
            sha = file_sha256(raw)
            reason = "forced" if force else staleness_reason(raw, avg, sha)
            if reason is None:
                up_to_date += 1
                continue
            if dry_run:
                print(f"[regenerate: {reason}] {raw}")
            else:
                print(f"[{reason}] {raw}")
                write_avg(raw, sha=sha)
            converted += 1
        except Exception as e:  # keep scanning past one bad file
            errors += 1
            print(f"[error] {raw}: {e}", file=sys.stderr)

    verb = "to regenerate" if dry_run else "regenerated"
    print(f"\nscan {root}: {converted} {verb}, {up_to_date} up-to-date, "
          f"{skipped} non-chrono skipped, {errors} errors")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+",
                   help="raw chrono CSV files and/or directories to scan recursively")
    p.add_argument("--compare", action="store_true",
                   help="compare exactly two files on their common columns")
    p.add_argument("-o", "--output", default=None,
                   help="output path for the comparison CSV")
    p.add_argument("--force", action="store_true",
                   help="scan: regenerate every chrono CSV, ignoring the fingerprint")
    p.add_argument("--dry-run", action="store_true",
                   help="scan: only report what would be regenerated; write nothing")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="scan: also list CSVs skipped for not looking like chrono output")
    args = p.parse_args()

    if args.compare:
        files = [x for x in args.paths if os.path.isfile(x)]
        if len(args.paths) != 2 or len(files) != 2:
            sys.exit("--compare needs exactly two CSV files")
        write_compare(args.paths[0], args.paths[1], args.output)
        return

    for path in args.paths:
        if os.path.isdir(path):
            scan_dir(path, force=args.force, dry_run=args.dry_run,
                     verbose=args.verbose)
        elif os.path.isfile(path):
            write_avg(path)  # explicit file: always (re)convert, as before
        else:
            print(f"[error] not found: {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
