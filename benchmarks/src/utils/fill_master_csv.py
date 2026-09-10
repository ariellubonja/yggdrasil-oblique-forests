#!/usr/bin/env python3
"""Auto-fill the full-depth (Depth -1) block of the master E2E CSV from result CSVs.

Fills, in the "Depth -1" median block and the "st dev - -1 depth" stddev block:
  col 3 = BFS row-major  (HIGGS only; trunk already present)
  col 5 = DW1 col-major  (Depthwise 1-pass BFS Col-Major)
  col 6 = DW1 row-major
Numeric median -> integer; stddev -> 1 decimal, leading zero stripped (".7").
An OOM/ERROR cell -> that label in the median cell, blank stddev.
Backs up the master CSV first; writes with '\n' line endings.

Usage: fill_master_csv.py <master_csv> <results_dir>
"""
import csv
import os
import shutil
import sys

MASTER_HDR_MED = "Depth -1"
MASTER_HDR_STD = "st dev - -1 depth"
DATASETS = ["HIGGS 11m x 29", "Trunk 1.5m x 4096", "Trunk 150k x 40k", "Trunk 15k x 400k"]

# (master_dataset, column, [ (result_csv_basename, src_dataset_name) ... priority order ])
TARGETS = []
_DW = [
    (5, "3runs_dw1_colmajor_no_bootstrap.csv"),
    (6, "3runs_dw1_rowmajor_no_bootstrap.csv"),
]
_SRC = {
    "HIGGS 11m x 29": "HIGGS_with_header",
    "Trunk 1.5m x 4096": "trunk_1500000_x_4096",
    "Trunk 150k x 40k": "trunk_150000_x_40000",
    "Trunk 15k x 400k": "trunk_15000_x_400000",
}
for col, fn in _DW:
    for md in DATASETS:
        TARGETS.append((md, col, [(fn, _SRC[md])]))
# BFS row-major HIGGS: prefer the 7-run rerun, fall back to the 3-run.
TARGETS.append(("HIGGS 11m x 29", 3, [
    ("7runs_bfs_row_nb_full_higgs_higgs_rerun.csv", "HIGGS_with_header"),
    ("3runs_bfs_row_nb_full_higgs.csv", "HIGGS_with_header"),
]))


def fmt_med(v):
    return str(int(round(v)))


def fmt_std(v):
    s = f"{v:.1f}"
    return s[1:] if s.startswith("0.") else s


def load_results(path):
    """basename(no ext) result CSV -> {dataset: (median_s, stddev_s)}"""
    out = {}
    with open(path) as f:
        seen = False
        for line in csv.reader(f):
            if not line:
                continue
            if not seen:
                if line[0] == 'dataset':
                    seen = True
                continue
            if len(line) >= 3:
                out[line[0]] = (line[2], line[3] if len(line) > 3 else '')
    return out


def block_rows(rows, header_label):
    """Return {dataset_name: row_index} for the 4 dataset rows after header_label."""
    out = {}
    started = False
    for i, r in enumerate(rows):
        c0 = r[0].strip() if r else ''
        if not started:
            if c0 == header_label:
                started = True
            continue
        if c0 in DATASETS:
            out[c0] = i
            if len(out) == len(DATASETS):
                break
        elif c0 and c0 not in DATASETS and out:
            break  # next header -> block ended
    return out


def main():
    if len(sys.argv) != 3:
        print("usage: fill_master_csv.py <master_csv> <results_dir>", file=sys.stderr)
        return 2
    master, results_dir = sys.argv[1], sys.argv[2]

    shutil.copy2(master, master + ".bak")
    print(f"backup: {master}.bak")

    with open(master, newline='') as f:
        rows = [r for r in csv.reader(f)]

    med_idx = block_rows(rows, MASTER_HDR_MED)
    std_idx = block_rows(rows, MASTER_HDR_STD)
    if len(med_idx) != 4 or len(std_idx) != 4:
        print(f"ERROR: could not locate both blocks (med={list(med_idx)}, std={list(std_idx)})",
              file=sys.stderr)
        return 1

    cache = {}
    def results_for(fn):
        if fn not in cache:
            p = os.path.join(results_dir, fn)
            cache[fn] = load_results(p) if os.path.exists(p) else {}
        return cache[fn]

    changes = []
    for master_ds, col, candidates in TARGETS:
        med_s = std_s = None
        used = None
        for fn, src in candidates:
            data = results_for(fn)
            if src in data:
                med_s, std_s = data[src]
                used = fn
                break
        if med_s is None:
            print(f"# SKIP {master_ds} col{col}: no source CSV/row yet", file=sys.stderr)
            continue
        try:
            med_cell = fmt_med(float(med_s))
            std_cell = fmt_std(float(std_s)) if std_s not in ('', None) else ''
        except ValueError:
            med_cell = med_s.strip().upper()   # OOM / ERROR
            std_cell = ''
        rows[med_idx[master_ds]][col] = med_cell
        rows[std_idx[master_ds]][col] = std_cell
        changes.append(f"{master_ds:18s} col{col} = {med_cell:>6s}  (std {std_cell or '-'})  <- {used}")

    with open(master, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        for r in rows:
            w.writerow(r)

    print(f"Filled {len(changes)} cells in {master}:")
    for c in changes:
        print("  " + c)
    return 0


if __name__ == '__main__':
    sys.exit(main())
