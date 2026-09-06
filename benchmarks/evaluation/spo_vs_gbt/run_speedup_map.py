#!/usr/bin/env python3
"""Driver 2 (SPEC.md + v2 addendum A7): speedup map grid runner — YDF-fork arms
only, timing only.

Protocol (PROTOCOL.md, part B): how our optimizations (Dynamic Random Histogram
+ vectorized histograms/VQSort) speed up ApplyProjection as a function of
dataset size, feature width, tree depth, and the min_examples stopping
criterion, holding everything else fixed. Designs are shipped as static cell
lists so they can be run without touching this file:

  B1 (size x depth):    cells_b1.json — HIGGS prefixes {100k..10.5M} x
                         max_depth {6,10,15,20,unlimited} x {exact_hwy,
                         dyn_vec}, plus the four remaining RF arms at
                         unlimited depth only.
  B2 (stopping crit.):  cells_b2.json — HIGGS {1M, 10.5M} x min_examples
                         {1,5,20,100} x {exact_hwy, dyn_vec}, unlimited depth.
  B4 (GBT depth axis):  cells_b4.json (D9) — HIGGS-1M + GiveMeSomeCredit x
                         max_depth {6,10,unlimited} x {spo_gbt_exact_hwy,
                         spo_gbt_dyn_vec}. GBT-family cells: no special driver
                         handling is needed beyond what already applies —
                         --ensemble_method Boosting is part of the arm's own
                         ydf_flags in arms.py, and each cell's own depth /
                         min_examples / trees (300, from the arm) flow through
                         the normal per-cell fields below.
  B5 (feature width):   cells_b5.json (D10) — synthetic "trunk" dataset,
                         1M rows x cols {32,128,512,2048,8192}, unlimited
                         depth, {spo_rf_exact_stdsort, spo_rf_exact_hwy,
                         spo_rf_dyn_vec}, 240 trees.

Each grid *cell* is a dict with the required keys "depth", "min_examples",
"arm", "rows", plus an optional "input_mode" ("csv", the default, or
"trunk") that selects how the row/feature data is sourced:
  - input_mode "csv" (default, back-compatible with the original cells_b1/b2
    files): "rows" is looked up in the built-in HIGGS-prefix rows->CSV map
    (or --train-csv overrides) unless the cell gives an explicit "csv" path;
    "label_col" defaults to --label-col ("class") unless the cell overrides
    it; the output "dataset" column defaults to f"higgs_{rows}" unless the
    cell gives an explicit "dataset" name (cells_b4.json's GiveMeSomeCredit
    entries use "csv"/"label_col"/"dataset" to point at a non-HIGGS,
    non-rows-keyed CSV — see SPEC.md A7).
  - input_mode "trunk": no CSV at all — the harness generates a synthetic
    dataset in-process (--input_mode trunk --rows R --cols C, see
    examples/train_oblique_forest.cc). "cols" is then required; the output
    "dataset" column is f"trunk_{rows}_x_{cols}" and "features" = cols
    (cells_b5.json).

Per-cell trees: SPEC.md A7 says trees come from the cell's own arm definition
(arms.py's "trees" field, e.g. 240 for RF arms, 300 for GBT arms) rather than
a single global --trees flag. --trees is now an *optional override*: when
given, it is used for every cell (an intentional global override, e.g. for a
fast smoke run); when omitted (the default), each cell uses its own arm's
"trees" value, which is always internally consistent by construction — no
mismatch is possible since each cell always draws from exactly one arm's own
number. The old "trees consistency check" (which used to hard-exit whenever
a cell's arm.trees != a single global --trees) is adapted accordingly: it now
only fires (as a non-fatal, informational note) when --trees is explicitly
given AND differs from some referenced arm's own default, since that case is
now a deliberate, allowed override rather than a latent bug.

No --test_csv is passed in either mode — this driver only measures the
"Training block took" timing line, never accuracy (that's Driver 1's job on
the suite + huge datasets). Resumable: a cell already present in --out (keyed
by (dataset, max_depth, min_examples, arm), status == "OK") is skipped unless
--no-skip-existing.

Arm definitions (family, engine, binary, ydf_flags, trees, tree_depth) come
from arms.py in this directory (a colleague's concurrent deliverable per
SPEC.md); only engine == "ydf_fork" arms are usable here (this driver never
shells out to xgboost/lightgbm/catboost). If arms.py is missing or fails to
import, a small fallback ARMS table (the 7 RF arms from SPEC.md, family="rf")
is used instead so this script stays testable in isolation -- see
_FALLBACK_ARMS below and the note this prints to stderr when it is used.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BIN_DIR = "/home/ubuntu/spo_vs_gbt/bin"
DEFAULT_LOG_DIR = "/home/ubuntu/spo_vs_gbt/logs/runs"
REPO_ROOT = "/home/ubuntu/yggdrasil-oblique-forests"

TIME_RE = re.compile(r"Training block took:\s*([0-9.eE+-]+)")
PRE_RE = re.compile(r"Loading/Init \(pre-train\):\s*([0-9.eE+-]+)s")

# --------------------------------------------------------------------------
# Default HIGGS row-prefix -> CSV path map (PROTOCOL.md "Size axis"; SPEC.md
# Driver 2 CLI). Overridable per-cell rows value with repeatable --train-csv.
# Used for input_mode "csv" cells that don't give an explicit "csv" path.
# --------------------------------------------------------------------------
DEFAULT_ROWS_TO_CSV = {
    100_000: "/home/ubuntu/spo_vs_gbt/data/higgs_train_100000.csv",
    300_000: "/home/ubuntu/spo_vs_gbt/data/higgs_train_300000.csv",
    1_000_000: "/home/ubuntu/spo_vs_gbt/data/higgs_train_1000000.csv",
    3_000_000: "/home/ubuntu/spo_vs_gbt/data/higgs_train_3000000.csv",
    10_500_000: os.path.join(REPO_ROOT, "benchmarks/data/HIGGS_train_10500k.csv"),
}
DEFAULT_LABEL_COL = "class"

# Harness flags this driver always sets itself; an arm's ydf_flags must not
# duplicate them (interface contract with arms.py, see SPEC.md Driver 2).
_DRIVER_OWNED_FLAGS = {
    "--num_trees", "--tree_depth", "--min_examples", "--num_threads", "--seed",
    "--train_csv", "--test_csv", "--label_col", "--input_mode",
    "--rows", "--cols",
}

# Fallback ARMS, used only if arms.py is absent/broken (see module docstring).
# RF arms only, from SPEC.md's "Arms" table; family=rf, trees=240.
_FALLBACK_ARMS: dict[str, dict] = {
    "spo_rf_exact_stdsort": {
        "family": "rf", "engine": "ydf_fork", "binary": "exact_std_sort",
        "ydf_flags": ["--feature_split_type", "Oblique", "--numerical_split_type", "Exact"],
        "trees": 240, "tree_depth": -1,
    },
    "spo_rf_exact_hwy": {
        "family": "rf", "engine": "ydf_fork", "binary": "default",
        "ydf_flags": ["--feature_split_type", "Oblique", "--numerical_split_type", "Exact"],
        "trees": 240, "tree_depth": -1,
    },
    "spo_rf_rand_scalar": {
        "family": "rf", "engine": "ydf_fork", "binary": "scalar",
        "ydf_flags": ["--feature_split_type", "Oblique", "--numerical_split_type", "Random",
                      "--histogram_num_bins", "64"],
        "trees": 240, "tree_depth": -1,
    },
    "spo_rf_rand_vec": {
        "family": "rf", "engine": "ydf_fork", "binary": "default",
        "ydf_flags": ["--feature_split_type", "Oblique", "--numerical_split_type", "Random",
                      "--histogram_num_bins", "64"],
        "trees": 240, "tree_depth": -1,
    },
    "spo_rf_dyn_scalar": {
        "family": "rf", "engine": "ydf_fork", "binary": "scalar",
        "ydf_flags": ["--feature_split_type", "Oblique",
                      "--numerical_split_type", "Dynamic Random Histogram",
                      "--histogram_num_bins", "64", "--dynamic_split_threshold", "250"],
        "trees": 240, "tree_depth": -1,
    },
    "spo_rf_dyn_vec": {
        "family": "rf", "engine": "ydf_fork", "binary": "default",
        "ydf_flags": ["--feature_split_type", "Oblique",
                      "--numerical_split_type", "Dynamic Random Histogram",
                      "--histogram_num_bins", "64", "--dynamic_split_threshold", "250"],
        "trees": 240, "tree_depth": -1,
    },
    "aa_rf_exact": {
        "family": "rf", "engine": "ydf_fork", "binary": "default",
        "ydf_flags": ["--feature_split_type", "Axis Aligned", "--numerical_split_type", "Exact"],
        "trees": 240, "tree_depth": -1,
    },
}


def load_arms() -> tuple[dict[str, dict], bool]:
    """Import ARMS from arms.py in this directory; fall back if unavailable.

    Returns (arms, used_fallback).
    """
    sys.path.insert(0, THIS_DIR)
    try:
        import arms as arms_module  # type: ignore
        importlib_arms = getattr(arms_module, "ARMS", None)
        if not isinstance(importlib_arms, dict) or not importlib_arms:
            raise ImportError("arms.py has no non-empty ARMS dict")
        return importlib_arms, False
    except Exception as exc:  # noqa: BLE001 - any import failure -> fallback
        print(f"[run_speedup_map] WARNING: could not import arms.ARMS from "
              f"{THIS_DIR}/arms.py ({exc!r}); using built-in fallback RF arms "
              f"table ({len(_FALLBACK_ARMS)} arms). Results using this fallback "
              f"should be re-verified once arms.py lands.", file=sys.stderr)
        return _FALLBACK_ARMS, True
    finally:
        if THIS_DIR in sys.path:
            sys.path.remove(THIS_DIR)


def check_driver_owned_flags(arms: dict[str, dict]) -> None:
    """Fail loudly if any arm's ydf_flags duplicates a flag this driver already
    sets itself (the _DRIVER_OWNED_FLAGS interface contract with arms.py)."""
    for name, spec in arms.items():
        flags = spec.get("ydf_flags") or []
        tokens = {f.split("=", 1)[0] for f in flags
                  if isinstance(f, str) and f.startswith("--")}
        bad = sorted(tokens & _DRIVER_OWNED_FLAGS)
        if bad:
            sys.exit(f"arm {name!r} ydf_flags duplicates driver-owned flag(s) "
                      f"{bad}; remove them from arms.py (this driver always sets "
                      f"them itself, see _DRIVER_OWNED_FLAGS / SPEC.md Driver 2)")


def parse_rows_to_csv(specs: list[str]) -> dict[int, str]:
    """Parse repeatable --train-csv ROWS=PATH into the default map (override/extend)."""
    mapping = dict(DEFAULT_ROWS_TO_CSV)
    for spec in specs:
        rows_s, _, path = spec.partition("=")
        if not path:
            sys.exit(f"--train-csv must be ROWS=PATH, got: {spec!r}")
        try:
            rows = int(rows_s)
        except ValueError:
            sys.exit(f"--train-csv rows must be an int, got: {rows_s!r} in {spec!r}")
        mapping[rows] = path
    return mapping


def count_features(csv_path: str) -> int | None:
    """Header column count minus the label column, or None if unreadable."""
    try:
        with open(csv_path, "r") as f:
            header = f.readline()
    except OSError:
        return None
    if not header:
        return None
    return max(0, len(header.rstrip("\n").split(",")) - 1)


def build_cells(rows_list: list[int], depths: list[int], min_examples: list[int],
                 arm_names: list[str]) -> list[dict]:
    """Full cross product of (rows, depth, min_examples, arm), in that nesting
    order. Always input_mode "csv" (the HIGGS-prefix rows->CSV map) — the
    trunk / explicit-csv / explicit-dataset forms (A7) are cells-file-only,
    see load_cells_file / resolve_cell_source."""
    cells = []
    for rows in rows_list:
        for depth in depths:
            for me in min_examples:
                for arm in arm_names:
                    cells.append({"rows": rows, "depth": depth, "min_examples": me, "arm": arm})
    return cells


def load_cells_file(path: str) -> list[dict]:
    with open(path) as f:
        cells = json.load(f)
    if not isinstance(cells, list):
        sys.exit(f"--cells-file must contain a JSON list, got {type(cells).__name__}")
    required = {"rows", "depth", "min_examples", "arm"}
    for i, cell in enumerate(cells):
        if not isinstance(cell, dict) or not required.issubset(cell):
            sys.exit(f"--cells-file entry {i} missing one of {sorted(required)}: {cell!r}")
        mode = cell.get("input_mode", "csv")
        if mode not in ("csv", "trunk"):
            sys.exit(f"--cells-file entry {i} has invalid input_mode {mode!r} "
                     f"(must be 'csv' or 'trunk'): {cell!r}")
        if mode == "trunk" and "cols" not in cell:
            sys.exit(f"--cells-file entry {i} has input_mode 'trunk' but no "
                     f"'cols' (required for trunk cells): {cell!r}")
    return cells


def resolve_cell_source(cell: dict, rows_to_csv: dict[int, str],
                         default_label_col: str) -> dict:
    """Resolve a cell's data source (A7): returns a dict with keys
    "input_mode" ("csv"|"trunk"), "dataset" (the output "dataset" column /
    skip-key component), "csv_path" (None for trunk), "label_col" (None for
    trunk), "cols" (None for csv), "rows" (echoed through, the trunk row
    count or the CSV lookup key)."""
    rows = cell["rows"]
    mode = cell.get("input_mode", "csv")
    if mode == "trunk":
        cols = cell["cols"]
        return {"input_mode": "trunk", "rows": rows, "cols": cols,
                "dataset": cell.get("dataset", f"trunk_{rows}_x_{cols}"),
                "csv_path": None, "label_col": None}
    csv_path = cell.get("csv") or rows_to_csv.get(rows)
    return {"input_mode": "csv", "rows": rows, "cols": None,
            "dataset": cell.get("dataset", f"higgs_{rows}"),
            "csv_path": csv_path,
            "label_col": cell.get("label_col", default_label_col)}


def check_dataset_source_consistency(cells: list[dict], rows_to_csv: dict[int, str],
                                     default_label_col: str) -> None:
    """Fail loudly if two cells resolve to the same output "dataset" name
    (the read_existing_keys skip-existing key component, see its docstring)
    but a different actual data source (input_mode, csv path/label_col, or
    trunk cols). Without this, a future or hand-edited cells-file entry that
    reuses a "dataset" name (or an implicit rows-derived one) while pointing
    at a different csv/cols would be silently treated as "already done" and
    skipped on a later run using the first cell's timing for the wrong
    input. Holds for the shipped cells_b1/b2/b4/b5.json today (verified no
    conflicts); this makes it a load-time error instead of a silent one for
    any future edit."""
    seen: dict[str, tuple] = {}
    for cell in cells:
        src = resolve_cell_source(cell, rows_to_csv, default_label_col)
        dataset = src["dataset"]
        fingerprint = (("trunk", src["cols"]) if src["input_mode"] == "trunk"
                       else ("csv", src["csv_path"], src["label_col"]))
        prior = seen.get(dataset)
        if prior is not None and prior != fingerprint:
            sys.exit(f"cells reference dataset {dataset!r} with two different data "
                     f"sources: {prior} vs {fingerprint} -- read_existing_keys() skips "
                     f"on 'dataset' alone, so this would silently conflate unrelated "
                     f"runs; fix the 'dataset'/'csv'/'label_col'/'cols' fields in the "
                     f"cells file(s) so each dataset name is unique to one source")
        seen[dataset] = fingerprint


def build_argv(binary_path: str, src: dict, trees: int, threads: int, seed: int,
               ydf_flags: list[str], depth: int, min_examples: int) -> list[str]:
    """Build the full harness argv for one cell, given its resolved data
    source (resolve_cell_source) and arm ydf_flags."""
    if src["input_mode"] == "trunk":
        data_args = ["--input_mode", "trunk", f"--rows={src['rows']}", f"--cols={src['cols']}"]
    else:
        data_args = ["--input_mode", "csv", "--train_csv", src["csv_path"],
                     "--label_col", src["label_col"]]
    return ([binary_path] + data_args +
            [f"--num_trees={trees}", f"--num_threads={threads}", f"--seed={seed}"] +
            list(ydf_flags) +
            [f"--tree_depth={depth}", f"--min_examples={min_examples}",
             "--compute_oob_performances=false"])


def read_existing_keys(out_path: str) -> set[tuple]:
    """(dataset, max_depth, min_examples, arm) tuples already present in --out
    with status == 'OK'. Keyed on "dataset" rather than the older "rows" (A7):
    "rows" alone no longer uniquely identifies the input once cells can point
    at an explicit non-HIGGS CSV or a trunk synthetic dataset that shares a
    numeric "rows" value with an unrelated HIGGS-prefix cell; "dataset" is
    always the actual distinguishing name written to the CSV. A failed cell
    (ERROR/OOM/TIMEOUT) is deliberately left out of the skip set so a later
    re-run retries it instead of freezing the failure in permanently."""
    keys: set[tuple] = set()
    if not os.path.exists(out_path):
        return keys
    with open(out_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "OK":
                continue
            try:
                keys.add((row["dataset"], int(row["max_depth"]),
                          int(row["min_examples"]), row["arm"]))
            except (KeyError, ValueError):
                continue
    return keys


def run_cell(argv: list[str], timeout: int, log_path: str) -> dict:
    """Run one harness invocation (argv fully built by the caller, see
    build_argv); returns a dict of the timing/status/cmd fields."""
    cmd_str = shlex.join(argv)
    result = {"train_s": "", "pre_train_s": "", "status": "OK", "cmd": cmd_str}
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        result["status"] = "TIMEOUT"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            f.write(f"CMD: {cmd_str}\n\n[TIMEOUT after {timeout}s]\n")
        return result

    out = proc.stdout + proc.stderr
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        f.write(f"CMD: {cmd_str}\nEXIT: {proc.returncode}\n\n{out}\n")

    # subprocess reports a signal-killed child as a NEGATIVE return code (e.g.
    # -9 for SIGKILL), never the shell '128+n' convention; the Linux OOM killer
    # sends SIGKILL, so check both forms (137 kept in case of a shell wrapper).
    if proc.returncode in (137, -signal.SIGKILL):
        result["status"] = "OOM"
        return result
    if proc.returncode != 0:
        result["status"] = "ERROR"
        return result

    time_hits = TIME_RE.findall(out)
    if not time_hits:
        result["status"] = "ERROR"
        return result
    result["train_s"] = float(time_hits[-1])
    pre_hits = PRE_RE.findall(out)
    if pre_hits:
        result["pre_train_s"] = float(pre_hits[-1])
    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Driver 2: YDF-fork-only speedup-map grid runner (timing only).")
    ap.add_argument("--train-csv", action="append", default=[], metavar="ROWS=PATH",
                     help="repeatable; override/extend the default HIGGS-prefix rows->CSV map")
    ap.add_argument("--label-col", default=DEFAULT_LABEL_COL)
    ap.add_argument("--depths", default="6,10,15,20,-1",
                     help="comma list of --tree_depth values (ignored if --cells-file given)")
    ap.add_argument("--min-examples", default="1",
                     help="comma list of --min_examples values (ignored if --cells-file given)")
    ap.add_argument("--rows", default="",
                     help="comma list of row-prefix keys to sweep (default: all keys in the "
                          "rows->CSV map, ascending); ignored if --cells-file given")
    ap.add_argument("--arms", default="",
                     help="comma list of arm names (default: all ydf_fork arms); ignored if "
                          "--cells-file given")
    ap.add_argument("--cells-file", default="",
                     help="JSON list of cells (e.g. cells_b1.json / cells_b2.json / "
                          "cells_b4.json / cells_b5.json) — overrides --depths/--min-examples/"
                          "--rows/--arms and defines the exact cell set and its order. Each "
                          "cell needs rows/depth/min_examples/arm, plus optionally "
                          "input_mode=csv|trunk, cols (required for trunk), csv, label_col, "
                          "dataset (see resolve_cell_source / module docstring, SPEC.md A7)")
    ap.add_argument("--trees", type=int, default=None,
                     help="override num_trees for every cell (default: each cell's own arm's "
                          "'trees' value from arms.py, e.g. 240 for RF arms, 300 for GBT arms "
                          "— SPEC.md A7)")
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bin-dir", default=DEFAULT_BIN_DIR,
                     help="dir holding the arm binaries + <name>.gitsha files")
    ap.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    ap.add_argument("--timeout", type=int, default=4 * 3600)
    ap.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--smoke", action="store_true",
                     help="run only the first cell (after any --cells-file/--rows/--arms filtering)")
    ap.add_argument("--dry-run", action="store_true",
                     help="print the argv of every cell (and the total count) without running")
    args = ap.parse_args()

    arms, used_fallback = load_arms()
    ydf_arms = {name: spec for name, spec in arms.items() if spec.get("engine") == "ydf_fork"}
    if not ydf_arms:
        sys.exit("no engine=='ydf_fork' arms found in ARMS")
    check_driver_owned_flags(ydf_arms)

    rows_to_csv = parse_rows_to_csv(args.train_csv)

    if args.cells_file:
        cells = load_cells_file(args.cells_file)
    else:
        depths = [int(x) for x in args.depths.split(",") if x != ""]
        min_examples = [int(x) for x in args.min_examples.split(",") if x != ""]
        if args.rows:
            rows_list = [int(x) for x in args.rows.split(",") if x != ""]
        else:
            rows_list = sorted(rows_to_csv.keys())
        arm_names = [a for a in args.arms.split(",") if a] if args.arms else sorted(ydf_arms)
        cells = build_cells(rows_list, depths, min_examples, arm_names)

    check_dataset_source_consistency(cells, rows_to_csv, args.label_col)

    unknown_arms = sorted({c["arm"] for c in cells} - set(ydf_arms))
    if unknown_arms:
        sys.exit(f"cells reference arms not in ARMS (or not engine=='ydf_fork'): {unknown_arms}")
    # rows->CSV coverage only matters for input_mode "csv" cells that don't
    # already give their own explicit "csv" path (trunk cells generate data
    # in-process; explicit-"csv" cells, e.g. cells_b4.json's GiveMeSomeCredit
    # entries, don't consult this map at all).
    missing_rows = sorted({c["rows"] for c in cells
                           if c.get("input_mode", "csv") == "csv" and not c.get("csv")}
                          - set(rows_to_csv))
    if missing_rows:
        sys.exit(f"no --train-csv mapping for rows values: {missing_rows} "
                 f"(known: {sorted(rows_to_csv)})")

    # Per-cell trees (A7): each cell uses its own arm's arms.py 'trees' value
    # unless --trees is given, in which case that value is used for every
    # cell as a deliberate global override. Since each cell always draws from
    # exactly one arm's own number, no cross-cell mismatch is possible when
    # --trees is omitted; when --trees IS given, note (non-fatally) which
    # arms it overrides so the operator can see what happened.
    for arm_name in {c["arm"] for c in cells}:
        if args.trees is None and ydf_arms[arm_name].get("trees") is None:
            sys.exit(f"arm {arm_name!r} has no 'trees' in arms.py and --trees was not "
                     f"given; pass --trees to set it explicitly for this run")
    if args.trees is not None:
        overridden = sorted({(c["arm"], ydf_arms[c["arm"]].get("trees")) for c in cells
                             if ydf_arms[c["arm"]].get("trees") not in (None, args.trees)})
        if overridden:
            print(f"[run_speedup_map] NOTE: --trees={args.trees} overrides arms.py's own "
                  f"'trees' default for: {overridden} (an intentional global override, "
                  f"not a per-cell mismatch -- SPEC.md A7)", file=sys.stderr)

    if args.smoke:
        cells = cells[:1]

    if args.dry_run:
        for i, cell in enumerate(cells, 1):
            arm = ydf_arms[cell["arm"]]
            binary_name = arm["binary"]
            binary_path = os.path.join(args.bin_dir, binary_name) if binary_name else "<none>"
            src = resolve_cell_source(cell, rows_to_csv, args.label_col)
            trees = args.trees if args.trees is not None else arm["trees"]
            argv = build_argv(binary_path, src, trees, args.threads, args.seed,
                              arm["ydf_flags"], cell["depth"], cell["min_examples"])
            print(f"[{i}/{len(cells)}] {shlex.join(argv)}")
        print(f"TOTAL_CELLS={len(cells)}")
        return 0

    if used_fallback:
        print("[run_speedup_map] running with fallback ARMS (arms.py not available)",
              file=sys.stderr)

    existing = read_existing_keys(args.out) if args.skip_existing else set()
    out_exists = os.path.exists(args.out)
    fieldnames = ["dataset", "rows", "features", "arm", "family", "binary", "trees",
                  "max_depth", "min_examples", "threads", "seed", "train_s", "pre_train_s",
                  "status", "timestamp_utc", "cmd"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out_f = open(args.out, "a", newline="")
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    if not out_exists or os.path.getsize(args.out) == 0:
        writer.writeheader()
        out_f.flush()

    feature_cache: dict[str, int | None] = {}
    n = len(cells)
    n_run = 0
    n_failed = 0
    for i, cell in enumerate(cells, 1):
        depth, min_ex, arm_name = cell["depth"], cell["min_examples"], cell["arm"]
        arm = ydf_arms[arm_name]
        binary_name = arm["binary"]
        trees = args.trees if args.trees is not None else arm["trees"]
        src = resolve_cell_source(cell, rows_to_csv, args.label_col)
        dataset = src["dataset"]
        key = (dataset, depth, min_ex, arm_name)

        if key in existing:
            print(f"[{i}/{n}] {dataset} depth={depth} min_ex={min_ex} {arm_name} -> SKIP (exists)",
                  flush=True)
            continue
        if not binary_name:
            print(f"[{i}/{n}] {dataset} depth={depth} min_ex={min_ex} {arm_name} -> "
                  f"SKIP (no binary for this arm)", flush=True)
            continue
        binary_path = os.path.join(args.bin_dir, binary_name)
        if not os.path.isfile(binary_path):
            print(f"[{i}/{n}] {dataset} depth={depth} min_ex={min_ex} {arm_name} -> "
                  f"ERROR (binary not found: {binary_path})", flush=True)
            row = {"dataset": dataset, "rows": src["rows"], "features": "", "arm": arm_name,
                   "family": arm.get("family", ""), "binary": binary_name,
                   "trees": trees, "max_depth": depth, "min_examples": min_ex,
                   "threads": args.threads, "seed": args.seed, "train_s": "",
                   "pre_train_s": "", "status": "ERROR",
                   "timestamp_utc": datetime.now(timezone.utc).isoformat(), "cmd": ""}
            writer.writerow(row)
            out_f.flush()
            n_run += 1
            n_failed += 1
            continue
        if src["input_mode"] == "csv" and not src["csv_path"]:
            print(f"[{i}/{n}] {dataset} depth={depth} min_ex={min_ex} {arm_name} -> "
                  f"ERROR (no CSV path resolved for this cell)", flush=True)
            row = {"dataset": dataset, "rows": src["rows"], "features": "", "arm": arm_name,
                   "family": arm.get("family", ""), "binary": binary_name,
                   "trees": trees, "max_depth": depth, "min_examples": min_ex,
                   "threads": args.threads, "seed": args.seed, "train_s": "",
                   "pre_train_s": "", "status": "ERROR",
                   "timestamp_utc": datetime.now(timezone.utc).isoformat(), "cmd": ""}
            writer.writerow(row)
            out_f.flush()
            n_run += 1
            n_failed += 1
            continue

        if src["input_mode"] == "trunk":
            features = src["cols"]
        else:
            if src["csv_path"] not in feature_cache:
                feature_cache[src["csv_path"]] = count_features(src["csv_path"])
            features = feature_cache[src["csv_path"]]

        argv = build_argv(binary_path, src, trees, args.threads, args.seed,
                          arm["ydf_flags"], depth, min_ex)
        log_path = os.path.join(args.log_dir, f"{dataset}_d{depth}_m{min_ex}_{arm_name}.log")
        t0 = time.time()
        result = run_cell(argv, args.timeout, log_path)
        wall = time.time() - t0

        row = {"dataset": dataset, "rows": src["rows"],
               "features": features if features is not None else "",
               "arm": arm_name, "family": arm.get("family", ""), "binary": binary_name,
               "trees": trees, "max_depth": depth, "min_examples": min_ex,
               "threads": args.threads, "seed": args.seed,
               "train_s": result["train_s"], "pre_train_s": result["pre_train_s"],
               "status": result["status"],
               "timestamp_utc": datetime.now(timezone.utc).isoformat(), "cmd": result["cmd"]}
        writer.writerow(row)
        out_f.flush()
        n_run += 1
        if row["status"] != "OK":
            n_failed += 1

        train_disp = row["train_s"] if row["train_s"] != "" else result["status"]
        print(f"[{i}/{n}] {dataset} depth={depth} min_ex={min_ex} {arm_name} -> "
              f"train_s={train_disp} ({wall:.1f}s wall)", flush=True)

    out_f.close()
    print(f"CSV: {args.out} ({n_run} rows run this session, {n_failed} failed, "
          f"{len(existing)} pre-existing)")
    # A non-OK cell (ERROR/OOM/TIMEOUT/missing binary) must not be reported as
    # success: run_all.sh gates its <STAGE>_DONE sentinel on this exit code, and
    # read_existing_keys() above already excludes non-OK rows from skip-existing,
    # so a non-zero exit here is exactly what makes a failed cell self-heal on
    # the next invocation instead of being silently frozen in as "done".
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
