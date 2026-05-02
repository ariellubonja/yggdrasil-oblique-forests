#!/usr/bin/env python3
"""eval_ab_e2e.py — A/B end-to-end comparison of two YDF compile configs.

Per variant (A = vanilla baseline by default, B = the experiment):
  1. Opt build (`-c opt --cxxopt=-O3 --cxxopt=-march=native [+config]`).
  2. Trunk e2e run (30 trees, all cores, no OOB)   -> training wall time.
  3. Same opt binary on the natural datasets for the chosen size, no OOB
                                                   -> training wall time.
  4. Same opt binary on a fixed set of CC18 datasets, with
     `--compute_oob_performances=true`             -> training wall time
                                                      + OOB accuracy.
     CC18 is the only place we measure accuracy: those datasets are small
     (run in seconds) and have a non-trivial signal-to-noise on a 30-tree
     forest, while big synthetic/natural datasets either run too long with
     OOB on or are too easy to discriminate variants.

Result: side-by-side summary + raw JSON written to
`benchmarks/results/eval_ab/<timestamp>_<variant_b>_<size>/`.

This script is **e2e only** — the binary it runs is the same one you would
deploy (no `-DCHRONO_ENABLED` overhead). For per-function insight (which
phase changed and by how much per depth), run `parallel_chrono.py`
directly: it accepts the same `--bazel_config=NAME` flag for A/B.

Modes (CC18 accuracy stage runs in both):
  --size=quick   trunk 100_000  × 4096   + epsilon            (fast turnaround)
  --size=full    trunk 3_000_000 × 4096   + epsilon, SUSY, HIGGS (full)

Typical use (vanilla baseline vs. an experiment flag):
  python3 benchmarks/evaluation/e2e_a-b_test.py \\
      --variant_b=loop_swap \\
      --bazel_config_b=projeval_loop_swap \\
      --size=quick

Comparing two non-default variants (e.g. V1 vs V2):
  python3 benchmarks/evaluation/e2e_a-b_test.py \\
      --variant_a=v1 --bazel_config_a=nodewise_proj_matrix \\
      --variant_b=v2 --bazel_config_b=depthwise_1_pass \\
      --size=full
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

# Reuse the existing CPU-toggle + cleanup helpers used by parallel_chrono.py.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import benchmarks.utils.utils as utils  # noqa: E402
TARGET = "//examples:train_oblique_forest"
BIN = REPO / "bazel-bin/examples/train_oblique_forest"

SIZES = {"quick": 100_000, "full": 3_000_000}

# Natural datasets used to measure accuracy alongside synthetic-trunk timing.
# Each entry is (csv_path, label_column, display_name).
# Quick mode skips HIGGS because it's 11M rows and would dominate run time;
# full mode adds it.
SUSY    = ("benchmarks/data/SUSY_with_header.csv",            "class", "SUSY")
EPSILON = ("benchmarks/data/epsilon_normalized_train.csv",    "label", "epsilon")
HIGGS   = ("benchmarks/data/HIGGS_with_header.csv",           "class", "HIGGS")

NATURAL = {
    "quick": [EPSILON],
    "full":  [EPSILON, SUSY, HIGGS],
}

# CC18 accuracy datasets — same set used by benchmarks/evaluation/runtime.sh.
# Small enough to run in seconds with --compute_oob_performances=true.
CC18 = [
    ("benchmarks/data/cc18_binary_csv/task_14952_PhishingWebsites/repeat0_fold0_sample0_train.csv",
     "Result", "task_14952_PhishingWebsites"),
    ("benchmarks/data/cc18_binary_csv/task_14965_bank-marketing/repeat0_fold0_sample0_train.csv",
     "Class",  "task_14965_bank-marketing"),
    ("benchmarks/data/cc18_binary_csv/task_29_credit-approval/repeat0_fold0_sample0_train.csv",
     "class",  "task_29_credit-approval"),
    ("benchmarks/data/cc18_binary_csv/task_167125_Internet-Advertisements/repeat0_fold0_sample0_train.csv",
     "class",  "task_167125_Internet-Advertisements"),
]

# The research binary on this branch short-circuits via "EXITING EARLY TO
# SPEED UP EXPERIMENTS!" in random_forest.cc, before either
# "train_oblique_forest wall-time" or "Final OOB metrics:" can fire.
# Parse the lines that DO get printed:
#   - "random_forest.cc Training block took: <X> s" — wall time of the loop
#     over trees, emitted just before the early exit.
#   - "Train tree N/N accuracy:<a> logloss:<l>" — per-tree OOB metrics
#     (only populated when --compute_oob_performances=true). We take the
#     last match for the final-tree accuracy.
RX_TRAIN_TIME = re.compile(
    r"random_forest\.cc Training block took:\s+([0-9.eE+-]+)\s*s")
RX_TREE_ACC = re.compile(
    r"Train tree \d+/\d+\s+accuracy:([0-9.eE+-]+)\s+logloss:([0-9.eE+-]+)")


# ----------------------------- Plumbing -----------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    p.add_argument("--variant_b",      required=True,
                   help='Display name for variant B (e.g. "loop_swap").')
    p.add_argument("--bazel_config_b", required=True,
                   help='Bazel --config=NAME for variant B.')
    p.add_argument("--variant_a",      default="baseline",
                   help='Display name for variant A (default: "baseline").')
    p.add_argument("--bazel_config_a", default=None,
                   help='Bazel --config=NAME for variant A. '
                        'Default: vanilla (no extra config).')
    p.add_argument("--size",           choices=list(SIZES), default="quick")
    p.add_argument("--num_trees",      type=int, default=30)
    return p.parse_args()


def sh(cmd):
    """Run cmd in REPO, stream stdout, exit on failure."""
    print(">>", " ".join(map(str, cmd)))
    r = subprocess.run(cmd, cwd=REPO)
    if r.returncode != 0:
        sys.exit(f"FAILED ({r.returncode}): {' '.join(map(str, cmd))}")


def capture(cmd, log_path: Path) -> str:
    """Run cmd in REPO, tee stdout+stderr to log_path, return text."""
    print(">>", " ".join(map(str, cmd)))
    r = subprocess.run(cmd, cwd=REPO,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(r.stdout)
    if r.returncode != 0:
        print(f"  binary returned {r.returncode}; see {log_path}")
    return r.stdout


def opt_build(extra_config: str | None):
    """Build train_oblique_forest with -O3 -march=native + optional --config.

    Toggles E-cores ON during the build (3× faster on the 185H), back to
    P-only afterwards so the runs that follow measure cleanly."""
    cmd = ["bazel", "build", "--ui_event_filters=-warning",
           "-c", "opt", "--cxxopt=-O3", "--cxxopt=-march=native"]
    if extra_config:
        cmd.append(f"--config={extra_config}")
    cmd.append(TARGET)

    utils.configure_cpu_for_benchmarks(False)   # all cores on -> fast build
    sh(cmd)
    utils.configure_cpu_for_benchmarks(True)    # P-only for runs


# ----------------------------- Per-variant steps -----------------------------

def e2e_trunk(variant: str, rows: int, num_trees: int,
              out_dir: Path) -> float | None:
    cmd = [str(BIN),
           "--input_mode=trunk", f"--rows={rows}", "--cols=4096",
           f"--num_trees={num_trees}", "--num_threads=-1", "--tree_depth=-1",
           "--feature_split_type=Oblique", "--numerical_split_type=Exact",
           "--compute_oob_performances=false"]
    out = capture(cmd, out_dir / f"trunk_{rows}_{variant}.log")
    m = RX_TRAIN_TIME.search(out)
    return float(m.group(1)) if m else None


def e2e_natural(variant: str, train_csv: str, label: str, name: str,
                num_trees: int, out_dir: Path) -> dict:
    """Wall time only on big natural datasets (no OOB — too slow / no signal)."""
    cmd = [str(BIN),
           "--input_mode=csv", f"--train_csv={train_csv}",
           f"--label_col={label}",
           f"--num_trees={num_trees}", "--num_threads=-1", "--tree_depth=-1",
           "--feature_split_type=Oblique", "--numerical_split_type=Exact",
           "--compute_oob_performances=false"]
    out = capture(cmd, out_dir / f"natural_{name}_{variant}.log")
    t = RX_TRAIN_TIME.search(out)
    return {
        "dataset":    name,
        "train_time": float(t.group(1)) if t else None,
    }


def e2e_cc18(variant: str, train_csv: str, label: str, name: str,
             num_trees: int, out_dir: Path) -> dict:
    """Wall + OOB accuracy on small CC18 datasets — accuracy is measured here."""
    cmd = [str(BIN),
           "--input_mode=csv", f"--train_csv={train_csv}",
           f"--label_col={label}",
           f"--num_trees={num_trees}", "--num_threads=-1", "--tree_depth=-1",
           "--feature_split_type=Oblique", "--numerical_split_type=Exact",
           "--compute_oob_performances=true"]
    out = capture(cmd, out_dir / f"cc18_{name}_{variant}.log")
    t = RX_TRAIN_TIME.search(out)
    accs = RX_TREE_ACC.findall(out)
    return {
        "dataset":      name,
        "train_time":   float(t.group(1)) if t else None,
        "oob_accuracy": float(accs[-1][0]) if accs else None,
        "oob_logloss":  float(accs[-1][1]) if accs else None,
    }


def evaluate_variant(variant: str, extra_config: str | None,
                     rows: int, num_trees: int, size: str,
                     out_dir: Path) -> dict:
    print(f"\n=========== Variant: {variant!r} "
          f"(--config={extra_config or '(none)'}) ===========")
    opt_build(extra_config)
    res = {
        "variant": variant,
        "config":  extra_config or "(none)",
        "trunk_train_time": e2e_trunk(variant, rows, num_trees, out_dir),
        "natural": [e2e_natural(variant, csv, label, name, num_trees, out_dir)
                    for csv, label, name in NATURAL[size]],
        "cc18":    [e2e_cc18(variant, csv, label, name, num_trees, out_dir)
                    for csv, label, name in CC18],
    }
    return res


# ----------------------------- Reporting -----------------------------

def fmt_pct(a, b):
    if a in (None, 0) or b is None:
        return "—"
    return f"{(b - a) / a * 100:+.2f}%"


def fmt_num(x, digits=3):
    return "—" if x is None else f"{x:.{digits}f}"


def fmt_acc_pp(a, b):
    if a is None or b is None:
        return "—"
    return f"{(b - a) * 100:+.3f}"


def write_summary(a: dict, b: dict, out_dir: Path,
                  size: str, rows: int, num_trees: int) -> str:
    L = []
    L.append(f"# eval_ab summary — {a['variant']} vs {b['variant']}\n")
    L.append(f"- Size      : `{size}` (trunk {rows:,} × 4096)")
    L.append(f"- Trees     : {num_trees}, threads: all (--num_threads=-1)")
    L.append(f"- A config  : `{a['config']}`")
    L.append(f"- B config  : `{b['config']}`")
    L.append("- Build     : `-c opt --cxxopt=-O3 --cxxopt=-march=native` "
             "(no chrono — for per-depth insight, run `parallel_chrono.py` "
             "with `--bazel_config=…`).\n")

    L.append("## Trunk: end-to-end training wall time\n")
    L.append("| Metric | A | B | Δ |")
    L.append("|---|---|---|---|")
    L.append(f"| Train wall (s) | "
             f"{fmt_num(a.get('trunk_train_time'))} | "
             f"{fmt_num(b.get('trunk_train_time'))} | "
             f"{fmt_pct(a.get('trunk_train_time'), b.get('trunk_train_time'))} |\n")

    L.append("## Natural datasets — training wall (no OOB)\n")
    L.append("| Dataset | A train (s) | B train (s) | Δ time |")
    L.append("|---|---|---|---|")
    a_nat = {r['dataset']: r for r in a.get('natural', [])}
    b_nat = {r['dataset']: r for r in b.get('natural', [])}
    for ds in sorted(set(a_nat) | set(b_nat)):
        ar = a_nat.get(ds, {}); br = b_nat.get(ds, {})
        at, bt = ar.get('train_time'), br.get('train_time')
        L.append(f"| {ds} | {fmt_num(at)} | {fmt_num(bt)} | "
                 f"{fmt_pct(at, bt)} |")
    L.append("")

    L.append("## CC18 — training wall + OOB accuracy\n")
    L.append("| Dataset | A train (s) | B train (s) | Δ time | "
             "A acc | B acc | Δ acc (pp) |")
    L.append("|---|---|---|---|---|---|---|")
    a_cc = {r['dataset']: r for r in a.get('cc18', [])}
    b_cc = {r['dataset']: r for r in b.get('cc18', [])}
    for ds in sorted(set(a_cc) | set(b_cc)):
        ar = a_cc.get(ds, {}); br = b_cc.get(ds, {})
        at, bt = ar.get('train_time'),   br.get('train_time')
        aa, ba = ar.get('oob_accuracy'), br.get('oob_accuracy')
        L.append(f"| {ds} | {fmt_num(at)} | {fmt_num(bt)} | "
                 f"{fmt_pct(at, bt)} | "
                 f"{fmt_num(aa, 4)} | {fmt_num(ba, 4)} | "
                 f"{fmt_acc_pp(aa, ba)} |")
    L.append("")

    text = "\n".join(L)
    (out_dir / "summary.md").write_text(text)
    (out_dir / "raw.json").write_text(
        json.dumps({"a": a, "b": b}, indent=2))
    return text


# ----------------------------- main -----------------------------

def main():
    args = parse_args()
    rows = SIZES[args.size]

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = (REPO / "benchmarks/results/eval_ab" /
               f"{ts}_{args.variant_b}_{args.size}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # CPU lifecycle contract:
    #   start          -> disable E-cores/HT/turbo (P-only) for clean timing.
    #   each build     -> opt_build() flips to all-on (3× faster build),
    #                     then back to P-only before runs.
    #   end (any path) -> finally{} below re-enables everything.
    # We do NOT rely on utils.cpu_modified because configure_cpu_for_benchmarks
    # short-circuits when state already matches, which can leave the flag
    # un-set if the system was pre-disabled. The finally block makes the
    # restore unconditional and self-contained.
    utils.setup_signal_handlers()  # for SIGINT/SIGTERM cleanup
    utils.configure_cpu_for_benchmarks(True)
    print("CPU configured: P-cores only, HT/E-cores/turbo disabled.")
    try:
        a_res = evaluate_variant(args.variant_a, args.bazel_config_a,
                                 rows, args.num_trees, args.size, out_dir)
        b_res = evaluate_variant(args.variant_b, args.bazel_config_b,
                                 rows, args.num_trees, args.size, out_dir)

        text = write_summary(a_res, b_res, out_dir,
                             args.size, rows, args.num_trees)
        print("\n" + text)
        print(f"\nFull artifacts: {out_dir}")
    finally:
        print("Restoring CPU configuration: re-enabling HT/E-cores/turbo.")
        utils.configure_cpu_for_benchmarks(False)


if __name__ == "__main__":
    main()
