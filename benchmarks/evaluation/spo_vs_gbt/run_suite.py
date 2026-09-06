#!/usr/bin/env python3
"""Driver 1 (SPEC.md): CV suite + large single-split accuracy/timing runner.

Protocol (PROTOCOL.md): compare SPO-RF/SPO-GBT (this YDF fork) against
axis-aligned YDF and XGBoost/LightGBM/CatBoost on (a) the TabArena+TabReD
"suite" (5-fold stratified CV, identical folds for every method) and (b) huge
single-split datasets (HIGGS/SUSY/Epsilon), for both accuracy and training
time. Arm definitions (family, engine, binary, flags, hyperparameters, python
constructors) live in arms.py in this directory so Driver 2
(run_speedup_map.py) can reuse the ydf_fork ones.

Two sub-modes, one script (SPEC.md "Driver 1"):
  --mode suite  StratifiedKFold(n_splits=--folds, shuffle=True,
                random_state=--fold-seed) per dataset under one or more
                --datasets-dir dirs (each <dir>/<name>/train.csv + meta.json),
                fold CSVs cached under --work-dir and reused across runs.
  --mode large  --dataset NAME=train.csv:test.csv:label_col (repeatable);
                fold is always 0, no splitting.

For ydf_fork arms, one harness subprocess (examples/train_oblique_forest.cc)
is run per (dataset, fold, rep, arm); for xgboost/lightgbm/catboost arms, the
model is fit in-process on the exact same fold CSV bytes (pandas, dtype
float32) so both engines see identical data. Every result row is appended to
--out immediately (flushed), so the run is resumable: a (dataset, fold, rep,
method) key already present in --out is skipped by default.

Examples:
  python3 run_suite.py --mode suite \\
      --datasets-dir /path/tabarena_binary_csv --datasets-dir /path/tabred_binary_csv \\
      --folds 5 --fold-seed 0 --work-dir /home/ubuntu/spo_vs_gbt/folds \\
      --out /home/ubuntu/spo_vs_gbt/results/suite_results.csv --threads 48

  python3 run_suite.py --mode large \\
      --dataset "HIGGS=/path/HIGGS_train_10500k.csv:/path/HIGGS_test_500k.csv:class" \\
      --out /home/ubuntu/spo_vs_gbt/results/large_results.csv --threads 48

  python3 run_suite.py --mode suite --datasets-dir /path/tabarena_binary_csv \\
      --only diabetes --arms spo_rf_exact_hwy,aa_rf_exact --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import platform
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
from arms import ARMS, ARM_ORDER  # noqa: E402  (needs sys.path tweak above)

DEFAULT_BIN_DIR = "/home/ubuntu/spo_vs_gbt/bin"
DEFAULT_WORK_DIR = "/home/ubuntu/spo_vs_gbt/folds"
DEFAULT_LOG_DIR = "/home/ubuntu/spo_vs_gbt/logs/runs"

TIME_RE = re.compile(r"Training block took:\s*([0-9.eE+-]+)")
PRE_RE = re.compile(r"Loading/Init \(pre-train\):\s*([0-9.eE+-]+)s")
POST_RE = re.compile(r"Training \+ Post-Processing:\s*([0-9.eE+-]+)s")
TEST_RE = re.compile(
    r"test-accuracy:([0-9.eE+-]+)\s+test-auc:(-?[0-9.eE+-]+)\s+test-logloss:([0-9.eE+-]+)")
# A2 / PROTOCOL.md D1: train_post_s ("Training + Post-Processing") is the
# headline time only if the run actually took the single-pass CSV loader --
# this is the loader's own completion marker, asserted below.
SINGLE_PASS_MARKER = "Single-pass CSV load complete"

FIELDNAMES = [
    "dataset", "suite", "rows", "features", "n_categorical_encoded",
    "nan_cells_imputed", "fold", "rep", "method", "family", "engine",
    "engine_version", "binary", "trees", "max_depth", "min_examples",
    "threads", "seed", "train_s", "pre_train_s", "train_post_s", "test_acc",
    "test_auc", "test_logloss", "train_rows", "test_rows", "status",
    "timestamp_utc", "cmd",
]

# --------------------------------------------------------------------------
# Dataset discovery / fold generation
# --------------------------------------------------------------------------


def discover_datasets(dirs: list[str]) -> list[dict]:
    """<dir>/<name>/train.csv (+meta.json) under each dir; dedup by name."""
    entries: list[dict] = []
    seen: set[str] = set()
    for d in dirs:
        if not os.path.isdir(d):
            sys.exit(f"--datasets-dir not found: {d}")
        for name in sorted(os.listdir(d)):
            ds_dir = os.path.join(d, name)
            train_csv = os.path.join(ds_dir, "train.csv")
            if not os.path.isfile(train_csv):
                continue
            if name in seen:
                print(f"[run_suite] WARNING: duplicate dataset name {name!r} "
                      f"(from {d}); keeping the first one seen", file=sys.stderr)
                continue
            seen.add(name)
            meta: dict = {}
            meta_path = os.path.join(ds_dir, "meta.json")
            if os.path.isfile(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
            entries.append({
                "name": name, "train_csv": train_csv, "meta": meta,
                "label_col": meta.get("label_col", "label"),
            })
    return entries


def parse_large_datasets(specs: list[str]) -> list[dict]:
    """--dataset NAME=train.csv:test.csv:label_col, order preserved."""
    entries = []
    for spec in specs:
        name, sep, rest = spec.partition("=")
        if not sep:
            sys.exit(f"--dataset must be NAME=train.csv:test.csv:label_col, got: {spec!r}")
        parts = rest.split(":")
        if len(parts) != 3:
            sys.exit(f"--dataset must be NAME=train.csv:test.csv:label_col, got: {spec!r}")
        train_csv, test_csv, label_col = parts
        entries.append({
            "name": name, "train_csv": train_csv, "test_csv": test_csv,
            "meta": {}, "label_col": label_col,
        })
    return entries


def _fold_column_means(df: "pd.DataFrame", feature_cols: list[str]) -> "pd.Series":
    """Per-feature-column mean over non-NaN values in `df` (a train fold);
    an all-NaN column -> 0.0 (A4 / PROTOCOL.md D3)."""
    return df[feature_cols].mean(skipna=True).fillna(0.0)


def _impute_fold(df: "pd.DataFrame", feature_cols: list[str],
                  means: "pd.Series") -> tuple["pd.DataFrame", int]:
    """Fill NaN feature cells in `df` with `means` (always the TRAIN fold's
    means, even when `df` is the test fold -- A4). Returns (imputed df,
    number of cells imputed). A no-NaN df is returned unchanged (same
    object, no copy), so a dataset without train_nan.csv writes byte-identical
    fold CSVs to before A4.
    """
    sub = df[feature_cols]
    n_imputed = int(sub.isna().sum().sum())
    if n_imputed:
        df = df.copy()
        df[feature_cols] = sub.fillna(means)
    return df, n_imputed


def make_folds(dataset_name: str, train_csv: str, label_col: str, folds: int,
                fold_seed: int, work_dir: str, chrono: bool = False,
                holdout_frac: float = 0.2,
                max_folds: int | None = None) -> list[tuple[int, str, str]]:
    """Fold builder, cached under work_dir/<dataset_name>/.

    Two split protocols (both write fold<k>_train.csv / fold<k>_test.csv,
    skipped if all already exist, plus folds.json for provenance; same
    float_format="%.9g" as the source, label written as int):
      - Stratified (default): StratifiedKFold(n_splits=folds, shuffle=True,
        random_state=fold_seed).
      - Chronological holdout (`chrono`, A3 / PROTOCOL.md D2/A3): exactly
        ONE fold (k=0); fold0_train.csv = the first (1-holdout_frac) rows,
        fold0_test.csv = the rest, IN FILE ORDER (no shuffle).

    A4 / PROTOCOL.md D3: if <train_csv's dir>/train_nan.csv exists, folds are
    built from it instead of train_csv, and every NaN feature cell is
    imputed with the TRAIN fold's own column mean -- the test fold reuses
    that SAME train-fold mean, never one computed from itself; an all-NaN
    train-fold column imputes to 0.0. n_nan_cells_imputed_{train,test} are
    recorded per fold in folds.json. When train_nan.csv is absent, this is a
    no-op: behaviour (and the written CSV bytes) are unchanged from before A4.

    `max_folds` (review finding, run_suite.py --smoke path): when set, only
    the first `max_folds` fold pairs are written/cached -- the StratifiedKFold
    split itself is still computed with the real `folds` split count (so
    fold 0's rows are identical to a full run), only the amount of disk I/O
    is capped. A later full (max_folds=None) call for the same dataset/
    work_dir sees fewer cached files than it expects, treats it as a cache
    miss, and transparently backfills the rest (re-writing fold 0 byte-
    identically). Default None preserves the exact prior behaviour.
    """
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold

    ds_out_dir = os.path.join(work_dir, dataset_name)
    os.makedirs(ds_out_dir, exist_ok=True)
    folds_json = os.path.join(ds_out_dir, "folds.json")

    nan_csv = os.path.join(os.path.dirname(train_csv), "train_nan.csv")
    have_nan_source = os.path.isfile(nan_csv)
    source_csv = nan_csv if have_nan_source else train_csv

    n_folds_effective = 1 if chrono else folds
    if max_folds is not None:
        n_folds_effective = min(n_folds_effective, max_folds)
    paths = [(k, os.path.join(ds_out_dir, f"fold{k}_train.csv"),
              os.path.join(ds_out_dir, f"fold{k}_test.csv"))
             for k in range(n_folds_effective)]

    # "source" stays train_csv (unchanged field, for stable provenance across
    # A4); have_train_nan_source is its own cache-validated field so a
    # train_nan.csv that later appears/disappears next to train_csv can't
    # silently reuse a stale (unimputed or differently-imputed) cached split.
    expected = {"chrono": chrono, "source": train_csv, "label_col": label_col,
                "have_train_nan_source": have_nan_source}
    if chrono:
        expected["holdout_frac"] = holdout_frac
    else:
        expected["folds"] = folds
        expected["fold_seed"] = fold_seed

    if os.path.exists(folds_json) and all(
            os.path.isfile(tr) and os.path.isfile(te) for _, tr, te in paths):
        # Cache hit by file existence alone is not enough: a later invocation
        # with different provenance against the same --work-dir would
        # otherwise silently reuse a stale split. Validate the recorded
        # provenance matches this call before trusting the cache.
        with open(folds_json) as f:
            cached = json.load(f)
        mismatched = {k: (cached.get(k), v) for k, v in expected.items()
                      if cached.get(k) != v}
        if mismatched:
            sys.exit(
                f"{dataset_name}: cached folds at {folds_json} do not match this "
                f"invocation (cached vs requested): {mismatched}. Use a different "
                f"--work-dir, or delete {ds_out_dir} to regenerate.")
        return [(k, tr, te) for k, tr, te in paths]

    df = pd.read_csv(source_csv)
    if label_col not in df.columns:
        sys.exit(f"{dataset_name}: label_col {label_col!r} not in {source_csv} "
                  f"columns {list(df.columns)}")
    df[label_col] = df[label_col].astype(int)
    feature_cols = [c for c in df.columns if c != label_col]

    if chrono:
        n_total = len(df)
        n_train = int(round(n_total * (1 - holdout_frac)))
        split_iter = [(0, df.index[:n_train].to_numpy(), df.index[n_train:].to_numpy())]
    else:
        y = df[label_col]
        # n_splits stays the real `folds` (not n_folds_effective) so fold k's
        # rows are byte-identical to a full, non-truncated run; max_folds only
        # limits how many of the yielded splits get written below.
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=fold_seed)
        split_iter = [(k, train_idx, test_idx)
                      for k, (train_idx, test_idx) in enumerate(skf.split(df, y))
                      if k < n_folds_effective]

    indices = {}
    for k, train_idx, test_idx in split_iter:
        _, train_path, test_path = paths[k]
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        means = _fold_column_means(train_df, feature_cols)
        train_df, n_imp_train = _impute_fold(train_df, feature_cols, means)
        test_df, n_imp_test = _impute_fold(test_df, feature_cols, means)

        train_df.to_csv(train_path, index=False, float_format="%.9g")
        test_df.to_csv(test_path, index=False, float_format="%.9g")

        record = {"n_nan_cells_imputed_train": n_imp_train,
                  "n_nan_cells_imputed_test": n_imp_test}
        if chrono:
            record["train_range"] = [int(train_idx[0]), int(train_idx[-1]) + 1] \
                if len(train_idx) else [0, 0]
            record["test_range"] = [int(test_idx[0]), int(test_idx[-1]) + 1] \
                if len(test_idx) else [n_total, n_total]
        else:
            record["train_idx"] = train_idx.tolist()
            record["test_idx"] = test_idx.tolist()
        indices[str(k)] = record

    meta_out = dict(expected)
    meta_out["indices"] = indices
    with open(folds_json, "w") as f:
        json.dump(meta_out, f)
    return [(k, tr, te) for k, tr, te in paths]


def fold_nan_cells_imputed(work_dir: str, dataset_name: str, fold_idx: int) -> int | None:
    """Per-fold nan_cells_imputed (train+test), read back from folds.json's
    per-fold n_nan_cells_imputed_train/test (A4, written by make_folds()).

    Review finding: the results row used to source nan_cells_imputed from
    meta.json's static, whole-dataset value -- correct in total (train+test
    always summed to it, verified across datasets including an all-NaN-column
    one) but not actually reflecting the per-fold train-mean imputation A4
    performs, and silently wrong if a dataset's encoding-vs-imputation counts
    ever diverged. This reads the number make_folds() itself computed for
    THIS fold instead. Returns None when folds.json (or this fold's record)
    isn't there -- e.g. --mode large (no folds.json at all), or a --dry-run
    job (make_folds() is now skipped there, see build_jobs) -- so callers can
    fall back to the meta.json value.
    """
    path = os.path.join(work_dir, dataset_name, "folds.json")
    try:
        with open(path) as f:
            data = json.load(f)
        rec = data["indices"][str(fold_idx)]
        return int(rec["n_nan_cells_imputed_train"]) + int(rec["n_nan_cells_imputed_test"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def csv_header_and_rowcount(path: str) -> tuple[int, int]:
    """(row_count, feature_count) via one plain-text pass; feature_count = cols-1.

    Plain newline counting, not a CSV-quote-aware parse: relies on this
    project's fully-numeric-features scope (CLAUDE.md / SPEC.md) so no field
    ever embeds a literal newline inside quotes. Deliberately not switched to
    csv.reader: these files include HIGGS-scale (10.5M-row) train CSVs, where
    a quote-aware parse would noticeably slow down every job's setup.
    """
    rows = 0
    header = ""
    with open(path) as f:
        header = f.readline()
        for _ in f:
            rows += 1
    features = max(0, len(header.rstrip("\n").split(",")) - 1) if header else 0
    return rows, features


# --------------------------------------------------------------------------
# Per-arm execution
# --------------------------------------------------------------------------


def resolve_min_examples(arm: dict) -> int:
    """Mirrors the C++ harness's own fallback (examples/train_oblique_forest.cc,
    "min_examples 1 for Bagging and 5 for Boosting unless --min_examples >= 1
    is passed", SPEC.md "YDF harness" section) so the driver always passes an
    explicit --min_examples and never depends on the harness computing it
    itself. Keep these two numbers (1, 5) in sync with the harness by hand if
    its fallback ever changes.
    """
    if arm["min_examples"] is not None:
        return arm["min_examples"]
    return 1 if arm["family"] == "rf" else 5


def build_ydf_argv(arm: dict, train_csv: str, test_csv: str, label_col: str,
                    threads: int, seed: int, bin_dir: str,
                    trees_override: int | None = None) -> tuple[list[str], str, int, int]:
    """Returns (argv, binary_path, resolved_min_examples, resolved_trees)."""
    binary_path = os.path.join(bin_dir, arm["binary"])
    min_ex = resolve_min_examples(arm)
    trees = trees_override if trees_override is not None else arm["trees"]
    argv = ([binary_path, "--input_mode", "csv", "--train_csv", train_csv,
             "--label_col", label_col, "--test_csv", test_csv,
             f"--num_trees={trees}", f"--tree_depth={arm['tree_depth']}",
             f"--min_examples={min_ex}", f"--num_threads={threads}",
             f"--seed={seed}"] + list(arm["ydf_flags"])
            + ["--compute_oob_performances=false"])
    return argv, binary_path, min_ex, trees


def read_gitsha(bin_dir: str, binary_name: str) -> str:
    path = os.path.join(bin_dir, f"{binary_name}.gitsha")
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def run_ydf(arm: dict, train_csv: str, test_csv: str, label_col: str, threads: int,
            seed: int, bin_dir: str, timeout: int, log_path: str,
            trees_override: int | None = None) -> dict:
    argv, binary_path, min_ex, trees = build_ydf_argv(
        arm, train_csv, test_csv, label_col, threads, seed, bin_dir, trees_override)
    cmd_str = shlex.join(argv)
    result = {
        "engine_version": read_gitsha(bin_dir, arm["binary"]), "binary": arm["binary"],
        "min_examples": min_ex, "trees": trees, "train_s": "", "pre_train_s": "",
        "train_post_s": "", "test_acc": "", "test_auc": "", "test_logloss": "",
        "status": "OK", "cmd": cmd_str,
    }
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if not os.path.isfile(binary_path):
        result["status"] = "ERROR"
        with open(log_path, "w") as f:
            f.write(f"CMD: {cmd_str}\n\n[binary not found: {binary_path}]\n")
        return result

    # A5: start the harness in its own process group (start_new_session=True)
    # so a timeout can kill the whole group, not just the direct child --
    # belt-and-suspenders in case the binary ever spawns helper processes;
    # today's harness doesn't, so this is a no-op in the common case.
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()  # reap so proc.returncode is populated
        result["status"] = "TIMEOUT"
        with open(log_path, "w") as f:
            f.write(f"CMD: {cmd_str}\n\n[TIMEOUT after {timeout}s]\n")
        return result

    out = stdout + stderr
    with open(log_path, "w") as f:
        f.write(f"CMD: {cmd_str}\nEXIT: {proc.returncode}\n\n{out}\n")

    if proc.returncode in (137, -9):
        # subprocess with argv a list (shell=False) reports a signal-killed
        # child as a NEGATIVE returncode (-9 for SIGKILL), not the shell's
        # 128+9=137 convention. The OOM-killer sends SIGKILL, so -9 is what
        # actually shows up here; 137 is kept as a defensive fallback in case
        # something upstream ever routes through a shell.
        result["status"] = "OOM"
        return result
    if proc.returncode != 0:
        result["status"] = "ERROR"
        return result

    time_hits = TIME_RE.findall(out)
    if not time_hits:
        result["status"] = "ERROR"
        return result
    result["train_s"] = round(float(time_hits[-1]), 6)

    pre_hits = PRE_RE.findall(out)
    if pre_hits:
        result["pre_train_s"] = round(float(pre_hits[-1]), 6)
    post_hits = POST_RE.findall(out)
    if post_hits:
        result["train_post_s"] = round(float(post_hits[-1]), 6)

    # Non-fatal per the harness's own contract: training already succeeded,
    # so a missing/failed evaluation just leaves the test_* cells blank.
    test_hits = TEST_RE.findall(out)
    if test_hits:
        acc, auc, ll = test_hits[-1]
        result["test_acc"] = round(float(acc), 6)
        result["test_auc"] = round(float(auc), 6)
        result["test_logloss"] = round(float(ll), 6)

    # A2 / D1: train_post_s is only the honest analogue of python's .fit()
    # region (excluding CSV parse) if the single-pass loader actually ran.
    # The run still succeeded and is still recorded -- just flagged.
    if SINGLE_PASS_MARKER not in out:
        result["status"] = "SLOWPATH"
    return result


def get_engine_version(engine: str) -> str:
    try:
        if engine == "xgboost":
            import xgboost
            return xgboost.__version__
        if engine == "lightgbm":
            import lightgbm
            return lightgbm.__version__
        if engine == "catboost":
            import catboost
            return catboost.__version__
    except Exception:  # noqa: BLE001 - version lookup must never crash the run
        return ""
    return ""


PY_WARMUP_ROWS = 500  # A6 / PROTOCOL.md D11: untimed warm-up fit slice size


def _apply_trees_override(model: object, trees_override: int | None) -> None:
    """Review finding (run_all.sh comment): run_suite.py had no way to
    override an arm's tree count, so SPEC.md A8's "smoke stage, 8 trees"
    can't literally be satisfied for python arms (arms.py's py_ctor bakes
    n_estimators in per-library). Rather than change the locked py_ctor(threads,
    n_features, seed) signature (depended on by both call sites below and
    documented in arms.py), set it via sklearn's standard set_params() on the
    already-constructed estimator. Silently a no-op if trees_override is None
    (the default, every existing invocation).

    CatBoost special case: arms.py's _catboost_ctor constructs with
    `iterations=300`, and CatBoostClassifier.set_params() raises
    ("only one of the parameters iterations, n_estimators, ... should be
    initialized") if `n_estimators` is set afterwards while `iterations` is
    already populated -- verified live. get_params() tells us which alias the
    constructor actually used ('iterations' in params, non-None, for
    CatBoost; XGBClassifier/XGBRFClassifier/LGBMClassifier all key it as
    'n_estimators'), so use that one back.
    """
    if trees_override is None:
        return
    if model.get_params().get("iterations") is not None:
        model.set_params(iterations=trees_override)
    else:
        model.set_params(n_estimators=trees_override)


def _python_fit_eval(arm_name: str, arm: dict, train_csv: str, test_csv: str,
                      label_col: str, threads: int, seed: int,
                      trees_override: int | None = None) -> dict:
    """The actual fit+eval body. Runs inside the isolated child process
    started by run_python() (see there for why); kept as a separate,
    ordinarily-callable function so it stays independently testable.
    """
    cmd_repr = f"{arm_name}: {arm['engine']}(threads={threads})"
    trees_val = trees_override if trees_override is not None else arm["trees"]
    result = {
        "engine_version": "", "binary": "", "min_examples": arm["min_examples"],
        "trees": trees_val, "train_s": "", "pre_train_s": "", "train_post_s": "",
        "test_acc": "", "test_auc": "", "test_logloss": "", "status": "OK", "cmd": cmd_repr,
    }
    warmup_note = ""
    try:
        import numpy as np
        import pandas as pd
        from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

        result["engine_version"] = get_engine_version(arm["engine"])
        train_df = pd.read_csv(train_csv, dtype="float32")
        test_df = pd.read_csv(test_csv, dtype="float32")
        y_train = train_df.pop(label_col).astype(int).to_numpy()
        x_train = train_df.to_numpy(dtype=np.float32)
        y_test = test_df.pop(label_col).astype(int).to_numpy()
        x_test = test_df.to_numpy(dtype=np.float32)
        n_features = x_train.shape[1]

        # A6 / D11: one untimed warm-up fit on a small slice of this SAME
        # fold's training data, inside this spawned child, before the timed
        # fit -- pays for OpenMP pool spin-up / code-page faults on the
        # child's clock instead of the measured region. Best-effort: a
        # warm-up failure (e.g. a single-class slice some library rejects)
        # must not fail the real, timed run below.
        try:
            n_warm = min(PY_WARMUP_ROWS, x_train.shape[0])
            warm_model = arm["py_ctor"](threads, n_features, seed)
            _apply_trees_override(warm_model, trees_override)
            warm_model.fit(x_train[:n_warm], y_train[:n_warm])
            del warm_model
        except Exception as warm_exc:  # noqa: BLE001 - warm-up is best-effort
            warmup_note = (f"[warm-up FAILED, continuing] "
                          f"{type(warm_exc).__name__}: {warm_exc}\n")

        model = arm["py_ctor"](threads, n_features, seed)
        _apply_trees_override(model, trees_override)
        result["cmd"] = " ".join(repr(model).split())  # single line: repr() wraps long ones

        t0 = time.perf_counter()
        model.fit(x_train, y_train)
        result["train_s"] = round(time.perf_counter() - t0, 6)

        proba = model.predict_proba(x_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        result["test_acc"] = round(float(accuracy_score(y_test, pred)), 6)
        try:
            result["test_auc"] = round(float(roc_auc_score(y_test, proba)), 6)
        except ValueError:
            result["test_auc"] = -1.0  # single-class test fold
        result["test_logloss"] = round(float(log_loss(y_test, proba, labels=[0, 1])), 6)

        result["_log_text"] = (
            f"CMD: {result['cmd']}\n{warmup_note}train_s={result['train_s']}\n"
            f"test_acc={result['test_acc']} test_auc={result['test_auc']} "
            f"test_logloss={result['test_logloss']}\n")
    except Exception as exc:  # noqa: BLE001 - never crash the worker on one arm
        result["status"] = "ERROR"
        result["_log_text"] = f"CMD: {cmd_repr}\n\n{warmup_note}[EXCEPTION] {type(exc).__name__}: {exc}\n"
    return result


def _python_worker(arm_name: str, train_csv: str, test_csv: str, label_col: str,
                    threads: int, seed: int, trees_override: int | None,
                    queue: "multiprocessing.Queue") -> None:
    """Entry point for the isolated child process (module-level so it is
    picklable under the "spawn" start method)."""
    try:
        from arms import ARMS as _ARMS  # fresh re-import in the child interpreter
        result = _python_fit_eval(arm_name, _ARMS[arm_name], train_csv, test_csv,
                                  label_col, threads, seed, trees_override)
    except Exception as exc:  # noqa: BLE001 - worker-level safety net
        result = {"engine_version": "", "binary": "", "min_examples": "", "trees": "",
                  "train_s": "", "pre_train_s": "", "train_post_s": "",
                  "test_acc": "", "test_auc": "", "test_logloss": "", "status": "ERROR",
                  "cmd": f"{arm_name}(threads={threads})",
                  "_log_text": f"[WORKER EXCEPTION] {type(exc).__name__}: {exc}\n"}
    queue.put(result)


def run_python(arm_name: str, arm: dict, train_csv: str, test_csv: str, label_col: str,
               threads: int, seed: int, log_path: str, timeout: int,
               trees_override: int | None = None) -> dict:
    """Fit+eval a python arm on the same fold CSV bytes as the YDF arms.

    Runs in a freshly-spawned child process, not in-process, for two reasons
    (both flagged in review): (1) fairness -- the ydf_fork arms always pay a
    cold-process start (subprocess.run of a new binary); an in-process python
    .fit() would instead get to reuse whatever native thread pools/libraries
    the long-lived driver process already warmed up on an earlier arm, which
    is a driver-introduced timing asymmetry the two engines shouldn't have.
    (2) robustness -- a hang (no subprocess timeout previously applied here)
    or a native crash/OOM-kill inside xgboost/lightgbm/catboost's C++ core
    would otherwise take down the whole multi-hour driver process instead of
    just this one row. "spawn" (not the Linux default "fork") is used
    deliberately so the child starts a genuinely fresh interpreter/native
    library state rather than a copy-on-write clone of the already-warmed
    parent.
    """
    cmd_repr = f"{arm_name}: {arm['engine']}(threads={threads})"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    ctx = multiprocessing.get_context("spawn")
    queue: "multiprocessing.Queue" = ctx.Queue()
    trees_val = trees_override if trees_override is not None else arm["trees"]
    proc = ctx.Process(target=_python_worker,
                       args=(arm_name, train_csv, test_csv, label_col, threads, seed,
                             trees_override, queue))
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        with open(log_path, "w") as f:
            f.write(f"CMD: {cmd_repr}\n\n[TIMEOUT after {timeout}s]\n")
        return {
            "engine_version": "", "binary": "", "min_examples": arm["min_examples"],
            "trees": trees_val, "train_s": "", "pre_train_s": "", "train_post_s": "",
            "test_acc": "", "test_auc": "", "test_logloss": "", "status": "TIMEOUT",
            "cmd": cmd_repr,
        }

    result = None
    try:
        if not queue.empty():
            result = queue.get_nowait()
    except Exception:  # noqa: BLE001 - fall through to the crash path below
        result = None

    if result is None:
        # Process ended without ever putting a result: SIGKILL (OOM-killer or
        # our own kill()), a native segfault, or a spawn-time failure.
        exitcode = proc.exitcode
        status = "OOM" if exitcode in (137, -9) else "ERROR"
        with open(log_path, "w") as f:
            f.write(f"CMD: {cmd_repr}\n\n[CHILD PROCESS EXITED, code={exitcode}]\n")
        return {
            "engine_version": "", "binary": "", "min_examples": arm["min_examples"],
            "trees": trees_val, "train_s": "", "pre_train_s": "", "train_post_s": "",
            "test_acc": "", "test_auc": "", "test_logloss": "", "status": status,
            "cmd": cmd_repr,
        }

    with open(log_path, "w") as f:
        f.write(result.pop("_log_text", ""))
    return result


# --------------------------------------------------------------------------
# Bookkeeping: resumability + provenance
# --------------------------------------------------------------------------


# A5: a row is keyed by (dataset, fold, rep, method, seed) -- seed added so a
# --seed 2 accuracy-variance rerun (PROTOCOL.md D12b) coexists with, rather
# than collides with or is skipped in favor of, the --seed 1 row.
DONE_STATUSES = {"OK", "SLOWPATH"}


def read_existing_keys(out_path: str) -> tuple[set[tuple[str, str, str, str, str]], int]:
    """(keys of rows worth skipping, count of non-done rows found).

    status in DONE_STATUSES counts as "done": OK, and (A2) SLOWPATH -- a
    SLOWPATH row completed training and carries real numbers, it's just
    flagged as not having taken the single-pass CSV loader, so re-running it
    would (deterministically) reproduce the same flag and burn the same
    multi-hour cost for nothing. A row that previously recorded
    ERROR/OOM/TIMEOUT is not treated as existing, so a transient failure (a
    box-contention OOM, a binary that was still being built, a noisy-neighbor
    timeout -- SPEC.md itself warns binaries "may still be building while you
    write code") gets retried on the next invocation instead of becoming a
    silent, permanent hole that --skip-existing (the default) would otherwise
    never revisit. Both the old and new row simply coexist in --out (rows are
    never deleted, per SPEC.md); analysis should key off status in
    DONE_STATUSES (or status=="OK" alone, if SLOWPATH rows are to be excluded
    from the headline train_post_s comparison).
    """
    keys: set[tuple[str, str, str, str, str]] = set()
    n_failed = 0
    if not os.path.exists(out_path):
        return keys, n_failed
    with open(out_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = (row["dataset"], str(int(row["fold"])),
                      str(int(row["rep"])), row["method"], str(int(row["seed"])))
            except (KeyError, ValueError):
                continue
            if row.get("status") in DONE_STATUSES:
                keys.add(key)
            else:
                n_failed += 1
    return keys, n_failed


def write_provenance(out_path: str, argv: list[str]) -> None:
    """Writes <out>.provenance.txt -- but only if it doesn't already exist.

    SPEC.md: "write <out>.provenance.txt once". A resumable multi-day run
    (crash recovery, an incremental --arms invocation) calls this on every
    run_jobs(); truncating it each time would replace the git sha/CLI that
    actually produced the majority of --out's rows with whatever invocation
    happened to run last, which is actively misleading given this project's
    emphasis on exact git-sha provenance (engine_version=gitsha per row).
    """
    prov_path = out_path + ".provenance.txt"
    if os.path.exists(prov_path):
        return
    sha = ""
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, timeout=10)
        sha = proc.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    lines = [
        f"date_utc: {datetime.now(timezone.utc).isoformat()}",
        f"repo_git_sha: {sha}",
        f"machine: {platform.node()} ({platform.platform()})",
        f"python: {sys.version.split()[0]}",
        f"cli: {shlex.join(argv)}",
    ]
    for mod_name in ("pandas", "numpy", "sklearn", "xgboost", "lightgbm", "catboost"):
        try:
            mod = __import__(mod_name)
            lines.append(f"{mod_name}: {getattr(mod, '__version__', '?')}")
        except Exception:  # noqa: BLE001
            lines.append(f"{mod_name}: not importable")
    os.makedirs(os.path.dirname(os.path.abspath(prov_path)) or ".", exist_ok=True)
    with open(prov_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def build_jobs(entries: list[dict], args: argparse.Namespace) -> list[dict]:
    """Flatten (dataset x fold x rep x arm) into an ordered job list.

    Run order (SPEC.md): datasets as given in `entries` (already sorted by
    rows ascending for --mode suite; CLI order for --mode large); within a
    dataset, fold-major, then rep, then arm in the listed --arms order.

    A5: `rep` values run from args.rep to args.rep + args.repeats - 1, so a
    single-shot invocation with --rep 2 --repeats 1 (default) writes exactly
    one rep=2 row per (dataset, fold, arm) -- PROTOCOL.md D12(a)'s "rep 2 for
    the headline arms" -- without disturbing the default --rep 0 --repeats 1
    behaviour every earlier invocation already relies on.

    Review finding: --dry-run's own --help says "no execution", but it used
    to call make_folds() (which builds AND WRITES all --folds fold CSVs to
    --work-dir) before truncating to one fold for display -- a real,
    unadvertised side effect. --dry-run now never calls make_folds() at all:
    run_dry_run() only stringifies job["train_path"]/["test_path"] into an
    argv it prints, it never opens those files, so a placeholder fold-0 path
    (which may not exist on disk) is enough. --smoke *does* still call
    make_folds() (it really trains), but passes max_folds=1 so it only ever
    writes the one fold it uses instead of all --folds of them.
    """
    jobs: list[dict] = []
    rep_offsets = range(1) if args.dry_run else range(args.repeats)
    for entry in entries:
        if args.mode == "suite":
            chrono = entry["name"] in args.chrono_holdout_set
            entry["chrono"] = chrono  # consumed by run_jobs for the suite column (A3)
            if args.dry_run:
                ds_dir = os.path.join(args.work_dir, entry["name"])
                fold_list = [(0, os.path.join(ds_dir, "fold0_train.csv"),
                             os.path.join(ds_dir, "fold0_test.csv"))]
            else:
                fold_list = make_folds(entry["name"], entry["train_csv"], entry["label_col"],
                                       args.folds, args.fold_seed, args.work_dir,
                                       chrono=chrono, holdout_frac=args.holdout_frac,
                                       max_folds=1 if args.smoke else None)
        else:
            fold_list = [(0, entry["train_csv"], entry["test_csv"])]

        for fold_idx, train_path, test_path in fold_list:
            for rep_offset in rep_offsets:
                for arm_name in args.selected_arm_names:
                    jobs.append({
                        "entry": entry, "fold": fold_idx, "rep": args.rep + rep_offset,
                        "arm": arm_name, "train_path": train_path, "test_path": test_path,
                        "label_col": entry["label_col"],
                    })
    return jobs


def run_dry_run(jobs: list[dict], args: argparse.Namespace) -> int:
    ydf_jobs = [j for j in jobs if ARMS[j["arm"]]["engine"] == "ydf_fork"]
    n = len(ydf_jobs)
    for i, job in enumerate(ydf_jobs, 1):
        arm = ARMS[job["arm"]]
        argv, _, _, _ = build_ydf_argv(arm, job["train_path"], job["test_path"],
                                       job["label_col"], args.threads, args.seed,
                                       args.bin_dir, args.trees_override)
        print(f"[{i}/{n}] {job['entry']['name']} fold{job['fold']} {job['arm']} -> "
              f"{shlex.join(argv)}")
    print(f"TOTAL_YDF_ARM_CELLS={n}")
    return 0


def run_jobs(jobs: list[dict], args: argparse.Namespace) -> int:
    out_exists = os.path.exists(args.out) and os.path.getsize(args.out) > 0
    if out_exists:
        # Guard against silently appending new-schema rows under an
        # old-schema header if FIELDNAMES is ever changed between an earlier
        # partial run and a later resumed run on the same --out path.
        with open(args.out, newline="") as f:
            existing_header = next(csv.reader(f), [])
        if existing_header != FIELDNAMES:
            sys.exit(f"--out {args.out} has a header that doesn't match this "
                     f"script's current FIELDNAMES; refusing to append.\n"
                     f"  existing: {existing_header}\n  expected: {FIELDNAMES}")

    existing_keys, n_failed = (
        read_existing_keys(args.out) if args.skip_existing else (set(), 0))
    if n_failed:
        print(f"[run_suite] {n_failed} row(s) in {args.out} have status not in "
              f"{sorted(DONE_STATUSES)} and will be retried this run.", file=sys.stderr)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out_f = open(args.out, "a", newline="")
    writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
    if not out_exists:
        writer.writeheader()
        out_f.flush()

    write_provenance(args.out, sys.argv)

    n_pre_existing = len(existing_keys)
    fold_counts_cache: dict[tuple[str, str], tuple[int, int]] = {}
    features_cache: dict[str, int] = {}
    fold_nan_cache: dict[tuple[str, int], "int | None"] = {}
    n = len(jobs)
    n_run = 0
    for i, job in enumerate(jobs, 1):
        entry, fold_idx, rep, arm_name = job["entry"], job["fold"], job["rep"], job["arm"]
        train_path, test_path, label_col = job["train_path"], job["test_path"], job["label_col"]
        # A5: seed is part of the key so a --seed 2 rerun (D12b) coexists
        # with the --seed 1 row instead of being skipped as a duplicate.
        key = (entry["name"], str(fold_idx), str(rep), arm_name, str(args.seed))

        if args.skip_existing and key in existing_keys:
            print(f"[{i}/{n}] {entry['name']} fold{fold_idx} rep{rep} {arm_name} -> "
                  f"SKIP (exists)", flush=True)
            continue

        arm = ARMS[arm_name]
        fc_key = (train_path, test_path)
        if fc_key not in fold_counts_cache:
            fold_counts_cache[fc_key] = (csv_header_and_rowcount(train_path)[0],
                                         csv_header_and_rowcount(test_path)[0])
        train_rows, test_rows = fold_counts_cache[fc_key]

        # SPEC.md names this "dataset_fold_arm.log"; that's what it stays for
        # the default --repeats=1 --rep 0. Any invocation that can produce
        # more than one row per (dataset, fold, arm) -- --repeats>1, a
        # nonzero --rep (A5), or a non-default --seed -- would otherwise
        # silently overwrite that same log with no diagnosable trace of the
        # earlier run, so those are folded into the filename whenever they
        # differ from the single-row default.
        log_suffix = ""
        if args.repeats != 1 or args.rep != 0:
            log_suffix += f"_rep{rep}"
        if args.seed != 1:
            log_suffix += f"_seed{args.seed}"
        if args.trees_override is not None:
            log_suffix += f"_trees{args.trees_override}"
        log_name = f"{entry['name']}_{fold_idx}_{arm_name}{log_suffix}.log"
        log_path = os.path.join(args.log_dir, log_name)
        if arm["engine"] == "ydf_fork":
            result = run_ydf(arm, train_path, test_path, label_col, args.threads,
                             args.seed, args.bin_dir, args.timeout, log_path,
                             args.trees_override)
        else:
            result = run_python(arm_name, arm, train_path, test_path, label_col,
                                args.threads, args.seed, log_path, args.timeout,
                                args.trees_override)
        seed_val = args.seed

        meta = entry.get("meta", {})
        if args.mode == "suite":
            # A3 / PROTOCOL.md D2: a chronological-holdout dataset always
            # reports suite='tabred_chrono' regardless of what its meta.json
            # says, per SPEC.md's addendum wording.
            suite_val = "tabred_chrono" if entry.get("chrono") else meta.get("suite", "suite")
            rows_val = meta.get("rows", "")
            features_val = meta.get("features", "")
        else:
            suite_val = "large"
            rows_val = train_rows
            if train_path not in features_cache:
                features_cache[train_path] = csv_header_and_rowcount(train_path)[1]
            features_val = features_cache[train_path]
        enc = meta.get("features_encoding", {})

        # Prefer the per-fold count this fold's own make_folds() call
        # actually computed (A4) over meta.json's dataset-wide static number;
        # fall back to the static number when unavailable (--mode large has
        # no folds.json; --dry-run jobs skip make_folds() entirely).
        nan_cells_val: object = enc.get("nan_cells_imputed", "")
        if args.mode == "suite":
            nan_cache_key = (entry["name"], fold_idx)
            if nan_cache_key not in fold_nan_cache:
                fold_nan_cache[nan_cache_key] = fold_nan_cells_imputed(
                    args.work_dir, entry["name"], fold_idx)
            per_fold_nan = fold_nan_cache[nan_cache_key]
            if per_fold_nan is not None:
                nan_cells_val = per_fold_nan

        row = {
            "dataset": entry["name"], "suite": suite_val, "rows": rows_val,
            "features": features_val,
            "n_categorical_encoded": enc.get("n_categorical_encoded", ""),
            "nan_cells_imputed": nan_cells_val,
            "fold": fold_idx, "rep": rep, "method": arm_name,
            "family": arm["family"], "engine": arm["engine"],
            "engine_version": result["engine_version"], "binary": result["binary"],
            "trees": result["trees"], "max_depth": arm["tree_depth"],
            "min_examples": result["min_examples"], "threads": args.threads,
            "seed": seed_val, "train_s": result["train_s"],
            "pre_train_s": result["pre_train_s"], "train_post_s": result["train_post_s"],
            "test_acc": result["test_acc"], "test_auc": result["test_auc"],
            "test_logloss": result["test_logloss"], "train_rows": train_rows,
            "test_rows": test_rows, "status": result["status"],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "cmd": result["cmd"],
        }
        writer.writerow(row)
        out_f.flush()
        if row["status"] in DONE_STATUSES:
            existing_keys.add(key)
        n_run += 1

        print(f"[{i}/{n}] {entry['name']} fold{fold_idx} rep{rep} {arm_name} -> "
              f"train_s={row['train_s']} acc={row['test_acc']} auc={row['test_auc']} "
              f"status={row['status']}", flush=True)

    out_f.close()
    print(f"CSV: {args.out} ({n_run} rows run this session, "
          f"{n_pre_existing} pre-existing)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Driver 1: SPO-vs-GBT CV suite + large single-split runner.")
    ap.add_argument("--mode", choices=["suite", "large"], required=True)
    ap.add_argument("--datasets-dir", action="append", default=[],
                    help="repeatable; dir of <name>/train.csv (+meta.json) (suite mode)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold-seed", type=int, default=0)
    ap.add_argument("--work-dir", default=DEFAULT_WORK_DIR,
                    help="where fold CSVs are cached (suite mode)")
    ap.add_argument("--dataset", action="append", default=[], metavar="NAME=TRAIN:TEST:LABEL",
                    help="repeatable; large mode")
    ap.add_argument("--arms", default="",
                    help="comma list of arm names (default: all %d, SPEC.md order)"
                    % len(ARM_ORDER))
    ap.add_argument("--out", default="", help="required unless --dry-run")
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--smoke", action="store_true", help="first dataset, fold 0 only")
    ap.add_argument("--only", default="", help="comma list of dataset names to keep")
    ap.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeat the (dataset, fold) arm block this many times "
                    "(rep column runs from --rep to --rep+--repeats-1)")
    ap.add_argument("--rep", type=int, default=0,
                    help="A5: base rep index written to the rep column (and part of "
                    "the skip-existing key); e.g. --rep 2 --repeats 1 (default) records "
                    "a single rep=2 row per (dataset, fold, arm) -- PROTOCOL.md D12(a)'s "
                    "second timing rep for the headline arms, run as its own invocation "
                    "so it doesn't re-run rep 0/1")
    ap.add_argument("--bin-dir", default=DEFAULT_BIN_DIR)
    ap.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    ap.add_argument("--timeout", "--timeout-s", dest="timeout", type=int, default=4 * 3600,
                    help="per-run timeout in seconds (both flag spellings set the same "
                    "value), for both a ydf_fork arm's subprocess and a python arm's "
                    "isolated worker process; expiry records status=TIMEOUT and kills "
                    "the whole process/process-group")
    ap.add_argument("--seed", "--ydf-seed", dest="seed", type=int, default=1,
                    help="A5: single seed for BOTH engines -- the harness's --seed for "
                    "every ydf_fork arm AND random_state/random_seed for every python "
                    "arm's constructor. PROTOCOL.md's default is 1 (v1 had 1 for YDF, "
                    "hard-coded 0 for python; A5 unifies them). --ydf-seed is kept as an "
                    "alias of --seed for existing invocations; override to sweep the "
                    "seed (PROTOCOL.md D12(b) uses --seed 2).")
    ap.add_argument("--chrono-holdout", default="",
                    help="A3 / PROTOCOL.md D2/A3: comma list of --mode suite dataset "
                    "names that get ONE chronological holdout fold (first "
                    "1-holdout-frac rows train / last holdout-frac rows test, file "
                    "order, no shuffle; suite column = 'tabred_chrono') instead of "
                    "StratifiedKFold. Datasets not listed are unaffected.")
    ap.add_argument("--holdout-frac", type=float, default=0.2,
                    help="test-fraction for --chrono-holdout datasets")
    ap.add_argument("--dry-run", action="store_true",
                    help="print argv for every YDF (ydf_fork) arm on one dataset; no execution")
    ap.add_argument("--trees-override", type=int, default=None,
                    help="override every arm's tree count (--num_trees for ydf_fork, "
                    "n_estimators via set_params() for python arms) for this invocation "
                    "only; max_depth/min_examples are untouched. For fast smoke/dev "
                    "checks (e.g. SPEC.md A8's '8 trees' smoke stage) -- not for real "
                    "timing or accuracy numbers. Default: each arm's own trees value "
                    "from arms.py (no override). CAUTION: trees is NOT part of the "
                    "skip-existing key (dataset, fold, rep, method, seed) -- an "
                    "8-tree row written with --trees-override into the SAME --out a "
                    "later full run also targets will be treated as already-done and "
                    "silently skipped instead of being re-run at the real tree count. "
                    "Use a separate --out for a --trees-override invocation unless "
                    "you want it to permanently stand in for the real row.")
    args = ap.parse_args()

    if args.mode == "suite" and not args.datasets_dir:
        ap.error("--mode suite requires at least one --datasets-dir")
    if args.mode == "large" and not args.dataset:
        ap.error("--mode large requires at least one --dataset")
    if not args.dry_run and not args.out:
        ap.error("--out is required unless --dry-run")
    if args.folds < 2:
        ap.error("--folds must be >= 2 (StratifiedKFold)")
    if args.repeats < 1:
        ap.error("--repeats must be >= 1")
    if not 0.0 < args.holdout_frac < 1.0:
        ap.error("--holdout-frac must be in (0, 1)")
    if args.trees_override is not None and args.trees_override < 1:
        ap.error("--trees-override must be >= 1")
    args.chrono_holdout_set = {s.strip() for s in args.chrono_holdout.split(",") if s.strip()}
    if args.mode != "suite" and args.chrono_holdout_set:
        ap.error("--chrono-holdout only applies to --mode suite")

    if args.arms:
        selected = [a.strip() for a in args.arms.split(",") if a.strip()]
        unknown = [a for a in selected if a not in ARMS]
        if unknown:
            ap.error(f"unknown arm name(s): {unknown}; known: {sorted(ARMS)}")
        args.selected_arm_names = selected
    else:
        args.selected_arm_names = list(ARM_ORDER)

    if args.mode == "suite":
        entries = discover_datasets(args.datasets_dir)
    else:
        entries = parse_large_datasets(args.dataset)
    if not entries:
        sys.exit("no datasets found")

    if args.only:
        only_set = {s.strip() for s in args.only.split(",") if s.strip()}
        entries = [e for e in entries if e["name"] in only_set]
        if not entries:
            sys.exit(f"--only matched no datasets (wanted {sorted(only_set)})")

    if args.chrono_holdout_set:
        unknown_chrono = args.chrono_holdout_set - {e["name"] for e in entries}
        if unknown_chrono:
            print(f"[run_suite] WARNING: --chrono-holdout name(s) not found among "
                  f"discovered datasets (typo, or filtered out by --only?): "
                  f"{sorted(unknown_chrono)}", file=sys.stderr)

    if args.mode == "suite":
        missing_rows = [e["name"] for e in entries if not e["meta"].get("rows")]
        if missing_rows:
            print(f"[run_suite] WARNING: meta.json missing/lacks 'rows' for "
                  f"{sorted(missing_rows)}; sorting them to the FRONT (rows=0), "
                  f"which defeats the ascending-size run order.", file=sys.stderr)
        entries.sort(key=lambda e: int(e["meta"].get("rows", 0) or 0))

    if args.smoke or args.dry_run:
        entries = entries[:1]

    print(f"{len(entries)} dataset(s) x {len(args.selected_arm_names)} arm(s), "
          f"mode={args.mode}, threads={args.threads}"
          + (", DRY RUN" if args.dry_run else ""), flush=True)

    jobs = build_jobs(entries, args)
    if args.dry_run:
        return run_dry_run(jobs, args)
    return run_jobs(jobs, args)


if __name__ == "__main__":
    sys.exit(main())
