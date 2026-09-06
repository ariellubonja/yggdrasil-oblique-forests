#!/usr/bin/env python3
"""Driver 4 — analysis + reporting for the SPO-vs-GBT study.

Reads the three CSVs the other drivers produce (see SPEC.md / PROTOCOL.md,
repo /home/ubuntu/yggdrasil-oblique-forests, run state under
/home/ubuntu/spo_vs_gbt):

  - suite_results.csv   (run_suite.py --mode suite):  15 arms x N datasets x
    5 folds (or 1 chronological holdout fold for suite='tabred_chrono'
    datasets, PROTOCOL.md D2), TabArena + TabReD.
  - large_results.csv   (run_suite.py --mode large):  15 arms x {HIGGS, SUSY,
    Epsilon}, single held-out split (fold=0), incl. D12 timing/seed repeats.
  - speedup_map.csv     (run_speedup_map.py):          YDF arms only, timing
    grid: HIGGS row-count prefixes x {max_depth, min_examples} (B1/B2), GBT
    depth cells (B4), and trunk-generator feature-width cells (B5).

The protocol this answers (PROTOCOL.md, then the "PROTOCOL v2 -- decisions"
D1-D14 after the review panel; SPEC.md's "SPEC v2 addendum" A1-A9): three
KDD'26 reviewers asked (a) whether our vectorized-histogram speedup over
exact oblique splits (SO-YDF) still matters once compared against optimized
axis-aligned learners (XGBoost/LightGBM/CatBoost, incl. their RF modes) on
accuracy *and* speed, and (b) for a size x depth x width x stopping-criterion
map of that speedup, on more/larger datasets. This script turns the three
result CSVs into:
  - per-dataset and two per-family (RF, GBT -- D4: never ranked against each
    other in a table) summary tables;
  - a D7 bit-identity report (spo_rf_exact_stdsort vs spo_rf_exact_hwy);
  - D8 statistics: AUC-primary Friedman tests + Nemenyi critical-difference
    diagrams per family, and Holm-corrected Wilcoxon signed-rank tests over
    the two FIXED comparison sets (spo_rf_dyn_vec vs {spo_rf_exact_hwy,
    aa_rf_exact, xgboost_rf, lightgbm_rf}; spo_gbt_dyn_vec vs
    {spo_gbt_exact_hwy, aa_gbt_exact, xgboost, lightgbm, catboost}) with
    win/tie/loss (tie = |dAUC| < 0.005);
  - speed: per-dataset training-time ratios (headline time = train_s, the
    "Training block took" timer -- PROTOCOL.md D1-rev) with geometric means
    that exclude rows < 2000 datasets from the headline number while still
    listing them;
  - figures (matplotlib, PDF+PNG): a D4 cross-family AUC-vs-train_s
    Pareto scatter per huge dataset (arms as consistent markers, family by
    color, frontier drawn), speedup-vs-rows (RF arms), the B1 rows x depth
    heatmap, the B2 min_examples lines, the B4 GBT-depth plot, the B5
    trunk-width plot, and two Nemenyi CD diagrams (RF/GBT, AUC);
  - booktabs LaTeX tables for the paper (make_paper_tables -- generated
    only, never hand-edited).

The 15-arm roster (names, family, engine, binary, canonical order) is
imported from arms.py in this directory rather than hardcoded here (SPEC.md
v2 addendum A9) -- see load_arms() for the fallback used while arms.py's own
addendum item (A1, a concurrent deliverable) hasn't landed the two new
RF-family arms yet.

Must run on whatever subset of the three CSVs exists so far, and on missing
individual arms/datasets/folds within them: absent data becomes NaN /
an empty output section, never a crash or a wrong-shaped table. Every
output-producing function below is independently robust to an empty input
frame.

Usage:
  analyze.py --out-dir /home/ubuntu/spo_vs_gbt/results/analysis
  analyze.py --selftest   # fabricates tiny synthetic CSVs and exercises
                           # every output; no real data or binaries needed.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.ticker import ScalarFormatter

from scipy import stats

# --------------------------------------------------------------------------
# Canonical arm table -- imported from arms.py (names/family/engine/binary/
# order), NOT hardcoded (SPEC.md v2 addendum A9). analyze.py never launches
# training; this is display metadata (family, label, plot style) plus the
# grouping keys the statistics below key off of.
# --------------------------------------------------------------------------

# Fallback, used only while arms.py hasn't landed the two new RF-family arms
# from PROTOCOL.md D6 / SPEC.md addendum A1 yet (a colleague's concurrent
# deliverable -- same defensive pattern run_speedup_map.py uses for arms.py
# itself). Mirrors arms.py's ARMS shape (family/engine/binary only; analyze.py
# never needs ydf_flags/py_ctor). Canonical SPEC.md order: RF family first
# (xgboost_rf, lightgbm_rf LAST within it), then GBT family.
_FALLBACK_ARMS: dict[str, dict] = {
    "spo_rf_exact_stdsort": {"family": "rf", "engine": "ydf_fork", "binary": "exact_std_sort"},
    "spo_rf_exact_hwy": {"family": "rf", "engine": "ydf_fork", "binary": "default"},
    "spo_rf_rand_scalar": {"family": "rf", "engine": "ydf_fork", "binary": "scalar"},
    "spo_rf_rand_vec": {"family": "rf", "engine": "ydf_fork", "binary": "default"},
    "spo_rf_dyn_scalar": {"family": "rf", "engine": "ydf_fork", "binary": "scalar"},
    "spo_rf_dyn_vec": {"family": "rf", "engine": "ydf_fork", "binary": "default"},
    "aa_rf_exact": {"family": "rf", "engine": "ydf_fork", "binary": "default"},
    "xgboost_rf": {"family": "rf", "engine": "xgboost", "binary": None},
    "lightgbm_rf": {"family": "rf", "engine": "lightgbm", "binary": None},
    "spo_gbt_exact_hwy": {"family": "gbt", "engine": "ydf_fork", "binary": "default"},
    "spo_gbt_dyn_vec": {"family": "gbt", "engine": "ydf_fork", "binary": "default"},
    "aa_gbt_exact": {"family": "gbt", "engine": "ydf_fork", "binary": "default"},
    "xgboost": {"family": "gbt", "engine": "xgboost", "binary": None},
    "lightgbm": {"family": "gbt", "engine": "lightgbm", "binary": None},
    "catboost": {"family": "gbt", "engine": "catboost", "binary": None},
}
_FALLBACK_ARM_ORDER: list[str] = list(_FALLBACK_ARMS.keys())


def load_arms() -> tuple[dict[str, dict], list[str], bool]:
    """Import ARMS/ARM_ORDER from arms.py in this directory; fall back to the
    built-in 15-arm table if it's missing, broken, or doesn't have the full
    15-arm roster yet (D6's xgboost_rf/lightgbm_rf). Returns
    (arms, arm_order, used_fallback)."""
    this_dir = str(Path(__file__).resolve().parent)
    sys.path.insert(0, this_dir)
    try:
        import arms as arms_module  # type: ignore
        imported = getattr(arms_module, "ARMS", None)
        order = getattr(arms_module, "ARM_ORDER", None)
        if not isinstance(imported, dict) or not imported or not isinstance(order, list):
            raise ImportError("arms.py has no non-empty ARMS dict / ARM_ORDER list")
        missing = [a for a in _FALLBACK_ARM_ORDER if a not in imported]
        if missing:
            raise ImportError(f"arms.py is missing {missing} (SPEC.md v2 addendum "
                               "A1/D6 arms not landed yet)")
        return imported, list(order), False
    except Exception as exc:  # noqa: BLE001 - any import failure -> fallback
        print(f"[analyze] WARNING: could not use a full 15-arm ARMS/ARM_ORDER from "
              f"{this_dir}/arms.py ({exc!r}); using the built-in 15-arm fallback "
              f"table. Family/engine/binary metadata here is display-only and "
              f"should be re-verified once arms.py lands the full roster (SPEC.md "
              f"v2 addendum A1).", file=sys.stderr)
        return _FALLBACK_ARMS, list(_FALLBACK_ARM_ORDER), True
    finally:
        if this_dir in sys.path:
            sys.path.remove(this_dir)


_ARMS_TABLE, ARM_ORDER, _USED_FALLBACK_ARMS = load_arms()
ALL_ARMS: list[str] = list(ARM_ORDER)
ARM_FAMILY: dict[str, str] = {a: _ARMS_TABLE[a]["family"] for a in ALL_ARMS}
ARM_ENGINE: dict[str, str] = {a: _ARMS_TABLE[a]["engine"] for a in ALL_ARMS}
ARM_BINARY: dict[str, str] = {a: (_ARMS_TABLE[a].get("binary") or "") for a in ALL_ARMS}
RF_ARMS: list[str] = [a for a in ALL_ARMS if ARM_FAMILY[a] == "rf"]
GBT_ARMS: list[str] = [a for a in ALL_ARMS if ARM_FAMILY[a] == "gbt"]

ARM_LABELS: dict[str, str] = {
    "spo_rf_exact_stdsort": "SPO-RF Exact (std::sort)",
    "spo_rf_exact_hwy": "SPO-RF Exact (Highway)",
    "spo_rf_rand_scalar": "SPO-RF Random (scalar)",
    "spo_rf_rand_vec": "SPO-RF Random (vec)",
    "spo_rf_dyn_scalar": "SPO-RF Dyn (scalar)",
    "spo_rf_dyn_vec": "SPO-RF Dyn-Vec [ours]",
    "aa_rf_exact": "AA-RF Exact",
    "xgboost_rf": "XGBoost RF mode",
    "lightgbm_rf": "LightGBM RF mode",
    "spo_gbt_exact_hwy": "SPO-GBT Exact",
    "spo_gbt_dyn_vec": "SPO-GBT Dyn-Vec [ours]",
    "aa_gbt_exact": "AA-GBT Exact",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}

EXACT_STDSORT = "spo_rf_exact_stdsort"  # published SO-YDF baseline
EXACT_HWY = "spo_rf_exact_hwy"
SPEED_BASELINE = "spo_rf_dyn_vec"       # denominator for all RF speed ratios
RF_BASELINE = "spo_rf_dyn_vec"          # "ours", RF family
GBT_BASELINE = "spo_gbt_dyn_vec"        # "ours", GBT family
ALPHA = 0.05
# D8: tie band for AUC (the primary metric) in the two fixed comparison sets.
AUC_TIE_BAND = 0.005
# D8: "datasets with < 2000 rows are excluded from headline speed ratios".
MIN_ROWS_FOR_HEADLINE_SPEED = 2000

# D8's fixed, small comparison sets (never the ad hoc "RF baseline vs GBT
# libs" cross-family comparison the v1 draft used -- D4 forbids ranking the
# two families against each other in a table; each set here stays within one
# family). {family: (baseline, [comparators])}.
FIXED_COMPARISONS: dict[str, tuple[str, list[str]]] = {
    "rf": (RF_BASELINE, ["spo_rf_exact_hwy", "aa_rf_exact", "xgboost_rf", "lightgbm_rf"]),
    "gbt": (GBT_BASELINE, ["spo_gbt_exact_hwy", "aa_gbt_exact", "xgboost", "lightgbm", "catboost"]),
}

SUITE_COLUMNS: list[str] = [
    "dataset", "suite", "rows", "features", "n_categorical_encoded",
    "nan_cells_imputed", "fold", "rep", "method", "family", "engine",
    "engine_version", "binary", "trees", "max_depth", "min_examples",
    "threads", "seed", "train_s", "pre_train_s", "train_post_s",
    "test_acc", "test_auc", "test_logloss", "train_rows", "test_rows",
    "status", "timestamp_utc", "cmd",
]
SPEEDUP_COLUMNS: list[str] = [
    "dataset", "rows", "features", "arm", "family", "binary", "trees",
    "max_depth", "min_examples", "threads", "seed", "train_s",
    "pre_train_s", "status", "timestamp_utc", "cmd",
]

# Okabe-Ito colorblind-safe qualitative palette (8 hues), cycled together
# with marker shapes so each arm gets one fixed (color, marker) pair that is
# reused, unchanged, across every figure in this file.
PALETTE: list[str] = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
                       "#56B4E9", "#F0E442", "#000000"]
MARKERS: list[str] = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "8", "p"]
ARM_STYLE: dict[str, tuple[str, str]] = {
    arm: (PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)])
    for i, arm in enumerate(ALL_ARMS)
}
# D4: the cross-family Pareto figure colors by family, not by arm.
FAMILY_COLOR: dict[str, str] = {"rf": PALETTE[0], "gbt": PALETTE[3]}


# ==========================================================================
# I/O — never raise: a missing/empty/partially-written CSV becomes an empty,
# correctly-typed frame so every downstream function can assume the columns
# exist.
# ==========================================================================

def load_csv(path: str | Path | None, columns: list[str]) -> pd.DataFrame:
    if not path or not Path(path).is_file() or Path(path).stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(f"failed to read {path}: {exc}")
        return pd.DataFrame(columns=columns)
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan
    numeric_cols = ["rows", "features", "n_categorical_encoded",
                     "nan_cells_imputed", "fold", "rep", "trees", "max_depth",
                     "min_examples", "threads", "seed", "train_s",
                     "pre_train_s", "train_post_s", "test_acc", "test_auc",
                     "test_logloss", "train_rows", "test_rows"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "status" in df.columns:
        # A missing/blank cell here is NOT known-good: it happens both when
        # the whole "status" column is absent from an older CSV (backfilled
        # above as all-NaN) and when a row was truncated mid-write by a
        # crash (this box wipes /tmp and has a 2h-inactivity auto-shutdown).
        # Silently mapping either onto "OK" would count an unmeasured row as
        # a completed, successful run. Use a distinct sentinel instead; every
        # downstream aggregate already filters on status == "OK", so
        # "UNKNOWN" rows (and any other non-OK status: ERROR/OOM/TIMEOUT/
        # SLOWPATH) are correctly excluded everywhere without further changes.
        df["status"] = df["status"].fillna("UNKNOWN").replace("", "UNKNOWN")
    if "test_auc" in df.columns:
        # harness contract: auc = -1 when a fold has no ROC (single class).
        df.loc[df["test_auc"] == -1, "test_auc"] = np.nan
    if "method" not in df.columns and "arm" in df.columns:
        df["method"] = df["arm"]
    if "arm" not in df.columns and "method" in df.columns:
        df["arm"] = df["method"]
    return df


def primary_run_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows from the protocol's single primary observation per
    (dataset, fold, method): --rep 0 at the method's default protocol seed
    (YDF 1, python 0 -- SPEC.md A5). D12's two huge-tier repeats are real,
    useful rows, but they are each a different *kind* of measurement, not
    another independent fold, and must never be silently pooled into a
    fold-mean or fold-paired statistic alongside the primary row:
      D12(a) reruns the SAME (dataset, fold, seed) at --rep 2 to get a
        median-of-2 timing sample -- its accuracy/AUC/logloss is a near-exact
        duplicate of the primary row's (deterministic RF/GBT at the same
        seed), so pooling it into an accuracy mean is a harmless no-op but
        pooling it into a fold-paired significance test doubles a fold that
        was never independently drawn.
      D12(b) reruns the SAME (dataset, fold) at --seed 2 for the seven YDF RF
        arms specifically to probe accuracy variance -- its accuracy *does*
        genuinely differ from the primary row, so silently averaging it into
        acc_mean/auc_mean or pairing it in cross_table moves the headline
        number based on an extra seed nobody asked to be part of the primary
        result (found in review: an un-independent extra "fold" fabricated
        this way inflates n_pairs and corrupts the D8 Wilcoxon statistic,
        worst on the huge/single-split datasets where it turns a real n=1
        into a fake n=2+).
    Older CSVs without rep/seed columns are treated as already-primary."""
    rep = pd.to_numeric(df["rep"], errors="coerce") if "rep" in df.columns else pd.Series(0, index=df.index)
    rep_ok = rep.fillna(0) == 0
    if "seed" not in df.columns:
        return rep_ok
    seed = pd.to_numeric(df["seed"], errors="coerce")
    # SPEC.md A5 unified the seed: run_suite.py --seed (default 1) drives BOTH
    # the harness --seed and every python arm's random_state, so the primary
    # seed is 1 for every engine (the v1 "python seed 0" is gone).
    seed_ok = seed.isna() | (seed == 1)
    return rep_ok & seed_ok


def add_headline_time(df: pd.DataFrame) -> pd.DataFrame:
    """PROTOCOL.md D1-rev (2026-09-04): the headline training time is train_s
    ("Training block took" for YDF arms = bootstrap + all trees + the split
    manager; the .fit() wall time for python arms). train_post_s ("Training +
    Post-Processing") is NOT used as the headline: on HIGGS RF it exceeds
    train_s by a constant ~400 s of YDF post-training finalization
    (structural variable importances + leaf indexing, single-threaded walks
    over 577M nodes) that has no counterpart inside XGBoost/LightGBM/CatBoost
    fit(). That finalization is exposed as `post_s` = train_post_s - train_s
    (NaN for python arms / rows without train_post_s) so it can be reported
    separately."""
    out = df.copy()
    out["time_s"] = pd.to_numeric(out.get("train_s", np.nan), errors="coerce")
    if "train_post_s" in out.columns:
        post = pd.to_numeric(out["train_post_s"], errors="coerce") - out["time_s"]
        out["post_s"] = post
    else:
        out["post_s"] = np.nan
    return out


# ==========================================================================
# (a) Per-dataset summary
# ==========================================================================

def per_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, method): mean+-std over folds of acc/auc/logloss (from
    the primary run only -- see primary_run_mask), median train_s (the
    headline time -- PROTOCOL.md D1-rev: "Training block took", bootstrap +
    all trees + split manager) and median_post_s (YDF's post-training
    finalization -- structural variable importances + leaf indexing, train_post_s
    - train_s; NaN for python arms, which have no such phase -- see
    add_headline_time; both medians span every OK row, including D12(a)'s
    huge-tier timing repeat, by design). Non-OK rows count toward
    n_folds_total but are excluded from the numeric aggregates."""
    cols = ["dataset", "suite", "_source", "method", "family",
            "n_categorical_encoded", "rows", "features", "n_folds_total",
            "n_folds_ok", "acc_mean", "acc_std", "auc_mean", "auc_std",
            "logloss_mean", "logloss_std", "train_s_median", "time_s_median",
            "median_post_s"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    if "time_s" not in df.columns:
        df = add_headline_time(df)
    df = df.copy()
    df["_primary"] = primary_run_mask(df)
    rows = []
    for (ds, method), g in df.groupby(["dataset", "method"], dropna=False):
        ok = g[g["status"] == "OK"]
        # Accuracy/AUC/logloss (and the fold count they're reported against)
        # come from the primary run only (rep 0, default seed) -- D12's
        # --rep 2 timing repeat and --seed 2 accuracy-variance repeat on the
        # huge datasets must never silently move these numbers or inflate
        # n_folds_ok past the real, independently-drawn fold count (review
        # finding). train_s_median/time_s_median deliberately keep every OK
        # row incl. D12(a)'s repeat -- that repeat exists precisely to give a
        # median-of-2 timing sample (PROTOCOL.md D12a).
        ok_primary = ok[ok["_primary"]] if "_primary" in ok else ok

        def first_or(colname, default):
            s = g[colname].dropna() if colname in g else pd.Series(dtype=float)
            return s.iloc[0] if len(s) else default

        rows.append({
            "dataset": ds, "method": method,
            "suite": first_or("suite", ""),
            "_source": first_or("_source", ""),
            "family": first_or("family", ARM_FAMILY.get(method, "")),
            "n_categorical_encoded": first_or("n_categorical_encoded", np.nan),
            "rows": first_or("rows", np.nan),
            "features": first_or("features", np.nan),
            "n_folds_total": g["fold"].nunique(),
            "n_folds_ok": ok_primary["fold"].nunique(),
            "acc_mean": ok_primary["test_acc"].mean(), "acc_std": ok_primary["test_acc"].std(),
            "auc_mean": ok_primary["test_auc"].mean(), "auc_std": ok_primary["test_auc"].std(),
            "logloss_mean": ok_primary["test_logloss"].mean(),
            "logloss_std": ok_primary["test_logloss"].std(),
            "train_s_median": ok["train_s"].median(),
            "time_s_median": ok["time_s"].median(),
            "median_post_s": ok["post_s"].median() if "post_s" in ok else np.nan,
        })
    out = pd.DataFrame(rows, columns=cols)
    order = {a: i for i, a in enumerate(ALL_ARMS)}
    out["_o"] = out["method"].map(order).fillna(999)
    out = out.sort_values(["dataset", "_o"]).drop(columns="_o").reset_index(drop=True)
    return out


def family_tables(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """D4: RF-family and GBT-family summary tables, split from (a). The only
    cross-family artifact anywhere in this file is fig_huge_pareto (D4's
    AUC-vs-train_s Pareto scatter) -- these two tables are never merged
    or ranked against each other."""
    rf = summary[summary["method"].isin(RF_ARMS)].copy()
    gbt = summary[summary["method"].isin(GBT_ARMS)].copy()
    return rf, gbt


# ==========================================================================
# D7: bit-identity report (spo_rf_exact_stdsort vs spo_rf_exact_hwy)
# ==========================================================================

def bit_identity_report(summary: pd.DataFrame) -> pd.DataFrame:
    """Datasets where spo_rf_exact_stdsort and spo_rf_exact_hwy produce
    identical accuracy/AUC/log-loss cells (fold-mean, from `summary`) -- if
    the trees are bit-identical, every fold's prediction is identical, so
    the fold aggregate is identical too; a NaN on either side is never
    "identical". Also requires equal n_folds_ok (else the two aggregates
    are averaging different fold sets and an incidental equal mean would be
    misleading)."""
    cols = ["dataset", "n_folds_ok_stdsort", "n_folds_ok_hwy", "acc_identical",
            "auc_identical", "logloss_identical", "bit_identical"]
    if summary.empty:
        return pd.DataFrame(columns=cols)
    a = summary[summary["method"] == EXACT_STDSORT].set_index("dataset")
    b = summary[summary["method"] == EXACT_HWY].set_index("dataset")
    common = sorted(set(a.index) & set(b.index))
    rows = []
    for ds in common:
        ra, rb = a.loc[ds], b.loc[ds]

        def _eq(col: str) -> bool:
            va, vb = ra[col], rb[col]
            if va != va or vb != vb:
                return False
            return bool(va == vb)

        acc_eq, auc_eq, ll_eq = _eq("acc_mean"), _eq("auc_mean"), _eq("logloss_mean")
        same_folds = ra["n_folds_ok"] == rb["n_folds_ok"]
        rows.append({
            "dataset": ds,
            "n_folds_ok_stdsort": int(ra["n_folds_ok"]), "n_folds_ok_hwy": int(rb["n_folds_ok"]),
            "acc_identical": acc_eq, "auc_identical": auc_eq, "logloss_identical": ll_eq,
            "bit_identical": bool(acc_eq and auc_eq and ll_eq and same_folds),
        })
    return pd.DataFrame(rows, columns=cols)


# ==========================================================================
# D8: fixed-set comparisons — Holm-corrected Wilcoxon, win/tie/loss
# ==========================================================================

def _paired_delta_tests(df: pd.DataFrame, baseline: str, comparator: str,
                         metric: str) -> dict:
    """One dataset's worth: fold-aligned delta = comparator - baseline on
    `metric`, Wilcoxon signed-rank + paired t-test. NaN p-values when there
    are fewer than 2 paired, non-NaN observations."""
    keep = ["fold", "rep", metric]
    b = df[(df["method"] == baseline) & (df["status"] == "OK")][keep]
    c = df[(df["method"] == comparator) & (df["status"] == "OK")][keep]
    m = b.merge(c, on=["fold", "rep"], suffixes=("_base", "_cmp")).dropna()
    result = {"n_pairs": len(m), "mean_delta": np.nan, "wilcoxon_p": np.nan,
              "ttest_p": np.nan}
    if len(m) == 0:
        return result
    delta = m[f"{metric}_cmp"] - m[f"{metric}_base"]
    # mean_delta is well-defined with a single pair (e.g. every huge dataset,
    # which has exactly one split) -- only the significance tests need >= 2.
    result["mean_delta"] = float(delta.mean())
    if len(m) < 2:
        return result
    # scipy's wilcoxon computes a well-defined exact p-value for a constant
    # nonzero delta too (e.g. 0.0625 at n=5 folds); ttest_rel degrades to a
    # clean p=0.0/nan rather than raising. Both handled by the try/except.
    try:
        result["wilcoxon_p"] = float(stats.wilcoxon(delta).pvalue)
    except Exception:
        pass
    try:
        result["ttest_p"] = float(
            stats.ttest_rel(m[f"{metric}_cmp"], m[f"{metric}_base"]).pvalue)
    except Exception:
        pass
    return result


def holm_correction(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down correction across a family of tests. NaN
    p-values (missing data) pass through as NaN, excluded from the
    correction (they neither consume nor shift the rank of real p-values)."""
    idx = [i for i, p in enumerate(pvalues) if p == p]
    m = len(idx)
    adj: list[float] = [np.nan] * len(pvalues)
    if m == 0:
        return adj
    order = sorted(idx, key=lambda i: pvalues[i])
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, (m - rank) * pvalues[i])
        adj[i] = min(1.0, running_max)
    return adj


def _wtl(delta: float, p_holm: float, higher_is_better: bool,
         tie_band: float | None = None) -> str:
    """win/tie/loss for `baseline` vs a comparator. Two tie definitions
    coexist by design (PROTOCOL.md D8): a fixed magnitude band (tie iff
    |delta| < tie_band, e.g. AUC's 0.005) when `tie_band` is given, else the
    Holm-corrected-significance rule (tie iff p_holm >= ALPHA) used for the
    metrics D8 doesn't define a magnitude band for (acc, log-loss)."""
    if delta != delta:
        return "n/a"
    if tie_band is not None:
        if abs(delta) < tie_band:
            return "tie"
        baseline_better = (delta < 0) if higher_is_better else (delta > 0)
        return "win" if baseline_better else "loss"
    if p_holm != p_holm:
        return "n/a"
    if p_holm >= ALPHA:
        return "tie"
    baseline_better = (delta < 0) if higher_is_better else (delta > 0)
    return "win" if baseline_better else "loss"


def cross_table(df: pd.DataFrame, baseline: str, comparators: list[str]) -> pd.DataFrame:
    """Per dataset x comparator: acc/auc/logloss deltas (comparator -
    baseline), Wilcoxon + paired-t p-values, Holm-corrected across datasets
    per (comparator, metric), and win/tie/loss for `baseline`. AUC (the
    primary metric, D8) uses the fixed |delta| < AUC_TIE_BAND tie band;
    accuracy and log-loss use the Holm-corrected-significance tie rule
    (see _wtl). Callers pass the two FIXED_COMPARISONS sets (D8), never an
    ad hoc list, and never a comparator from the other model family.
    Restricted to the primary run (primary_run_mask) before pairing: D12's
    huge-tier --rep 2 / --seed 2 repeats are not independent folds, and
    _paired_delta_tests merges only on (fold, rep) -- without this filter a
    huge dataset's single real fold plus its D12(a)/(b) repeats collide on
    that key and the merge fabricates 2-4 paired "folds" out of one real
    observation, corrupting n_pairs and the Wilcoxon/Holm statistic
    (review finding)."""
    cols = ["dataset", "comparator"]
    for metric in ("test_acc", "test_auc", "test_logloss"):
        cols += [f"{metric}_delta", f"{metric}_n_pairs", f"{metric}_wilcoxon_p",
                 f"{metric}_ttest_p", f"{metric}_wilcoxon_p_holm", f"wtl_{metric}"]
    if not df.empty:
        df = df[primary_run_mask(df)]
    if df.empty:
        return pd.DataFrame(columns=cols)

    higher_better = {"test_acc": True, "test_auc": True, "test_logloss": False}
    tie_bands = {"test_auc": AUC_TIE_BAND}
    blocks = []
    for comparator in comparators:
        recs = []
        for ds, g in df.groupby("dataset"):
            methods_here = set(g["method"])
            if baseline not in methods_here or comparator not in methods_here:
                continue
            rec = {"dataset": ds, "comparator": comparator}
            for metric in higher_better:
                t = _paired_delta_tests(g, baseline, comparator, metric)
                rec[f"{metric}_delta"] = t["mean_delta"]
                rec[f"{metric}_n_pairs"] = t["n_pairs"]
                rec[f"{metric}_wilcoxon_p"] = t["wilcoxon_p"]
                rec[f"{metric}_ttest_p"] = t["ttest_p"]
            recs.append(rec)
        if not recs:
            continue
        block = pd.DataFrame(recs)
        for metric in higher_better:
            block[f"{metric}_wilcoxon_p_holm"] = holm_correction(
                block[f"{metric}_wilcoxon_p"].tolist())
            block[f"wtl_{metric}"] = [
                _wtl(d, p, higher_better[metric], tie_bands.get(metric))
                for d, p in zip(block[f"{metric}_delta"], block[f"{metric}_wilcoxon_p_holm"])
            ]
        blocks.append(block)
    if not blocks:
        return pd.DataFrame(columns=cols)
    return pd.concat(blocks, ignore_index=True)[cols]


def win_tie_loss_summary(cross: pd.DataFrame, metric: str = "test_auc") -> pd.DataFrame:
    """AUC (test_auc) is the primary metric (D8); pass a different metric to
    summarize acc/logloss instead."""
    col = f"wtl_{metric}"
    if cross.empty or col not in cross.columns:
        return pd.DataFrame(columns=["comparator", "wins", "ties", "losses", "n_a", "n_datasets"])
    rows = []
    for comparator, g in cross.groupby("comparator"):
        counts = g[col].value_counts()
        rows.append({
            "comparator": comparator,
            "wins": int(counts.get("win", 0)), "ties": int(counts.get("tie", 0)),
            "losses": int(counts.get("loss", 0)), "n_a": int(counts.get("n/a", 0)),
            "n_datasets": len(g),
        })
    return pd.DataFrame(rows)


# ==========================================================================
# D8: Friedman test + Nemenyi critical difference (per family, AUC primary)
# ==========================================================================

def _nemenyi_cd(k: int, n: int, alpha: float = ALPHA) -> float:
    """Nemenyi post-hoc critical difference (Demsar 2006): CD = q_alpha *
    sqrt(k(k+1)/6n), q_alpha = studentized-range(1-alpha, k, inf) / sqrt(2).
    k = number of methods compared, n = number of datasets (blocks)."""
    if k < 2 or n < 2:
        return float("nan")
    q = float(stats.studentized_range.ppf(1 - alpha, k, np.inf)) / math.sqrt(2.0)
    return q * math.sqrt(k * (k + 1) / (6.0 * n))


def mean_rank_table(summary: pd.DataFrame, methods: list[str], metric: str,
                     ascending: bool) -> tuple[pd.DataFrame, dict]:
    """Mean rank per method across datasets (rank 1 = best) + a Friedman
    test and Nemenyi CD, restricted to datasets where every requested method
    has a value (partial data just shrinks that set)."""
    piv = summary[summary["method"].isin(methods)].pivot_table(
        index="dataset", columns="method", values=metric, aggfunc="mean")
    piv = piv.reindex(columns=methods)
    complete = piv.dropna(axis=0, how="any")
    ranks = piv.rank(axis=1, ascending=ascending, method="average")
    mean_rank = ranks.mean(axis=0).reindex(methods)
    friedman = {"n_datasets": int(len(complete)), "k_methods": len(methods),
                "statistic": np.nan, "p_value": np.nan,
                "cd_nemenyi_0.05": _nemenyi_cd(len(methods), len(complete))}
    if len(complete) >= 3 and complete.shape[1] >= 3:
        try:
            stat, p = stats.friedmanchisquare(*[complete[m].values for m in methods])
            friedman["statistic"], friedman["p_value"] = float(stat), float(p)
        except Exception:
            pass
    out = pd.DataFrame({"method": methods, "mean_rank": mean_rank.values,
                         "n_datasets_used": len(complete)})
    return out, friedman


def _cd_cliques(sorted_ranks: list[float], cd: float) -> list[tuple[int, int]]:
    """Maximal runs of methods (already sorted by rank ascending) whose rank
    span is <= cd -- the "not significantly different" groups a CD diagram
    draws as a connecting bar."""
    n = len(sorted_ranks)
    raw = []
    for i in range(n):
        j = i
        while j + 1 < n and sorted_ranks[j + 1] - sorted_ranks[i] <= cd:
            j += 1
        if j > i:
            raw.append((i, j))
    return [c for c in raw if not any(c != d and d[0] <= c[0] and c[1] <= d[1] for d in raw)]


# ==========================================================================
# (c) Speed: per-dataset time ratios + geometric means
# ==========================================================================

def speed_ratio_table(df: pd.DataFrame, baseline: str,
                       min_rows_for_headline: int = MIN_ROWS_FOR_HEADLINE_SPEED) -> pd.DataFrame:
    """Per dataset: t(method)/t(baseline) using the median headline time_s
    over folds (train_s, the "Training block took" timer -- add_headline_time,
    PROTOCOL.md D1-rev). Two trailing rows give the across-dataset geometric
    mean per method:
    GEOMEAN_ALL (every dataset) and GEOMEAN_HEADLINE (D8: rows >= 2000 only
    -- small datasets are excluded from the headline number because
    thread-pool startup dominates there, but every dataset's own ratio is
    still listed in the per-dataset rows above)."""
    cols = ["dataset", "rows"] + ALL_ARMS
    if "time_s" not in df.columns:
        df = add_headline_time(df)
    ok = df[df["status"] == "OK"]
    if ok.empty:
        return pd.DataFrame(columns=cols)
    med = ok.groupby(["dataset", "method"])["time_s"].median().unstack("method")
    rows_by_ds = ok.groupby("dataset")["rows"].first()
    if baseline not in med.columns:
        med[baseline] = np.nan
    ratio = med.div(med[baseline], axis=0).replace([np.inf, -np.inf], np.nan)

    def _gmean(col: pd.Series) -> float:
        vals = col.dropna()
        vals = vals[vals > 0]
        return float(stats.gmean(vals)) if len(vals) else np.nan

    headline_mask = (rows_by_ds.reindex(ratio.index) >= min_rows_for_headline).fillna(False)
    geo_all = ratio.apply(_gmean, axis=0)
    geo_headline = ratio[headline_mask].apply(_gmean, axis=0)
    excluded = sorted(rows_by_ds[~headline_mask].index) if not rows_by_ds.empty else []
    if excluded:
        print(f"[speed_ratio_table] excluded from GEOMEAN_HEADLINE (rows < "
              f"{min_rows_for_headline}, still listed per-dataset above): "
              f"{', '.join(str(x) for x in excluded)}", file=sys.stderr)

    out = ratio.reset_index()
    out.insert(1, "rows", out["dataset"].map(rows_by_ds))
    summary_rows = pd.DataFrame([
        {"dataset": "GEOMEAN_ALL", "rows": np.nan, **geo_all.to_dict()},
        {"dataset": f"GEOMEAN_HEADLINE(rows>={min_rows_for_headline})", "rows": np.nan,
         **geo_headline.to_dict()},
    ])
    out = pd.concat([out, summary_rows], ignore_index=True)
    ordered_cols = ["dataset", "rows"] + [a for a in ALL_ARMS if a in out.columns]
    return out[ordered_cols]


# ==========================================================================
# (d) Figures — matplotlib, PDF+PNG, colorblind-safe palette, log-x time
# axes, consistent (color, marker) per method across every panel, direct
# labels preferred over legends where the point count allows it.
# ==========================================================================

def save_fig(fig: plt.Figure, out_dir: Path, name: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200)
        paths.append(str(p))
    plt.close(fig)
    return paths


def _empty_panel(out_dir: Path, name: str, message: str) -> list[str]:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=9, wrap=True)
    ax.axis("off")
    return save_fig(fig, out_dir, name)


def _clean_log_axis(axis) -> None:
    """A log axis whose data spans less than one decade (e.g. one huge
    dataset's RF arms, or --selftest's synthetic single-dataset panel) gets
    NO major tick from the default LogLocator, and matplotlib's fallback
    then either shows nothing or auto-fills minor ticks WITH numeric labels
    at every 2/3/4/6/7/8/9 subdivision -- unreadable clutter on a wide range,
    blank on a narrow one. Fix both: major ticks at the 1/2/5 x 10^n
    subdivisions (always at least one is in view), no minor tick labels."""
    axis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation(base=10.0, labelOnlyBase=False))
    axis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def _pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Skyline maximizing y_col while minimizing x_col (e.g. AUC vs time)."""
    pts = df[[x_col, y_col]].dropna().sort_values(x_col)
    keep = []
    best_y = -np.inf
    for idx, row in pts.iterrows():
        if row[y_col] > best_y:
            keep.append(idx)
            best_y = row[y_col]
    return df.loc[keep].sort_values(x_col)


def _repel_labels(ax: plt.Axes, fig: plt.Figure, items: list[dict], max_iter: int = 60,
                   max_offset_pt: float = 60.0, max_seconds: float = 4.0) -> None:
    """Simple, dependency-free label repel: place each item's text at a small
    offset from its point (pre-staggered by y-rank so a tight vertical
    cluster doesn't start fully overlapped), then iteratively nudge any pair
    whose rendered bounding boxes overlap apart vertically, re-measuring with
    the actual Agg renderer each pass (no adjustText in this venv). The step
    size decays over the iteration budget so the loop settles instead of
    oscillating indefinitely between two overlapping configurations; a
    wall-clock cutoff (each fig.canvas.draw() during the loop repaints the
    WHOLE figure, so cost grows with every panel already finished) is a hard
    backstop against runaway cost on a dense multi-panel figure. Offsets are
    clamped to +/-max_offset_pt so a dense cluster settles for a residual
    overlap rather than drifting a label into a neighboring axes' title. Only
    the FINAL placement gets a connecting leader line (arrowprops) back to
    its point for labels that moved far -- computing that during every
    intermediate iteration is unnecessary rendering cost.
    `items`: [{"x","y","text","color","fontsize","fontweight"}, ...]."""
    if not items:
        return
    order = sorted(range(len(items)), key=lambda i: -items[i]["y"])
    stagger = {idx: rank for rank, idx in enumerate(order)}
    offsets = [[6.0, 4.0 + 3.0 * stagger[i]] for i in range(len(items))]
    artists: list = [None] * len(items)

    def render(with_arrows: bool) -> None:
        for a in artists:
            if a is not None:
                a.remove()
        for i, it in enumerate(items):
            dx, dy = offsets[i]
            far = with_arrows and (dx * dx + dy * dy) ** 0.5 > 14.0
            artists[i] = ax.annotate(
                it["text"], (it["x"], it["y"]), fontsize=it.get("fontsize", 7.0),
                color=it.get("color", "black"), fontweight=it.get("fontweight", "normal"),
                xytext=tuple(offsets[i]), textcoords="offset points", zorder=6, clip_on=False,
                arrowprops=(dict(arrowstyle="-", color=it.get("color", "black"), lw=0.6,
                                  alpha=0.55, shrinkA=0, shrinkB=2) if far else None))
        fig.canvas.draw()

    render(with_arrows=False)
    renderer = fig.canvas.get_renderer()
    start = time.monotonic()
    for it_num in range(max_iter):
        if time.monotonic() - start > max_seconds:
            break
        step = 1.8 * (1.0 - it_num / max_iter)  # decay -> converge, not oscillate
        boxes = [a.get_window_extent(renderer=renderer) for a in artists]
        moved = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if boxes[i].overlaps(boxes[j]):
                    ci = 0.5 * (boxes[i].y0 + boxes[i].y1)
                    cj = 0.5 * (boxes[j].y0 + boxes[j].y1)
                    signed = step if ci <= cj else -step
                    new_i = offsets[i][1] - signed
                    new_j = offsets[j][1] + signed
                    if abs(new_i) <= max_offset_pt or abs(new_i) < abs(offsets[i][1]):
                        offsets[i][1] = new_i
                        moved = True
                    if abs(new_j) <= max_offset_pt or abs(new_j) < abs(offsets[j][1]):
                        offsets[j][1] = new_j
                        moved = True
        if not moved:
            break
        render(with_arrows=False)
        renderer = fig.canvas.get_renderer()
    render(with_arrows=True)


def _draw_huge_pareto(large_df: pd.DataFrame, arms_filter: list[str] | None, out_dir: Path,
                       name: str, suptitle: str) -> list[str]:
    """D4: the only cross-family artifact in this file (when arms_filter is
    None) -- AUC-vs-train_s Pareto scatter, one panel per huge dataset.
    RF-family points are filled markers, GBT-family points are hollow
    (facecolor "none"); each arm keeps one consistent (color, marker) pair
    across every panel (ARM_STYLE). The non-dominated (Pareto) frontier is
    drawn per panel. Our two shipped arms (spo_rf_dyn_vec, spo_gbt_dyn_vec)
    get a larger marker and a bold, larger-font label. Point labels use
    _repel_labels so they don't overlap. Arms that didn't complete (OOM /
    TIMEOUT / ERROR) are listed in a small text box per panel instead of
    silently vanishing. D12 gives some arms extra rows on a huge dataset (a
    rep=1 timing repeat, a seed=2 accuracy repeat) -- one point per
    (dataset, method) is plotted, median over whatever rows exist, consistent
    with how per_dataset_summary/speed_ratio_table collapse those."""
    if "time_s" not in large_df.columns:
        large_df = add_headline_time(large_df)
    base = large_df if arms_filter is None else large_df[large_df["method"].isin(arms_filter)]
    ok = base[(base["status"] == "OK") & base["time_s"].notna() & base["test_auc"].notna()]
    all_datasets = sorted(base["dataset"].dropna().unique())
    datasets = sorted(ok["dataset"].dropna().unique())
    all_missing = sorted(set(all_datasets) - set(datasets))
    if all_missing:
        print(f"[{name}] no OK rows with AUC+time for: {', '.join(all_missing)}", file=sys.stderr)
    if not datasets:
        return _empty_panel(out_dir, name, "no huge-dataset results yet")

    fig, axes = plt.subplots(1, len(datasets), figsize=(5.1 * len(datasets), 6.2), squeeze=False)
    axes = axes[0]
    arms_seen: list[str] = []
    for ax, ds in zip(axes, datasets):
        g_raw = ok[ok["dataset"] == ds]
        g = (g_raw.groupby(["dataset", "method"], as_index=False)
                  .agg(time_s=("time_s", "median"), test_auc=("test_auc", "median"),
                       family=("family", "first")))
        items = []
        for _, row in g.iterrows():
            method = row["method"]
            color, marker = ARM_STYLE.get(method, ("#999999", "o"))
            is_ours = method in (RF_BASELINE, GBT_BASELINE)
            size = 190 if is_ours else 65
            if row["family"] == "gbt":
                ax.scatter(row["time_s"], row["test_auc"], marker=marker, s=size,
                           facecolors="none", edgecolors=color,
                           linewidths=2.0 if is_ours else 1.2, zorder=4)
            else:
                ax.scatter(row["time_s"], row["test_auc"], marker=marker, s=size,
                           facecolors=color, edgecolors=("black" if is_ours else "white"),
                           linewidths=1.5 if is_ours else 0.6, zorder=4)
            if method not in arms_seen:
                arms_seen.append(method)
            items.append({"x": row["time_s"], "y": row["test_auc"],
                          "text": ARM_LABELS.get(method, method), "color": color,
                          "fontsize": 8.3 if is_ours else 6.5,
                          "fontweight": "bold" if is_ours else "normal"})
        front = _pareto_frontier(g, "time_s", "test_auc")
        if len(front) >= 2:
            ax.plot(front["time_s"], front["test_auc"], color="#555555", lw=1.1, ls="--", zorder=2)
        # Extra vertical headroom: several arms land within a few thousandths
        # of AUC of each other on these datasets, so the repel below needs
        # room to stack labels without climbing into the per-panel title.
        ax.margins(y=0.35)
        _repel_labels(ax, fig, items)

        bad = base[(base["dataset"] == ds) & (base["status"] != "OK")][["method", "status"]]
        bad = bad.drop_duplicates()
        if not bad.empty:
            bad_txt = "\n".join(f"{ARM_LABELS.get(m, m)}: {s}" for m, s in bad.itertuples(index=False))
            ax.text(0.03, 0.03, bad_txt, transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=6.6, bbox=dict(boxstyle="round", fc="#fff4e0", ec="#D55E00", lw=0.8),
                    zorder=7)
        ax.set_xscale("log")
        # A huge dataset's own arms can land within under one decade (e.g.
        # EPSILON's RF arms all sit within ~30-400s, or --selftest's single
        # synthetic dataset) -- see _clean_log_axis for why the default
        # LogLocator/formatter can't be trusted at either extreme.
        _clean_log_axis(ax.xaxis)
        ax.set_xlabel("Training time [s] (log)")
        ax.set_ylabel("Test ROC AUC")
        ax.set_title(ds)
        ax.grid(True, which="both", alpha=0.25)

    handles = []
    for m in [a for a in ALL_ARMS if a in arms_seen]:
        color, marker = ARM_STYLE[m]
        hollow = ARM_FAMILY[m] == "gbt"
        handles.append(plt.Line2D(
            [0], [0], marker=marker, color="w", lw=0, markersize=8,
            markerfacecolor=("none" if hollow else color), markeredgecolor=color,
            markeredgewidth=1.4 if hollow else 0.8, label=ARM_LABELS.get(m, m)))
    ncol = min(5, max(1, len(handles)))
    fig.tight_layout()
    fig.legend(handles=handles, loc="upper center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=7.4,
               title="filled = RF family, hollow = GBT family (bold/larger = our arms)")
    fig.suptitle(suptitle, y=1.04)
    return save_fig(fig, out_dir, name)


def fig_huge_pareto(large_df: pd.DataFrame, out_dir: Path) -> list[str]:
    """D4: cross-family Pareto scatter, all (up to) 15 arms."""
    return _draw_huge_pareto(
        large_df, None, out_dir, "fig_huge_pareto",
        "AUC vs training time (train_s) -- huge datasets (single split); "
        "dashed line = Pareto frontier")


def fig_huge_pareto_rf(large_df: pd.DataFrame, out_dir: Path) -> list[str]:
    """Paper's main huge-dataset figure: same Pareto scatter restricted to
    the RF-family arms (the paper's contribution lives in the RF family)."""
    return _draw_huge_pareto(
        large_df, RF_ARMS, out_dir, "fig_huge_pareto_rf",
        "AUC vs training time (train_s) -- huge datasets, RF family only; "
        "dashed line = Pareto frontier")


def fig_speedup_vs_rows(suite_df: pd.DataFrame, large_df: pd.DataFrame,
                         out_dir: Path) -> list[str]:
    """Speedup-vs-rows scatter for the RF-family SPO arms (the six SPO-RF
    arms only -- aa_rf_exact is dropped: it's a different algorithm
    (axis-aligned), belongs in the tables not this speedup claim, and
    xgboost_rf/lightgbm_rf are excluded too: no exact-VQSort baseline to
    compare against). speedup = median headline-time_s(spo_rf_exact_stdsort)
    / median headline-time_s(arm) per dataset -- the paper's exact-vs-dyn_vec
    claim generalized to every SPO-RF arm on the exact-std_sort baseline, log2
    y axis (a doubling/halving grid reads naturally for a speedup ratio), a
    y=1 reference line, and the three huge datasets (HIGGS/SUSY/EPSILON)
    appended as labelled rightmost points so the size axis reaches its real
    extreme instead of stopping at the suite's largest dataset."""
    if "time_s" not in suite_df.columns:
        suite_df = add_headline_time(suite_df)
    if "time_s" not in large_df.columns:
        large_df = add_headline_time(large_df)
    arms = [a for a in RF_ARMS if a not in ("xgboost_rf", "lightgbm_rf", "aa_rf_exact")]
    combined = pd.concat(
        [suite_df.assign(_scope="suite"), large_df.assign(_scope="large")], ignore_index=True)
    df = combined[(combined["status"] == "OK") & combined["method"].isin(arms)]
    if df.empty:
        return _empty_panel(out_dir, "fig_speedup_vs_rows", "no suite results yet")
    med = (df.groupby(["dataset", "method", "_scope"])
             .agg(rows=("rows", "first"), time_s=("time_s", "median"))
             .reset_index())
    base = med[med["method"] == EXACT_STDSORT][["dataset", "time_s"]].rename(
        columns={"time_s": "base_s"})
    med = med.merge(base, on="dataset", how="left")
    med["speedup"] = med["base_s"] / med["time_s"]

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    any_points = False
    for arm in arms:
        color, marker = ARM_STYLE[arm]
        g_suite = (med[(med["method"] == arm) & (med["_scope"] == "suite")]
                   .dropna(subset=["rows", "speedup"]).sort_values("rows"))
        if not g_suite.empty:
            any_points = True
            ax.scatter(g_suite["rows"], g_suite["speedup"], color=color, marker=marker, s=45,
                       label=ARM_LABELS[arm], alpha=0.9, zorder=3)
        g_large = (med[(med["method"] == arm) & (med["_scope"] == "large")]
                   .dropna(subset=["rows", "speedup"]).sort_values("rows"))
        if not g_large.empty:
            any_points = True
            ax.scatter(g_large["rows"], g_large["speedup"], color=color, marker=marker, s=100,
                       edgecolor="black", linewidth=1.1, zorder=4,
                       label=None if not g_suite.empty else ARM_LABELS[arm])
            for _, row in g_large.iterrows():
                ax.annotate(row["dataset"], (row["rows"], row["speedup"]), fontsize=6.6,
                            xytext=(4, 3), textcoords="offset points", zorder=5)
    if not any_points:
        plt.close(fig)
        return _empty_panel(out_dir, "fig_speedup_vs_rows", "no suite results yet")
    ax.axhline(1.0, color="#999999", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("Training rows (log scale; large markers = HIGGS/SUSY/EPSILON)")
    ax.set_ylabel(f"Speedup vs {ARM_LABELS[EXACT_STDSORT]} (log2)")
    ax.set_title("Speedup vs dataset size -- suite + huge datasets, RF family")
    ax.legend(fontsize=7.5, ncols=2, loc="best", frameon=False)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    return save_fig(fig, out_dir, "fig_speedup_vs_rows")


def _speedup_pivot(spm: pd.DataFrame, numerator: str, denominator: str,
                    min_examples: int = 1) -> pd.DataFrame:
    df = spm[(spm["status"] == "OK") & (spm["min_examples"] == min_examples)]
    med = df.groupby(["rows", "max_depth", "method"])["train_s"].median().reset_index()
    num = med[med["method"] == numerator].rename(columns={"train_s": "t_num"})
    den = med[med["method"] == denominator].rename(columns={"train_s": "t_den"})
    m = num.merge(den[["rows", "max_depth", "t_den"]], on=["rows", "max_depth"])
    m["speedup"] = m["t_num"] / m["t_den"]
    return m[["rows", "max_depth", "speedup"]]


def fig_speedup_vs_depth(spm: pd.DataFrame, out_dir: Path) -> list[str]:
    """B1 (PROTOCOL.md U1: HIGGS at full size only, so the rows axis has a
    single level and a heatmap degenerates): speedup of Dyn-Vec over the
    exact baselines as a function of max_depth on full HIGGS, min_examples=1.
    One line per exact baseline (Highway sort at every depth; std::sort only
    at unlimited depth by design) plus the raw training times as bars."""
    if spm.empty:
        return _empty_panel(out_dir, "fig_speedup_vs_depth", "no speedup-map results yet")
    df = spm[(spm["status"] == "OK") & (spm["min_examples"] == 1)
             & spm["dataset"].astype(str).str.startswith("higgs")]
    if df.empty:
        return _empty_panel(out_dir, "fig_speedup_vs_depth", "no HIGGS depth cells yet")
    med = df.groupby(["max_depth", "method"])["train_s"].median().unstack()
    depths = sorted(med.index, key=lambda d: (d == -1, d))
    xl = ["unlim." if d == -1 else str(int(d)) for d in depths]
    x = np.arange(len(depths))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.4))
    arms = [a for a in (EXACT_STDSORT, EXACT_HWY, SPEED_BASELINE) if a in med.columns]
    w = 0.8 / max(1, len(arms))
    for i, a in enumerate(arms):
        vals = [med.loc[d, a] if d in med.index else np.nan for d in depths]
        ax0.bar(x + (i - (len(arms) - 1) / 2) * w, vals, width=w, label=ARM_LABELS.get(a, a),
                color=ARM_STYLE.get(a, ("#999999", "o"))[0])
    ax0.set_xticks(x); ax0.set_xticklabels(xl); ax0.set_xlabel("max_depth")
    ax0.set_ylabel("Training time [s]"); ax0.set_title("HIGGS (10.5M rows, 240 trees): time by depth")
    ax0.legend(fontsize=8); ax0.grid(alpha=0.3, axis="y")
    for a, lab, mk in ((EXACT_HWY, "vs Exact (Highway)", "s"), (EXACT_STDSORT, "vs Exact (std::sort)", "o")):
        if a not in med.columns or SPEED_BASELINE not in med.columns:
            continue
        sp = [(med.loc[d, a] / med.loc[d, SPEED_BASELINE]) if d in med.index and pd.notna(med.loc[d, a]) else np.nan
              for d in depths]
        ax1.plot(x, sp, marker=mk, label=lab)
        for xi, v in zip(x, sp):
            if pd.notna(v):
                ax1.annotate(f"{v:.2f}x", (xi, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax1.axhline(1.0, color="gray", ls="--", lw=1)
    ax1.set_xticks(x); ax1.set_xticklabels(xl); ax1.set_xlabel("max_depth")
    ax1.set_ylabel("Speedup of SPO-RF Dyn-Vec [ours]"); ax1.set_title("Speedup by depth (min_examples=1)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    fig.tight_layout()
    return save_fig(fig, out_dir, "fig_speedup_vs_depth")


def fig_heatmap_rows_depth(spm: pd.DataFrame, out_dir: Path) -> list[str]:
    """B1: heatmap(s) of speedup = t(exact)/t(dyn_vec) over rows x max_depth
    (min_examples=1), one panel for the Highway sort exact baseline and one
    for the std::sort exact baseline."""
    if spm.empty:
        return _empty_panel(out_dir, "fig_heatmap_rows_depth", "no speedup-map results yet")
    pairs = [(EXACT_HWY, SPEED_BASELINE, "Exact (Highway) / Dyn-Vec"),
             (EXACT_STDSORT, SPEED_BASELINE, "Exact (std::sort) / Dyn-Vec")]
    # Full depth grid seen anywhere in the B1 design (min_examples=1). An arm
    # like spo_rf_exact_stdsort is by design only run at max_depth=-1 --
    # _speedup_pivot's inner merge then drops the other depths entirely
    # rather than leaving NaN cells, so that panel's own depth_order is just
    # narrower, not sparse within a shared grid. Compare against this full
    # set (not the panel's own column count) to actually detect that case.
    # HIGGS cells only: the trunk width cells (B5) also have 1M rows and
    # min_examples=1 and must not appear as a "rows" level of this heatmap.
    spm = spm[spm["dataset"].astype(str).str.startswith("higgs")]
    b1 = spm[(spm["status"] == "OK") & (spm["min_examples"] == 1)]
    all_depths_seen = set(b1["max_depth"].dropna().unique())
    fig, axes = plt.subplots(1, len(pairs), figsize=(6.8 * len(pairs), 5.2), squeeze=False)
    axes = axes[0]
    cmap = plt.get_cmap("cividis")  # perceptually-uniform, colorblind-safe
    any_data = False
    for ax, (num, den, title) in zip(axes, pairs):
        piv = _speedup_pivot(spm, num, den)
        if piv.dropna().empty:
            ax.text(0.5, 0.5, f"no data for\n{title}", ha="center", va="center", fontsize=9)
            ax.axis("off")
            continue
        any_data = True
        table = piv.pivot(index="rows", columns="max_depth", values="speedup").sort_index()
        depth_order = sorted(table.columns, key=lambda d: (d == -1, d))
        table = table[depth_order]
        vals = table.values.astype(float)
        im = ax.imshow(vals, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels(["unlim." if d == -1 else str(int(d)) for d in table.columns])
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels([f"{int(r):,}" for r in table.index])
        ax.set_xlabel("max_depth")
        ax.set_ylabel("rows")
        ax.set_title(title)
        # This arm may be measured at fewer depths than the full B1 grid by
        # design (e.g. spo_rf_exact_stdsort at max_depth=-1 only) -- a
        # narrower panel next to a fully-populated one is expected, not a
        # broken run; say so rather than let it read as "something is
        # missing".
        if 1 <= len(depth_order) < len(all_depths_seen):
            ax.text(0.5, -0.14, "measured at fewer depths than the full grid, by design",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=7.5, style="italic", color="#555555")
        finite = vals[np.isfinite(vals)]
        thresh = finite.mean() if finite.size else 0.0
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}x", ha="center", va="center", fontsize=7,
                            color="white" if v < thresh else "black")
        fig.colorbar(im, ax=ax, label="speedup (x)")
    if not any_data:
        plt.close(fig)
        return _empty_panel(out_dir, "fig_heatmap_rows_depth", "no speedup-map results yet")
    fig.suptitle("Speedup heatmap -- rows x max_depth (min_examples=1, B1 design)")
    fig.tight_layout()
    return save_fig(fig, out_dir, "fig_heatmap_rows_depth")


def fig_speedup_vs_min_examples(spm: pd.DataFrame, out_dir: Path) -> list[str]:
    """B2: line plot of speedup = t(exact_hwy)/t(dyn_vec) vs min_examples at
    unlimited depth, one line per row-count level."""
    if spm.empty:
        return _empty_panel(out_dir, "fig_speedup_vs_min_examples", "no speedup-map results yet")
    df = spm[(spm["status"] == "OK") & (spm["max_depth"] == -1)]
    med = df.groupby(["rows", "min_examples", "method"])["train_s"].median().reset_index()
    num = med[med["method"] == EXACT_HWY].rename(columns={"train_s": "t_num"})
    den = med[med["method"] == SPEED_BASELINE].rename(columns={"train_s": "t_den"})
    m = num.merge(den[["rows", "min_examples", "t_den"]], on=["rows", "min_examples"])
    m["speedup"] = m["t_num"] / m["t_den"]
    if m.dropna().empty:
        return _empty_panel(out_dir, "fig_speedup_vs_min_examples",
                             "no B2 (min_examples grid) results yet")
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for i, rows in enumerate(sorted(m["rows"].dropna().unique())):
        g = m[m["rows"] == rows].dropna(subset=["min_examples", "speedup"]).sort_values("min_examples")
        if g.empty:
            continue
        ax.plot(g["min_examples"], g["speedup"], color=PALETTE[i % len(PALETTE)],
                marker=MARKERS[i % len(MARKERS)], label=f"{int(rows):,} rows")
    ax.axhline(1.0, color="#999999", lw=0.8, ls="--")
    ax.set_xlabel("min_examples (stopping threshold)")
    ax.set_ylabel(f"Speedup, {ARM_LABELS[EXACT_HWY]} / {ARM_LABELS[SPEED_BASELINE]}")
    ax.set_title("Speedup vs min_examples (HIGGS, unlimited depth)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return save_fig(fig, out_dir, "fig_speedup_vs_min_examples")


def fig_trunk_width(spm: pd.DataFrame, out_dir: Path) -> list[str]:
    """B5 (PROTOCOL.md D10): trunk-generator feature-width axis -- rows 1M,
    unlimited depth, cols {32,128,512,2048,8192}, arms {exact_stdsort,
    exact_hwy, dyn_vec}. Left panel: time vs cols (log-x), one line per arm.
    Right panel: speedup vs dyn_vec, same x axis. Cells are identified by
    run_speedup_map.py's `dataset` column ("trunk_<rows>_x_<cols>"; A7) --
    read generically (dataset prefix "trunk_"), not by exact cell count, so
    this keeps working if the grid changes."""
    if spm.empty:
        return _empty_panel(out_dir, "fig_trunk_width", "no trunk-width (B5) results yet")
    df = spm[(spm["status"] == "OK") & spm["dataset"].astype(str).str.startswith("trunk_")]
    if df.empty:
        return _empty_panel(out_dir, "fig_trunk_width", "no trunk-width (B5) results yet")
    arms = [a for a in (EXACT_STDSORT, EXACT_HWY, SPEED_BASELINE) if a in set(df["method"])]
    if not arms:
        return _empty_panel(out_dir, "fig_trunk_width", "no B5 arms found in speedup_map.csv")
    med = df.groupby(["features", "method"])["train_s"].median().reset_index()
    base = med[med["method"] == SPEED_BASELINE][["features", "train_s"]].rename(
        columns={"train_s": "t_base"})
    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for arm in arms:
        g = med[med["method"] == arm].sort_values("features")
        if g.empty:
            continue
        color, marker = ARM_STYLE[arm]
        ax_t.plot(g["features"], g["train_s"], color=color, marker=marker, label=ARM_LABELS[arm])
        gm = g.merge(base, on="features", how="left")
        gm["speedup"] = gm["t_base"] / gm["train_s"]
        ax_s.plot(gm["features"], gm["speedup"], color=color, marker=marker, label=ARM_LABELS[arm])
    for ax in (ax_t, ax_s):
        ax.set_xscale("log")
        ax.set_xlabel("Trunk width (features, log scale)")
        ax.grid(True, which="both", alpha=0.25)
    ax_t.set_ylabel("Training time [s]")
    ax_t.set_title("Time vs trunk width (1M rows, unlimited depth, 240 trees)")
    ax_s.axhline(1.0, color="#999999", lw=0.8, ls="--")
    ax_s.set_ylabel(f"Speedup vs {ARM_LABELS[SPEED_BASELINE]}")
    ax_s.set_title("Speedup vs trunk width")
    ax_t.legend(fontsize=8, frameon=False)
    fig.suptitle("Feature-width axis (B5) -- trunk generator")
    fig.tight_layout()
    return save_fig(fig, out_dir, "fig_trunk_width")


def fig_gbt_depth(spm: pd.DataFrame, out_dir: Path) -> list[str]:
    """B4 (PROTOCOL.md D9): GBT depth axis -- {spo_gbt_exact_hwy,
    spo_gbt_dyn_vec} x max_depth {6,10,-1} on HIGGS-1M and GiveMeSomeCredit.
    Top row: time vs depth per dataset. Bottom row: speedup exact_hwy/
    dyn_vec vs depth. Datasets/arms read generically from spm (family=="gbt"
    rows), not hardcoded, so it tracks whatever cells_b4.json (A7) actually
    ran."""
    if spm.empty:
        return _empty_panel(out_dir, "fig_gbt_depth", "no GBT-depth (B4) results yet")
    df = spm[(spm["status"] == "OK") & (spm["family"] == "gbt")]
    # PROTOCOL.md U4: the harness maps --tree_depth -1 to 6 for Boosting, so
    # "unlimited" GBT cells are duplicates of depth 6 -- drop them here.
    df = df[pd.to_numeric(df["max_depth"], errors="coerce") != -1]
    if df.empty:
        return _empty_panel(out_dir, "fig_gbt_depth", "no GBT-depth (B4) results yet")
    datasets = sorted(df["dataset"].dropna().unique())
    arms = [a for a in ("spo_gbt_exact_hwy", "spo_gbt_dyn_vec") if a in set(df["method"])]
    if not datasets or not arms:
        return _empty_panel(out_dir, "fig_gbt_depth", "no GBT-depth (B4) arms/datasets found")
    med = df.groupby(["dataset", "max_depth", "method"])["train_s"].median().reset_index()
    fig, axes = plt.subplots(2, len(datasets), figsize=(4.6 * len(datasets), 7.4), squeeze=False)
    for col, ds in enumerate(datasets):
        g = med[med["dataset"] == ds]
        depths = sorted(g["max_depth"].dropna().unique(), key=lambda d: (d == -1, d))
        xt = list(range(len(depths)))
        xlabels = ["unlim." if d == -1 else str(int(d)) for d in depths]
        ax_t, ax_s = axes[0][col], axes[1][col]
        piv = g.pivot(index="max_depth", columns="method", values="train_s").reindex(depths)
        for arm in arms:
            if arm not in piv.columns:
                continue
            color, marker = ARM_STYLE[arm]
            ax_t.plot(xt, piv[arm].values, color=color, marker=marker, label=ARM_LABELS[arm])
        if all(a in piv.columns for a in ("spo_gbt_exact_hwy", "spo_gbt_dyn_vec")):
            speedup = piv["spo_gbt_exact_hwy"] / piv["spo_gbt_dyn_vec"]
            ax_s.plot(xt, speedup.values, color=PALETTE[2], marker="o")
            ax_s.axhline(1.0, color="#999999", lw=0.8, ls="--")
        ax_t.set_xticks(xt); ax_t.set_xticklabels(xlabels)
        ax_s.set_xticks(xt); ax_s.set_xticklabels(xlabels)
        ax_t.set_title(ds); ax_t.set_ylabel("Training time [s]")
        ax_s.set_ylabel(f"Speedup {ARM_LABELS['spo_gbt_exact_hwy']} / {ARM_LABELS['spo_gbt_dyn_vec']}")
        ax_s.set_xlabel("max_depth")
        ax_t.grid(True, alpha=0.25); ax_s.grid(True, alpha=0.25)
        if col == 0:
            ax_t.legend(fontsize=7.5, frameon=False)
    fig.suptitle("SPO-GBT depth axis (B4) -- min_examples=5, 300 trees, 48 threads")
    fig.tight_layout()
    return save_fig(fig, out_dir, "fig_gbt_depth")


def fig_cd_diagram(rank_df: pd.DataFrame, friedman: dict, out_dir: Path, name: str,
                    title: str) -> list[str]:
    """D8: Nemenyi critical-difference diagram, standard Demsar (2006) layout.
    A single rank axis (1 = best) is drawn at the TOP of the figure. Methods
    are split at the median rank: the best half is labelled hanging off the
    LEFT margin, the rest off the RIGHT margin. Each method gets its own
    horizontal row (so labels never overlap) with an L-shaped leader line --
    straight down from its rank position on the axis, then sideways to the
    label at the margin. Rows are assigned closest-to-the-axis-first within
    each side (by distance from that side's edge) so leader lines never
    cross. Thick black bars just under the axis connect maximal cliques of
    methods whose rank span is <= CD (Nemenyi post-hoc, not significantly
    different). Text is allowed to run outside the axes box; save_fig's
    bbox_inches="tight" crops the final canvas to fit it, so exact margin
    widths don't need to be pre-computed for label length."""
    d = rank_df.dropna(subset=["mean_rank"]).sort_values("mean_rank").reset_index(drop=True)
    cd = friedman.get("cd_nemenyi_0.05", np.nan)
    if len(d) < 2 or cd != cd:
        return _empty_panel(out_dir, name, f"{title}\n(insufficient data for a CD diagram)")
    k = len(d)
    ranks = d["mean_rank"].to_numpy(dtype=float)  # already ascending (ties broken stably)
    labels = [ARM_LABELS.get(m, m) for m in d["method"]]
    lo, hi = 1.0, max(float(k), float(np.ceil(ranks.max())))

    half = (k + 1) // 2  # best (lowest-rank) half -> left margin
    left_idx = list(range(half))
    right_idx = list(range(half, k))
    n_left, n_right = len(left_idx), len(right_idx)
    max_rows = max(n_left, n_right, 1)

    # Clique bars: greedily pack into as few stacked rows as possible so
    # overlapping-in-x cliques don't collide.
    cliques = _cd_cliques(ranks.tolist(), cd)
    clique_row_h = 0.16
    clique_rows: list[list[tuple[float, float]]] = []
    clique_row_of: list[int] = []
    for a, b in cliques:
        s, e = ranks[a], ranks[b]
        placed = False
        for ridx, occ in enumerate(clique_rows):
            if all(e < os_ - 0.02 or s > oe_ + 0.02 for os_, oe_ in occ):
                occ.append((s, e)); clique_row_of.append(ridx); placed = True
                break
        if not placed:
            clique_rows.append([(s, e)]); clique_row_of.append(len(clique_rows) - 1)
    n_clique_rows = len(clique_rows)

    axis_y = 0.0
    row_h = 0.42
    label_zone_top = axis_y - 0.22 - n_clique_rows * clique_row_h - 0.30
    scale_y = axis_y + 0.55
    bottom_y = label_zone_top - (max_rows - 1) * row_h - 0.4

    fig_w = max(9.0, 0.11 * max(len(s) for s in labels) * 2 + 0.9 * (hi - lo))
    fig_h = 1.4 + row_h * max_rows
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()

    left_margin_x = lo - 0.35
    right_margin_x = hi + 0.35

    # Rank axis at the top, with integer tick marks.
    ax.plot([lo, hi], [axis_y, axis_y], color="black", lw=1.3, zorder=5)
    for x in range(int(lo), int(hi) + 1):
        ax.plot([x, x], [axis_y - 0.05, axis_y + 0.05], color="black", lw=1.0, zorder=5)
        ax.text(x, axis_y + 0.10, str(x), ha="center", va="bottom", fontsize=8.5)

    # CD reference scale, drawn above the axis.
    ax.plot([lo, lo + cd], [scale_y, scale_y], color="black", lw=1.6)
    ax.plot([lo, lo], [scale_y - 0.05, scale_y + 0.05], color="black", lw=1.2)
    ax.plot([lo + cd, lo + cd], [scale_y - 0.05, scale_y + 0.05], color="black", lw=1.2)
    ax.text(lo + cd / 2, scale_y + 0.10, f"CD = {cd:.3f}", ha="center", va="bottom", fontsize=8.5)

    # Clique bars, stacked just under the axis.
    for (a, b), ridx in zip(cliques, clique_row_of):
        y = axis_y - 0.22 - ridx * clique_row_h
        ax.plot([ranks[a] - 0.03, ranks[b] + 0.03], [y, y], color="black", lw=3.4,
                solid_capstyle="butt", zorder=4)

    def _draw_side(idx_list: list[int], side: str) -> None:
        if side == "left":
            # nearest the left edge (smallest rank) gets the row closest to
            # the axis -- its horizontal leader is short and never crosses
            # a farther-inward method's longer leader.
            ordered = idx_list
            margin_x, ha, dx = left_margin_x, "right", -0.10
        else:
            ordered = sorted(idx_list, key=lambda i: -ranks[i])
            margin_x, ha, dx = right_margin_x, "left", 0.10
        for row_pos, i in enumerate(ordered):
            y = label_zone_top - row_pos * row_h
            x = ranks[i]
            color = PALETTE[i % len(PALETTE)]
            ax.plot([x, x], [axis_y, y], color=color, lw=1.0, zorder=3)
            ax.plot([x, margin_x], [y, y], color=color, lw=1.0, zorder=3)
            ax.plot([x], [axis_y], marker="o", color=color, ms=4.5, zorder=6)
            ax.text(margin_x + dx, y, f"{labels[i]}  ({ranks[i]:.2f})", ha=ha, va="center",
                    fontsize=8.0, color=color)

    _draw_side(left_idx, "left")
    _draw_side(right_idx, "right")

    ax.set_xlim(left_margin_x - 0.3, right_margin_x + 0.3)
    ax.set_ylim(bottom_y - 0.2, scale_y + 0.35)
    p, n_ds = friedman.get("p_value", np.nan), friedman.get("n_datasets", 0)
    subtitle = (f"Friedman p={p:.4g}, n={n_ds} datasets, k={k} methods"
                if p == p else f"n={n_ds} datasets, k={k} methods (Friedman not computed)")
    ax.set_title(f"{title}\n{subtitle}, CD={cd:.3f}\nthick bars = not significantly different "
                 f"(Nemenyi, \u03b1={ALPHA})", fontsize=9.5, y=1.0)
    fig.tight_layout()
    return save_fig(fig, out_dir, name)


# ==========================================================================
# (e) LaTeX tables — booktabs, hand-rolled (pandas>=2.1's DataFrame.to_latex
# routes through Styler, which needs jinja2 -- outside the stdlib +
# pandas/numpy/sklearn/matplotlib budget). Generated only; never hand-edit.
# ==========================================================================

_LATEX_SPECIAL_RE = re.compile(r"([\\&%$#_{}~^])")
_LATEX_SPECIAL_MAP: dict[str, str] = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def _latex_escape(s: object) -> str:
    # Single regex pass (vs. chained .replace calls) so escaping one special
    # character can't accidentally re-match/corrupt another's replacement.
    return _LATEX_SPECIAL_RE.sub(lambda m: _LATEX_SPECIAL_MAP[m.group(1)], str(s))


def _df_to_booktabs(df: pd.DataFrame, caption: str, label: str,
                     float_fmt: str = "{:.4f}", align: str | None = None) -> str:
    cols = list(df.columns)
    align = align or ("l" + "r" * (len(cols) - 1))

    def fmt_cell(v: object) -> str:
        if isinstance(v, (int, np.integer)):
            return str(v)
        if isinstance(v, (float, np.floating)):
            return "--" if v != v else float_fmt.format(v)
        return _latex_escape(v)

    header = " & ".join(_latex_escape(c) for c in cols) + r" \\"
    body_lines = [" & ".join(fmt_cell(v) for v in row) + r" \\"
                  for row in df[cols].itertuples(index=False)]
    body = "\n".join(body_lines) if body_lines else r"\multicolumn{%d}{c}{(no data yet)} \\" % len(cols)
    return (
        "% auto-generated by analyze.py:make_paper_tables -- regenerate, do not hand-edit\n"
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{align}}}\n\\toprule\n"
        f"{header}\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"
        "\\end{table}\n"
    )


def _write_tex(path: Path, content: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return str(path)


_HUGE_TABLE_FOOTNOTE = (
    "Histogram bin counts are not matched across libraries (SPO-RF/SPO-GBT: 64; XGBoost: 256; "
    "LightGBM: 255; CatBoost: 254), disclosed rather than hidden. CatBoost's bootstrap is "
    "disabled (bootstrap type set to No) so that ``no subsampling'' holds for every GBT arm; "
    "no arm uses class weighting. YDF's post-training finalization (structural variable "
    "importances and leaf indexing, a single-threaded walk with no counterpart in the python "
    "libraries' lazily-computed feature importances) is excluded from the headline training "
    "time and shown separately in the Post column.")


def _family_ordered_arms() -> list[str]:
    # RF family first, GBT family second -- both in ARM_ORDER's own within-
    # family order (matches PROTOCOL.md's arm numbering).
    return [a for a in ALL_ARMS if ARM_FAMILY[a] == "rf"] + \
           [a for a in ALL_ARMS if ARM_FAMILY[a] == "gbt"]


def _main_summary_block(summary_sub: pd.DataFrame, all_df_sub: pd.DataFrame) -> pd.DataFrame:
    """One row per arm: n datasets attempted, mean AUC/acc, mean rank of AUC
    within its own family (Friedman/Nemenyi's rank, not cross-family --
    D4), and the geometric-mean training-time ratio vs this arm's own
    family baseline (spo_rf_dyn_vec for RF, spo_gbt_dyn_vec for GBT),
    restricted to datasets with >= MIN_ROWS_FOR_HEADLINE_SPEED rows (D8's
    headline-speed rule, reused here rather than re-derived)."""
    rf_rank, _ = mean_rank_table(summary_sub, RF_ARMS, "auc_mean", ascending=False)
    gbt_rank, _ = mean_rank_table(summary_sub, GBT_ARMS, "auc_mean", ascending=False)
    rank_lookup = dict(zip(rf_rank["method"], rf_rank["mean_rank"]))
    rank_lookup.update(dict(zip(gbt_rank["method"], gbt_rank["mean_rank"])))

    rf_speed = speed_ratio_table(all_df_sub, RF_BASELINE)
    gbt_speed = speed_ratio_table(all_df_sub, GBT_BASELINE)

    def _geo(tbl: pd.DataFrame, method: str) -> float:
        row = tbl[tbl["dataset"].astype(str).str.startswith("GEOMEAN_HEADLINE")]
        if row.empty or method not in row.columns:
            return np.nan
        return float(row[method].iloc[0])

    rows = []
    for m in _family_ordered_arms():
        g = summary_sub[summary_sub["method"] == m]
        fam = ARM_FAMILY[m]
        ratio = _geo(rf_speed, m) if fam == "rf" else _geo(gbt_speed, m)
        rows.append({
            "family": fam, "_arm": m, "method": ARM_LABELS.get(m, m),
            "n": int(g["dataset"].nunique()),
            "mean_auc": g["auc_mean"].mean(), "mean_acc": g["acc_mean"].mean(),
            "mean_rank_auc": rank_lookup.get(m, np.nan),
            "time_ratio_geo": ratio,
        })
    return pd.DataFrame(rows)


def _main_summary_tex(block: pd.DataFrame, caption: str, label: str) -> str:
    """Hand-rolled (not _df_to_booktabs, which would escape the raw column
    names like a literal "median_train_post_s" into an ugly
    "median\\_train\\_post\\_s" header -- SPEC ask: fix the "\\_" escaping so
    headers read well). Human-written column headers, RF/GBT grouped with a
    midrule, our two shipped arms bolded."""
    ncols = 6
    if block.empty or block["mean_auc"].isna().all():
        body = r"\multicolumn{%d}{c}{(no data yet)} \\" % ncols
    else:
        lines = []
        prev_fam = None
        for _, r in block.iterrows():
            if prev_fam is not None and r["family"] != prev_fam:
                lines.append(r"\midrule")
            prev_fam = r["family"]
            bold = r["_arm"] in (RF_BASELINE, GBT_BASELINE)

            def _b(s: str) -> str:
                return f"\\textbf{{{s}}}" if bold else s

            def _num(v: float, fmt: str) -> str:
                return _b("--" if v != v else fmt.format(v))

            name = _b(_latex_escape(r["method"]))
            ratio = r["time_ratio_geo"]
            ratio_s = _b("--" if ratio != ratio else f"{ratio:.2f}$\\times$")
            cells = [name, str(int(r["n"])), _num(r["mean_auc"], "{:.4f}"),
                     _num(r["mean_acc"], "{:.4f}"), _num(r["mean_rank_auc"], "{:.2f}"), ratio_s]
            lines.append(" & ".join(cells) + r" \\")
        body = "\n".join(lines)
    header = " & ".join(["Method", "$n$", "Mean AUC", "Mean acc.", "Mean rank (AUC)",
                          "Time ratio vs.\\ ours"]) + r" \\"
    return (
        "% auto-generated by analyze.py:make_paper_tables -- regenerate, do not hand-edit\n"
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        f"{header}\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"
        "\\end{table}\n"
    )


def _huge_table_tex(large_summary: pd.DataFrame, large_raw: pd.DataFrame) -> str:
    """Rows = all 15 arms, grouped RF/GBT with a midrule; columns = 4 per
    dataset (train_s, post_s, AUC, acc). A dataset x arm cell whose raw rows
    are non-OK (OOM/TIMEOUT/ERROR -- D6: a 3h timeout on the huge datasets is
    itself a result) prints that status spanning all 4 sub-columns instead of
    a numeric "--", so e.g. EPSILON's xgboost_rf/lightgbm_rf OOMs are visible
    facts in the table, not blank cells. Our two shipped arms are bolded."""
    all_ds_seen = sorted(set(large_raw["dataset"].dropna().unique())
                          | set(large_summary["dataset"].dropna().unique()))
    preferred = [d for d in ("HIGGS", "SUSY", "EPSILON") if d in all_ds_seen]
    datasets = preferred + [d for d in all_ds_seen if d not in preferred]
    ncols = 1 + 4 * max(len(datasets), 1)
    if not datasets:
        return (
            "% auto-generated by analyze.py:make_paper_tables -- regenerate, do not hand-edit\n"
            "\\begin{table}[t]\n\\centering\n"
            "\\caption{Huge-dataset results (single held-out split)}\n"
            "\\label{tab:huge-datasets}\n\\begin{tabular}{l}\n\\toprule\n"
            "(no data yet) \\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    status_map: dict[tuple[str, str], str] = {}
    if not large_raw.empty:
        for (ds, m), s in large_raw.groupby(["dataset", "method"])["status"]:
            modes = s.mode()
            status_map[(ds, m)] = modes.iat[0] if len(modes) else "UNKNOWN"
    idx = (large_summary.set_index(["dataset", "method"])
           if not large_summary.empty else large_summary)

    def _cell(ds: str, m: str, col: str, fmt: str) -> str | None:
        if (ds, m) not in idx.index:
            return None
        v = idx.loc[(ds, m), col]
        if isinstance(v, pd.Series):
            v = v.iloc[0]
        return None if pd.isna(v) else fmt.format(v)

    align = "l" + "cccc" * len(datasets)
    header1 = ["\\multicolumn{1}{l}{}"] + [
        f"\\multicolumn{{4}}{{c}}{{{_latex_escape(ds)}}}" for ds in datasets]
    cmidrules = " ".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(datasets)))
    header2 = ["Method"] + ["Train (s)", "Post (s)", "AUC", "Acc."] * len(datasets)

    def _row(m: str) -> str:
        bold = m in (RF_BASELINE, GBT_BASELINE)
        label = _latex_escape(ARM_LABELS.get(m, m))
        cells = [f"\\textbf{{{label}}}" if bold else label]
        for ds in datasets:
            st = status_map.get((ds, m))
            if st is not None and st != "OK":
                cells.append(f"\\multicolumn{{4}}{{c}}{{{_latex_escape(st)}}}")
                continue

            def _mb(v: str | None) -> str:
                v = v if v is not None else "--"
                return f"\\textbf{{{v}}}" if bold else v

            cells.append(_mb(_cell(ds, m, "time_s_median", "{:.1f}")))
            cells.append(_mb(_cell(ds, m, "median_post_s", "{:.1f}")))
            cells.append(_mb(_cell(ds, m, "auc_mean", "{:.4f}")))
            cells.append(_mb(_cell(ds, m, "acc_mean", "{:.4f}")))
        return " & ".join(cells) + r" \\"

    lines: list[str] = []
    prev_fam = None
    for m in _family_ordered_arms():
        fam = ARM_FAMILY[m]
        if prev_fam is not None and fam != prev_fam:
            lines.append("\\midrule")
        prev_fam = fam
        lines.append(_row(m))

    return (
        "% auto-generated by analyze.py:make_paper_tables -- regenerate, do not hand-edit\n"
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Huge-dataset results (single held-out split). Post = YDF post-training "
        "finalization time, excluded from the headline Train column (see footnote).}\n"
        "\\label{tab:huge-datasets}\n"
        f"\\begin{{tabular}}{{{align}}}\n\\toprule\n"
        f"{' & '.join(header1)} \\\\\n{cmidrules}\n{' & '.join(header2)} \\\\\n\\midrule\n"
        + "\n".join(lines) +
        f"\n\\addlinespace\n\\multicolumn{{{ncols}}}{{p{{0.92\\linewidth}}}}"
        f"{{\\footnotesize {_HUGE_TABLE_FOOTNOTE}}} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def make_paper_tables(summary: pd.DataFrame, all_df: pd.DataFrame, large_df: pd.DataFrame,
                       out_dir: Path) -> list[str]:
    """(e) booktabs LaTeX tables for the paper: main summary (fully-numeric
    subset + all suite datasets), a per-dataset appendix table, and a
    huge-dataset table. Purely a function of the per-dataset summary frame
    (a) plus the raw all_df/large_df (needed for the geometric-mean speed
    ratios and the huge table's OOM/TIMEOUT/ERROR status labels), so it
    degrades to "(no data yet)" placeholder rows rather than crashing when a
    family/dataset is still missing."""
    tables_dir = out_dir / "tables"
    written: list[str] = []

    # Split on "_source" (the mode flag per_dataset_summary carries through
    # from analyze()), NOT the "suite" column -- "suite" holds the dataset's
    # own meta.json collection tag (e.g. "TabArena-v0.1"/"TabReD"/
    # "tabred_chrono", D2) for suite-mode rows, not the literal string
    # "suite" (see cross_table / analyze() for the same distinction).
    suite = summary[summary["_source"] == "suite"].copy()
    large = summary[summary["_source"] == "large"].copy()
    suite_raw = all_df[all_df["_source"] == "suite"] if "_source" in all_df.columns else all_df

    cat_by_ds = (suite.groupby("dataset")["n_categorical_encoded"].max()
                 if not suite.empty else pd.Series(dtype=float))
    fully_numeric_ds = set(cat_by_ds[cat_by_ds == 0].index) if len(cat_by_ds) else set()

    written.append(_write_tex(
        tables_dir / "table_main_summary_all_datasets.tex",
        _main_summary_tex(_main_summary_block(suite, suite_raw),
                           "Mean accuracy/AUC, mean AUC rank (within family), and geometric-mean "
                           "training-time ratio vs.\\ our arm -- all suite datasets",
                           "tab:main-summary-all")))
    fn_raw = suite_raw[suite_raw["dataset"].isin(fully_numeric_ds)]
    written.append(_write_tex(
        tables_dir / "table_main_summary_fully_numeric.tex",
        _main_summary_tex(_main_summary_block(suite[suite["dataset"].isin(fully_numeric_ds)], fn_raw),
                           "Mean accuracy/AUC, mean AUC rank (within family), and geometric-mean "
                           "training-time ratio vs.\\ our arm -- fully-numeric subset",
                           "tab:main-summary-fn")))

    appendix = suite[["dataset", "method", "n_folds_ok", "acc_mean", "auc_mean",
                       "logloss_mean", "time_s_median"]].copy()
    appendix["method"] = appendix["method"].map(lambda m: ARM_LABELS.get(m, m))
    appendix = appendix.rename(columns={
        "dataset": "Dataset", "method": "Method", "n_folds_ok": "Folds OK",
        "acc_mean": "Acc.", "auc_mean": "AUC", "logloss_mean": "Log-loss",
        "time_s_median": "Train (s)"})
    appendix = appendix.sort_values(["Dataset", "Method"])
    written.append(_write_tex(
        tables_dir / "table_per_dataset_appendix.tex",
        _df_to_booktabs(appendix,
                         "Per-dataset, per-method accuracy/AUC/log-loss and median training "
                         "time (5-fold suite)", "tab:per-dataset-appendix", align="llrrrrr")))

    large_raw = large_df[large_df["_source"] == "large"] if "_source" in large_df.columns else large_df
    written.append(_write_tex(
        tables_dir / "table_huge_datasets.tex", _huge_table_tex(large, large_raw)))

    return written


# ==========================================================================
# Orchestration
# ==========================================================================

def analyze(suite_csv: str | Path | None, large_csv: str | Path | None,
            speedup_csv: str | Path | None, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suite_df = load_csv(suite_csv, SUITE_COLUMNS)
    large_df = load_csv(large_csv, SUITE_COLUMNS)
    spm = load_csv(speedup_csv, SPEEDUP_COLUMNS)
    # Mode provenance, tracked separately from the "suite" column: run_suite.py
    # writes the dataset's own meta.json "suite" tag (e.g. "TabArena-v0.1", or
    # the literal "tabred_chrono" for D2's chronological-holdout datasets)
    # into that column for --mode suite rows, not the literal string "suite"
    # -- so "suite" collides with meta.json's own field and can't be used to
    # tell suite-mode rows from large-mode rows. "_source" is that mode flag.
    suite_df["_source"] = "suite"
    large_df["_source"] = "large"
    suite_df = add_headline_time(suite_df)
    large_df = add_headline_time(large_df)
    all_df = pd.concat([suite_df, large_df], ignore_index=True) if not large_df.empty else suite_df.copy()

    result: dict = {"csvs": {}, "figures": [], "tables": []}

    summary = per_dataset_summary(all_df)
    p = out_dir / "per_dataset_summary.csv"; summary.to_csv(p, index=False)
    result["csvs"]["per_dataset_summary"] = str(p)

    # D4: two family tables, never merged/ranked against each other.
    rf_table, gbt_table = family_tables(summary)
    p = out_dir / "family_rf_summary.csv"; rf_table.to_csv(p, index=False)
    result["csvs"]["family_rf_summary"] = str(p)
    p = out_dir / "family_gbt_summary.csv"; gbt_table.to_csv(p, index=False)
    result["csvs"]["family_gbt_summary"] = str(p)

    # D7: bit-identity report.
    bit_id = bit_identity_report(summary)
    p = out_dir / "bit_identity_stdsort_vs_hwy.csv"; bit_id.to_csv(p, index=False)
    result["csvs"]["bit_identity_stdsort_vs_hwy"] = str(p)

    # D8: the two FIXED comparison sets, Holm-corrected Wilcoxon, AUC primary
    # (tie band 0.005), one table + one win/tie/loss summary per family.
    rf_baseline, rf_comparators = FIXED_COMPARISONS["rf"]
    gbt_baseline, gbt_comparators = FIXED_COMPARISONS["gbt"]
    rf_cross = cross_table(all_df, rf_baseline, rf_comparators)
    gbt_cross = cross_table(all_df, gbt_baseline, gbt_comparators)
    p = out_dir / "cross_rf_fixed_comparisons.csv"; rf_cross.to_csv(p, index=False)
    result["csvs"]["cross_rf_fixed_comparisons"] = str(p)
    p = out_dir / "cross_gbt_fixed_comparisons.csv"; gbt_cross.to_csv(p, index=False)
    result["csvs"]["cross_gbt_fixed_comparisons"] = str(p)

    wtl_rf = win_tie_loss_summary(rf_cross, metric="test_auc")
    p = out_dir / "win_tie_loss_rf_auc.csv"; wtl_rf.to_csv(p, index=False)
    result["csvs"]["win_tie_loss_rf_auc"] = str(p)
    wtl_gbt = win_tie_loss_summary(gbt_cross, metric="test_auc")
    p = out_dir / "win_tie_loss_gbt_auc.csv"; wtl_gbt.to_csv(p, index=False)
    result["csvs"]["win_tie_loss_gbt_auc"] = str(p)

    # D8: Friedman + Nemenyi CD per family, for each of acc/auc(primary)/
    # logloss -- friedman_tests.csv/mean_rank.csv carry all three; the two CD
    # diagrams (figures) are drawn for AUC only, per family.
    rank_specs = [
        ("rf", RF_ARMS, "acc_mean", False), ("rf", RF_ARMS, "auc_mean", False),
        ("rf", RF_ARMS, "logloss_mean", True),
        ("gbt", GBT_ARMS, "acc_mean", False), ("gbt", GBT_ARMS, "auc_mean", False),
        ("gbt", GBT_ARMS, "logloss_mean", True),
    ]
    rank_rows, friedman_rows, cd_inputs = [], [], {}
    for family, methods, metric, ascending in rank_specs:
        rank_df, friedman = mean_rank_table(summary, methods, metric, ascending)
        if metric == "auc_mean":
            cd_inputs[family] = (rank_df.copy(), friedman)
        rank_df.insert(0, "metric", metric); rank_df.insert(0, "family", family)
        rank_rows.append(rank_df)
        friedman_rows.append({"family": family, "metric": metric, **friedman})
    rank_out = pd.concat(rank_rows, ignore_index=True)
    p = out_dir / "mean_rank.csv"; rank_out.to_csv(p, index=False)
    result["csvs"]["mean_rank"] = str(p)
    friedman_out = pd.DataFrame(friedman_rows)
    p = out_dir / "friedman_tests.csv"; friedman_out.to_csv(p, index=False)
    result["csvs"]["friedman_tests"] = str(p)

    speed = speed_ratio_table(all_df, SPEED_BASELINE)
    p = out_dir / "speed_ratios.csv"; speed.to_csv(p, index=False)
    result["csvs"]["speed_ratios"] = str(p)

    fig_dir = out_dir / "figures"
    result["figures"] += fig_huge_pareto(large_df, fig_dir)
    result["figures"] += fig_huge_pareto_rf(large_df, fig_dir)
    result["figures"] += fig_speedup_vs_rows(suite_df, large_df, fig_dir)
    result["figures"] += fig_heatmap_rows_depth(spm, fig_dir)
    result["figures"] += fig_speedup_vs_depth(spm, fig_dir)
    result["figures"] += fig_speedup_vs_min_examples(spm, fig_dir)
    result["figures"] += fig_trunk_width(spm, fig_dir)
    result["figures"] += fig_gbt_depth(spm, fig_dir)
    if "rf" in cd_inputs:
        rank_df, friedman = cd_inputs["rf"]
        result["figures"] += fig_cd_diagram(rank_df, friedman, fig_dir, "fig_cd_rf_auc",
                                             "Nemenyi CD diagram -- RF family (AUC)")
    if "gbt" in cd_inputs:
        rank_df, friedman = cd_inputs["gbt"]
        result["figures"] += fig_cd_diagram(rank_df, friedman, fig_dir, "fig_cd_gbt_auc",
                                             "Nemenyi CD diagram -- GBT family (AUC)")

    result["tables"] += make_paper_tables(summary, all_df, large_df, out_dir)
    return result


def fabricate_selftest_data(root: Path, seed: int = 0) -> tuple[Path, Path, Path]:
    """Synthetic suite/large/speedup-map CSVs, exact SPEC.md/PROTOCOL.md
    schemas: all 15 arms, ~6 datasets (5 suite incl. one tabred_chrono
    single-fold holdout + 1 huge), 5 folds (1 for tabred_chrono), plausible
    numbers, rep/seed repeat rows (D12), a TIMEOUT row (D6), B4/B5 speedup-map
    cells, and a handful of deliberately missing/ERROR cells so --selftest
    also exercises the partial-data paths. Used only by --selftest."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)

    def base_row(dataset, suite, rows, features, ncat, fold, rep, method, ts,
                 seed_val: int | None = None) -> dict:
        family = ARM_FAMILY[method]
        return {
            "dataset": dataset, "suite": suite, "rows": rows, "features": features,
            "n_categorical_encoded": ncat, "nan_cells_imputed": 0 if ncat == 0 else 17,
            "fold": fold, "rep": rep, "method": method, "family": family,
            "engine": ARM_ENGINE[method],
            "engine_version": "c80ffbf7" if ARM_ENGINE[method] == "ydf_fork" else "1.0.0",
            "binary": ARM_BINARY.get(method, ""),
            "trees": 240 if family == "rf" else 300,
            "max_depth": -1 if family == "rf" else 6,
            "min_examples": 1 if family == "rf" else 5,
            "threads": 48,
            "seed": seed_val if seed_val is not None else (1 if ARM_ENGINE[method] == "ydf_fork" else 0),
            "train_rows": int(rows * 0.8), "test_rows": rows - int(rows * 0.8),
            "timestamp_utc": ts, "cmd": f"selftest synthetic row: {method} {dataset} fold={fold}",
        }

    ts = "2026-09-03T00:00:00Z"
    # "collection" mimics run_suite.py's `meta.get("suite", "suite")`, which
    # writes the dataset's OWN meta.json "suite" tag into the "suite" column
    # for --mode suite rows -- never the literal string "suite". "toy_d"
    # below uses the literal "tabred_chrono" (D2's real value for a
    # chronological-holdout dataset), single fold only, so --selftest
    # exercises that path too.
    datasets = [("toy_a", 768, 8, 0, "TabArena-v0.1"), ("toy_b", 1200, 15, 0, "TabArena-v0.1"),
                ("toy_c", 5000, 30, 3, "TabArena-v0.1"), ("toy_d", 20000, 60, 0, "tabred_chrono"),
                ("toy_e", 90000, 120, 5, "TabReD")]
    arm_bias = {a: rng.normal(0, 0.02) for a in ALL_ARMS}
    # D7 selftest: spo_rf_exact_stdsort and spo_rf_exact_hwy must be
    # bit-identical (same accuracy/AUC/logloss per fold) on at least one
    # dataset for bit_identity_report to have something to find. Cache the
    # first of the pair computed per (dataset, fold) and reuse it for the
    # second rather than drawing independent numbers.
    bitid_cache: dict[tuple, tuple[float, float, float]] = {}
    suite_rows = []
    for name, rows, feats, ncat, collection in datasets:
        base_time = 0.02 * (rows / 1000) * (feats / 20)
        n_folds = 1 if collection == "tabred_chrono" else 5
        for fold in range(n_folds):
            for method in ALL_ARMS:
                if name == "toy_c" and method == "aa_gbt_exact" and fold == 3:
                    continue  # simulate a run the driver never produced
                row = base_row(name, collection, rows, feats, ncat, fold, 0, method, ts)
                if name == "toy_e" and method == "catboost" and fold == 1:
                    row.update(status="OOM", train_s=np.nan, pre_train_s=np.nan,
                               train_post_s=np.nan, test_acc=np.nan, test_auc=np.nan,
                               test_logloss=np.nan)
                else:
                    mult = 1.0 if row["family"] == "rf" else 1.3
                    train_s = max(0.001, base_time * mult * row["trees"] / 240
                                  * (1 + rng.normal(0, 0.05)))
                    bitid_key = (name, fold)
                    if method in (EXACT_STDSORT, EXACT_HWY) and bitid_key in bitid_cache:
                        acc, auc, ll = bitid_cache[bitid_key]
                    else:
                        acc = float(np.clip(0.80 + arm_bias[method] + rng.normal(0, 0.01), 0.5, 0.999))
                        auc = float(np.clip(acc + 0.08 + rng.normal(0, 0.01), 0.5, 0.999))
                        ll = float(np.clip(0.6 - 0.4 * (acc - 0.5) + abs(rng.normal(0, 0.01)), 0.02, 1.5))
                        if method in (EXACT_STDSORT, EXACT_HWY):
                            bitid_cache[bitid_key] = (acc, auc, ll)
                    row.update(status="OK", train_s=round(train_s, 6),
                               pre_train_s=round(train_s * 0.05, 6),
                               train_post_s=round(train_s * 1.02, 6),
                               test_acc=acc, test_auc=auc, test_logloss=ll)
                suite_rows.append(row)
    suite_df = pd.DataFrame(suite_rows, columns=SUITE_COLUMNS)
    suite_path = root / "suite_results.csv"
    suite_df.to_csv(suite_path, index=False)

    large_rows = []
    huge_name, huge_rows, huge_feats = "huge_x", 3_000_000, 28
    for method in ALL_ARMS:
        if method == "spo_gbt_exact_hwy":
            continue  # missing arm on the huge dataset (still running)
        row = base_row(huge_name, "large", huge_rows, huge_feats, 0, 0, 0, method, ts)
        if method == "aa_gbt_exact":
            row.update(status="ERROR", train_s=np.nan, pre_train_s=np.nan,
                       train_post_s=np.nan, test_acc=np.nan, test_auc=np.nan,
                       test_logloss=np.nan)
        elif method == "lightgbm_rf":
            # D6: a huge-dataset run that hits the 3h timeout is a result,
            # not a crash.
            row.update(status="TIMEOUT", train_s=np.nan, pre_train_s=np.nan,
                       train_post_s=np.nan, test_acc=np.nan, test_auc=np.nan,
                       test_logloss=np.nan)
        else:
            mult = {"rf": 1.0, "gbt": 1.3}[row["family"]]
            train_s = max(1.0, 40.0 * mult * (1 + rng.normal(0, 0.03))
                          * (0.4 if "dyn" in method else 1.0))
            acc = float(np.clip(0.83 + arm_bias[method] + rng.normal(0, 0.003), 0.5, 0.999))
            auc = float(np.clip(acc + 0.07, 0.5, 0.999))
            ll = float(np.clip(0.55 - 0.4 * (acc - 0.5), 0.02, 1.5))
            row.update(status="OK", train_s=round(train_s, 6),
                       pre_train_s=round(train_s * 0.02, 6),
                       train_post_s=round(train_s * 1.01, 6),
                       test_acc=acc, test_auc=auc, test_logloss=ll)
        large_rows.append(row)
    # D12a: timing repeat (rep=1) for a headline arm; D12b: a second seed
    # (seed=2) for a YDF RF arm -- exercise the rep/seed columns beyond the
    # constant 0/1 every other fabricated row uses.
    rep_row = base_row(huge_name, "large", huge_rows, huge_feats, 0, 0, 1, SPEED_BASELINE, ts)
    rep_row.update(status="OK", train_s=round(16.5 * (1 + rng.normal(0, 0.01)), 6),
                   pre_train_s=0.3, train_post_s=round(16.7 * (1 + rng.normal(0, 0.01)), 6),
                   test_acc=0.86, test_auc=0.93, test_logloss=0.32)
    large_rows.append(rep_row)
    seed_row = base_row(huge_name, "large", huge_rows, huge_feats, 0, 0, 0, EXACT_HWY, ts,
                         seed_val=2)
    seed_row.update(status="OK", train_s=round(34.0 * (1 + rng.normal(0, 0.01)), 6),
                   pre_train_s=0.3, train_post_s=round(34.3 * (1 + rng.normal(0, 0.01)), 6),
                   test_acc=0.855, test_auc=0.925, test_logloss=0.33)
    large_rows.append(seed_row)
    large_df = pd.DataFrame(large_rows, columns=SUITE_COLUMNS)
    large_path = root / "large_results.csv"
    large_df.to_csv(large_path, index=False)

    def speed_row(dataset, rows, features, depth, min_ex, arm, family, trees, ts) -> dict:
        return {"dataset": dataset, "rows": rows, "features": features, "arm": arm,
                "family": family, "binary": ARM_BINARY.get(arm, ""),
                "trees": trees, "max_depth": depth, "min_examples": min_ex, "threads": 48,
                "seed": 1, "pre_train_s": round(0.3 * rows / 1e6, 6),
                "timestamp_utc": ts, "cmd": f"selftest synthetic: {arm} {dataset} depth={depth}"}

    spm_rows = []
    b1_rows = [100_000, 300_000, 1_000_000]
    b1_depths = [6, 10, 15, 20, -1]
    b1_core_arms = [EXACT_HWY, SPEED_BASELINE]
    b1_extra_arms = [EXACT_STDSORT, "spo_rf_rand_vec", "spo_rf_rand_scalar", "spo_rf_dyn_scalar"]
    for rows in b1_rows:
        depth_factor = {6: 0.3, 10: 0.55, 15: 0.85, 20: 0.97, -1: 1.0}
        for depth in b1_depths:
            for arm in b1_core_arms + (b1_extra_arms if depth == -1 else []):
                row = speed_row(f"higgs_{rows}", rows, 28, depth, 1, arm, "rf", 240, ts)
                base_t = 8.0 * (rows / 1e6) * depth_factor[depth]
                speed_mult = {EXACT_STDSORT: 2.6, EXACT_HWY: 2.1, "spo_rf_rand_vec": 1.15,
                              "spo_rf_rand_scalar": 1.5, "spo_rf_dyn_scalar": 1.3,
                              SPEED_BASELINE: 1.0}[arm]
                if rows == 1_000_000 and depth == 15 and arm == EXACT_HWY:
                    row.update(status="ERROR", train_s=np.nan)  # a missing cell
                else:
                    row.update(status="OK",
                               train_s=round(max(0.05, base_t * speed_mult
                                                  * (1 + rng.normal(0, 0.02))), 6))
                spm_rows.append(row)
    b2_rows = [1_000_000, 3_000_000]
    b2_min_ex = [1, 5, 20, 100]
    for rows in b2_rows:
        for min_ex in b2_min_ex:
            stop_relief = {1: 1.0, 5: 0.9, 20: 0.7, 100: 0.5}[min_ex]
            for arm in (EXACT_HWY, SPEED_BASELINE):
                row = speed_row(f"higgs_{rows}", rows, 28, -1, min_ex, arm, "rf", 240, ts)
                base_t = 8.0 * (rows / 1e6) * stop_relief
                speed_mult = {EXACT_HWY: 2.1, SPEED_BASELINE: 1.0}[arm]
                row.update(status="OK",
                           train_s=round(max(0.05, base_t * speed_mult
                                              * (1 + rng.normal(0, 0.02))), 6))
                spm_rows.append(row)
    # B4 (D9): GBT depth axis, HIGGS-1M + GiveMeSomeCredit.
    b4_datasets = [("higgs_1000000", 1_000_000, 28), ("GiveMeSomeCredit", 150_000, 10)]
    b4_depths = [6, 10, -1]
    b4_depth_factor = {6: 1.0, 10: 1.6, -1: 2.1}
    b4_speed_mult = {"spo_gbt_exact_hwy": 1.8, "spo_gbt_dyn_vec": 1.0}
    for name, rows, feats in b4_datasets:
        for depth in b4_depths:
            for arm, mult in b4_speed_mult.items():
                base_t = 5.0 * max(rows / 1e6, 0.15) * b4_depth_factor[depth]
                row = speed_row(name, rows, feats, depth, 5, arm, "gbt", 300, ts)
                row.update(status="OK",
                           train_s=round(max(0.05, base_t * mult * (1 + rng.normal(0, 0.02))), 6))
                spm_rows.append(row)
    # B5 (D10): trunk-generator feature-width axis.
    b5_cols = [32, 128, 512, 2048, 8192]
    b5_speed_mult = {EXACT_STDSORT: 2.4, EXACT_HWY: 2.0, SPEED_BASELINE: 1.0}
    for cols in b5_cols:
        base_t = 6.0 * (1 + cols / 2000.0)
        for arm, mult in b5_speed_mult.items():
            row = speed_row(f"trunk_1000000_x_{cols}", 1_000_000, cols, -1, 1, arm, "rf", 240, ts)
            row.update(status="OK",
                       train_s=round(max(0.05, base_t * mult * (1 + rng.normal(0, 0.02))), 6))
            spm_rows.append(row)
    spm_df = pd.DataFrame(spm_rows, columns=SPEEDUP_COLUMNS)
    spm_path = root / "speedup_map.csv"
    spm_df.to_csv(spm_path, index=False)

    return suite_path, large_path, spm_path


def run_selftest(root: Path) -> int:
    print(f"[selftest] fabricating synthetic CSVs under {root}", flush=True)
    assert len(ALL_ARMS) == 15, f"expected the 15-arm SPEC.md roster, got {len(ALL_ARMS)}: {ALL_ARMS}"
    suite_path, large_path, spm_path = fabricate_selftest_data(root)
    out_dir = root / "analysis_out"
    result = analyze(suite_path, large_path, spm_path, out_dir)

    expected = (list(result["csvs"].values()) + result["figures"] + result["tables"])
    missing = [p for p in expected if not Path(p).is_file() or Path(p).stat().st_size == 0]
    print(f"[selftest] {len(expected)} outputs expected, {len(missing)} missing/empty", flush=True)
    for p in expected:
        print(f"  {'OK ' if p not in missing else 'MISS'} {p}")
    if missing:
        print(f"[selftest] FAIL: missing outputs: {missing}", file=sys.stderr)
        return 1

    # A couple of content sanity checks beyond "the file exists".
    rf_cross = pd.read_csv(result["csvs"]["cross_rf_fixed_comparisons"])
    assert not rf_cross.empty, "RF fixed-comparison table is empty on fabricated data"
    assert set(rf_cross["comparator"]) <= set(FIXED_COMPARISONS["rf"][1])
    gbt_cross = pd.read_csv(result["csvs"]["cross_gbt_fixed_comparisons"])
    assert not gbt_cross.empty, "GBT fixed-comparison table is empty on fabricated data"
    assert set(gbt_cross["comparator"]) <= set(FIXED_COMPARISONS["gbt"][1])
    bit_id = pd.read_csv(result["csvs"]["bit_identity_stdsort_vs_hwy"])
    assert not bit_id.empty and bool(bit_id["bit_identical"].any()), \
        "expected at least one bit-identical dataset in fabricated data (D7 selftest)"
    speed = pd.read_csv(result["csvs"]["speed_ratios"])
    assert (speed["dataset"] == "GEOMEAN_ALL").any(), "speed_ratios missing GEOMEAN_ALL row"
    assert speed["dataset"].str.startswith("GEOMEAN_HEADLINE").any(), \
        "speed_ratios missing GEOMEAN_HEADLINE row"
    friedman = pd.read_csv(result["csvs"]["friedman_tests"])
    assert "cd_nemenyi_0.05" in friedman.columns
    summary = pd.read_csv(result["csvs"]["per_dataset_summary"])
    assert summary["dataset"].nunique() >= 5, "expected ~6 fabricated datasets"
    assert set(summary["method"].unique()) == set(ALL_ARMS), \
        "expected all 15 fabricated arms to appear in per_dataset_summary"
    assert (summary["suite"] == "tabred_chrono").any(), \
        "expected a tabred_chrono (D2) dataset in fabricated data"
    large_raw = pd.read_csv(large_path)
    assert (large_raw["rep"] == 1).any(), "expected a rep=1 (D12a) row in fabricated large data"
    assert (large_raw["seed"] == 2).any(), "expected a seed=2 (D12b) row in fabricated large data"
    assert (large_raw["status"] == "TIMEOUT").any(), "expected a TIMEOUT (D6) row"
    spm_raw = pd.read_csv(spm_path)
    assert spm_raw["dataset"].astype(str).str.startswith("trunk_").any(), \
        "expected trunk_* (B5) rows in fabricated speedup_map data"
    assert (spm_raw["family"] == "gbt").any(), "expected GBT-depth (B4) rows in fabricated speedup_map data"

    print("[selftest] PASS", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite-csv", default="/home/ubuntu/spo_vs_gbt/results/suite_results.csv")
    ap.add_argument("--large-csv", default="/home/ubuntu/spo_vs_gbt/results/large_results.csv")
    ap.add_argument("--speedup-csv", default="/home/ubuntu/spo_vs_gbt/results/speedup_map.csv")
    ap.add_argument("--out-dir", default="/home/ubuntu/spo_vs_gbt/results/analysis")
    ap.add_argument("--selftest", action="store_true",
                     help="fabricate synthetic CSVs under "
                          "/home/ubuntu/spo_vs_gbt/results/selftest/ and exercise every output")
    args = ap.parse_args()

    if args.selftest:
        overridden = [f"--{dest.replace('_', '-')}" for dest in
                      ("suite_csv", "large_csv", "speedup_csv", "out_dir")
                      if getattr(args, dest) != ap.get_default(dest)]
        if overridden:
            print(f"[selftest] ignoring {', '.join(overridden)} -- --selftest "
                  "always analyzes its own fabricated data under "
                  "/home/ubuntu/spo_vs_gbt/results/selftest/, not real CSVs",
                  file=sys.stderr)
        return run_selftest(Path("/home/ubuntu/spo_vs_gbt/results/selftest"))

    result = analyze(args.suite_csv, args.large_csv, args.speedup_csv, args.out_dir)
    for kind, items in result.items():
        n = len(items) if isinstance(items, list) else len(items)
        print(f"{kind}: {n} file(s) written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
