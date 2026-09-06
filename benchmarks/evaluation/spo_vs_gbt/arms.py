"""ARMS table for the SPO-vs-GBT study (SPEC.md "Arms" section / PROTOCOL.md "Methods").

Shared between Driver 1 (run_suite.py: accuracy + timing over all 15 arms) and
Driver 2 (run_speedup_map.py: YDF-fork-only timing grid, filtered to
engine == "ydf_fork"). run_speedup_map.py already depends on the exact key
names family/engine/binary/ydf_flags/trees/tree_depth -- do not rename them.
Extra keys (min_examples, py_ctor) are ignored by that driver.

Each arm is a dict:
  family       "rf" | "gbt"
  engine       "ydf_fork" | "xgboost" | "lightgbm" | "catboost"
  binary       basename of the harness binary under --bin-dir (ydf_fork arms
               only: "default" | "exact_std_sort" | "scalar"), else None.
  ydf_flags    list[str] of harness flags beyond the ones the driver always
               sets itself (--input_mode/--train_csv/--test_csv/--label_col/
               --num_trees/--tree_depth/--min_examples/--num_threads/--seed/
               --compute_oob_performances=false). Empty for python arms.
  trees        int, num_trees / n_estimators.
  tree_depth   int, --tree_depth (-1 = unlimited). Recorded for python arms
               too (mirrors what py_ctor bakes in -- keep the two in sync).
  min_examples int | None: explicit --min_examples for ydf_fork arms, or None
               to mean "the harness default" (1 for Bagging/rf, 5 for
               Boosting/gbt) -- callers resolve None themselves. For python
               arms this is an informational value for the results row only
               (e.g. LightGBM's min_child_samples); None if not applicable.
  py_ctor      callable(threads: int, n_features: int, seed: int) ->
               unfitted sklearn-style estimator, or None for ydf_fork arms.
               Called fresh per (dataset, fold, rep) so no fitted state leaks
               across runs, and once more (untimed, on a slice) for
               run_suite.py's A6 warm-up. n_features is X.shape[1] of the
               fold's training matrix, needed by the RF-family arms below to
               compute their feature-fraction hyperparameter at fit time
               (D6); the GBT arms ignore it. Imports its library lazily so
               importing this module never requires xgboost/lightgbm/
               catboost to be installed.

Seeds: SPEC.md v2 addendum A5 replaces the old split (ydf_fork seed=1
default, python seed hard-coded 0) with a single --seed CLI flag on
run_suite.py (default 1) that sets both: the harness's --seed and every
python arm's random_state/random_seed (threaded through here as py_ctor's
`seed` argument, not hard-coded). --seed can be overridden on the CLI --
PROTOCOL.md D12(b)'s second-seed accuracy-variance runs pass --seed 2.
"""
from __future__ import annotations

import math

# SPEC.md v1 said "min_child_samples: make it a single constant at the top,
# default 20", but that left LightGBM's leaf-size floor 4x looser than every
# ydf_fork RF/GBT arm (min_examples=5) and xgboost/catboost (no comparable
# floor set), an axis reviewers flagged as capable of making a speed/accuracy
# comparison unfair without a matched-capacity justification. SPEC.md's v2
# addendum (A1, "v2 decisions" D5) resolves this: 5, matching the ydf_fork
# GBT arms' min_examples. (The RF-family lightgbm_rf arm below uses its own
# D6-mandated min_child_samples=1 instead -- that arm matches XGBRFClassifier's
# no-comparable-floor default, not the GBT arms.)
LIGHTGBM_MIN_CHILD_SAMPLES = 5


def _rf_feature_fraction(n_features: int) -> float:
    """sqrt(F)/F clipped to [1/F, 1] (PROTOCOL.md D6) -- the per-split feature
    sampling rate that stands in for YDF RF's mtry=ceil(sqrt(F)) when the
    library only exposes a *fraction* knob (colsample_bynode /
    feature_fraction_bynode). Computed at fit time from the fold's actual
    X.shape[1], not baked in per-dataset.
    """
    if n_features <= 0:
        return 1.0
    frac = math.sqrt(n_features) / n_features
    return max(1.0 / n_features, min(1.0, frac))


def _xgb_ctor(threads: int, n_features: int, seed: int):
    del n_features  # GBT arm: no per-node feature subsampling knob here
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist",
        n_jobs=threads, objective="binary:logistic", random_state=seed,
    )


def _lgbm_ctor(threads: int, n_features: int, seed: int):
    del n_features
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=300, max_depth=6, num_leaves=63, learning_rate=0.1,
        n_jobs=threads, random_state=seed, verbose=-1,
        min_child_samples=LIGHTGBM_MIN_CHILD_SAMPLES,
    )


def _catboost_ctor(threads: int, n_features: int, seed: int):
    del n_features
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.1, thread_count=threads,
        random_seed=seed, verbose=0, allow_writing_files=False,
        # D5: CatBoost 1.2.10's CPU default bootstrap is MVS (subsample 0.8);
        # "No" makes "no subsampling" actually true for every GBT arm.
        bootstrap_type="No",
    )


def _xgb_rf_ctor(threads: int, n_features: int, seed: int):
    """D6 RF-family baseline. XGBoost has no bootstrap-with-replacement mode;
    subsample=0.632 (row sampling without replacement) is the usual stand-in
    for RF's ~63.2% distinct-row bootstrap (disclosed, not hidden).
    max_depth=0 (unlimited) is verified to work with tree_method="hist".
    """
    from xgboost import XGBRFClassifier
    return XGBRFClassifier(
        n_estimators=240, max_depth=0, subsample=0.632,
        colsample_bynode=_rf_feature_fraction(n_features), learning_rate=1.0,
        tree_method="hist", n_jobs=threads, random_state=seed,
    )


def _lgbm_rf_ctor(threads: int, n_features: int, seed: int):
    """D6 RF-family baseline. num_leaves=131071 is LightGBM's library-enforced
    maximum (verified) -- the closest available stand-in for RF's unlimited
    depth/purity growth. min_child_samples=1 here (not
    LIGHTGBM_MIN_CHILD_SAMPLES=5) per D6: this arm is compared against
    ydf_fork's min_examples=1 RF arms, not the GBT arms.
    """
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        boosting_type="rf", n_estimators=240, num_leaves=131071, max_depth=-1,
        min_child_samples=1, bagging_fraction=0.632, bagging_freq=1,
        feature_fraction_bynode=_rf_feature_fraction(n_features),
        n_jobs=threads, random_state=seed, verbose=-1,
    )


_OBLIQUE = ["--feature_split_type", "Oblique"]
_AXIS_ALIGNED = ["--feature_split_type", "Axis Aligned"]
_EXACT = ["--numerical_split_type", "Exact"]
_RANDOM64 = ["--numerical_split_type", "Random", "--histogram_num_bins", "64"]
_DYN64 = ["--numerical_split_type", "Dynamic Random Histogram",
          "--histogram_num_bins", "64", "--dynamic_split_threshold", "250"]
_BOOSTING = ["--ensemble_method", "Boosting"]

ARMS: dict[str, dict] = {
    # ---- RF family: 240 trees, unlimited depth, min_examples 1 (purity) ----
    "spo_rf_exact_stdsort": {
        "family": "rf", "engine": "ydf_fork", "binary": "exact_std_sort",
        "ydf_flags": _OBLIQUE + _EXACT,
        "trees": 240, "tree_depth": -1, "min_examples": None, "py_ctor": None,
    },
    "spo_rf_exact_hwy": {
        "family": "rf", "engine": "ydf_fork", "binary": "default",
        "ydf_flags": _OBLIQUE + _EXACT,
        "trees": 240, "tree_depth": -1, "min_examples": None, "py_ctor": None,
    },
    "spo_rf_rand_scalar": {
        "family": "rf", "engine": "ydf_fork", "binary": "scalar",
        "ydf_flags": _OBLIQUE + _RANDOM64,
        "trees": 240, "tree_depth": -1, "min_examples": None, "py_ctor": None,
    },
    "spo_rf_rand_vec": {
        "family": "rf", "engine": "ydf_fork", "binary": "default",
        "ydf_flags": _OBLIQUE + _RANDOM64,
        "trees": 240, "tree_depth": -1, "min_examples": None, "py_ctor": None,
    },
    "spo_rf_dyn_scalar": {
        "family": "rf", "engine": "ydf_fork", "binary": "scalar",
        "ydf_flags": _OBLIQUE + _DYN64,
        "trees": 240, "tree_depth": -1, "min_examples": None, "py_ctor": None,
    },
    "spo_rf_dyn_vec": {
        "family": "rf", "engine": "ydf_fork", "binary": "default",
        "ydf_flags": _OBLIQUE + _DYN64,
        "trees": 240, "tree_depth": -1, "min_examples": None, "py_ctor": None,
    },
    "aa_rf_exact": {
        "family": "rf", "engine": "ydf_fork", "binary": "default",
        "ydf_flags": _AXIS_ALIGNED + _EXACT,
        "trees": 240, "tree_depth": -1, "min_examples": None, "py_ctor": None,
    },
    # ---- RF-family external baselines (D6). Last within the RF family: on
    # huge datasets these may hit the run timeout, so they run after every
    # ydf_fork RF arm has already recorded its row.
    "xgboost_rf": {
        "family": "rf", "engine": "xgboost", "binary": None, "ydf_flags": [],
        "trees": 240, "tree_depth": -1, "min_examples": None,
        "py_ctor": _xgb_rf_ctor,
    },
    "lightgbm_rf": {
        "family": "rf", "engine": "lightgbm", "binary": None, "ydf_flags": [],
        "trees": 240, "tree_depth": -1, "min_examples": 1,
        "py_ctor": _lgbm_rf_ctor,
    },
    # ---- GBT family: 300 trees, depth 6, min_examples 5 (harness default) --
    "spo_gbt_exact_hwy": {
        "family": "gbt", "engine": "ydf_fork", "binary": "default",
        "ydf_flags": _OBLIQUE + _EXACT + _BOOSTING,
        "trees": 300, "tree_depth": 6, "min_examples": None, "py_ctor": None,
    },
    "spo_gbt_dyn_vec": {
        "family": "gbt", "engine": "ydf_fork", "binary": "default",
        "ydf_flags": _OBLIQUE + _DYN64 + _BOOSTING,
        "trees": 300, "tree_depth": 6, "min_examples": None, "py_ctor": None,
    },
    "aa_gbt_exact": {
        "family": "gbt", "engine": "ydf_fork", "binary": "default",
        "ydf_flags": _AXIS_ALIGNED + _EXACT + _BOOSTING
        + ["--num_candidate_attributes", "-1"],
        "trees": 300, "tree_depth": 6, "min_examples": None, "py_ctor": None,
    },
    "xgboost": {
        "family": "gbt", "engine": "xgboost", "binary": None, "ydf_flags": [],
        "trees": 300, "tree_depth": 6, "min_examples": None, "py_ctor": _xgb_ctor,
    },
    "lightgbm": {
        "family": "gbt", "engine": "lightgbm", "binary": None, "ydf_flags": [],
        "trees": 300, "tree_depth": 6,
        "min_examples": LIGHTGBM_MIN_CHILD_SAMPLES, "py_ctor": _lgbm_ctor,
    },
    "catboost": {
        "family": "gbt", "engine": "catboost", "binary": None, "ydf_flags": [],
        "trees": 300, "tree_depth": 6, "min_examples": None, "py_ctor": _catboost_ctor,
    },
}

# Canonical "listed order" (SPEC.md Arms table + v2 addendum A1: 15 arms,
# RF family first then GBT family, xgboost_rf/lightgbm_rf last within the RF
# family) that run_suite.py's --arms default and its "run order: ... arm in
# arms (listed order)" use.
ARM_ORDER: list[str] = list(ARMS.keys())
