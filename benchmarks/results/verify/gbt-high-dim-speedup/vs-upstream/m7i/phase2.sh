#!/usr/bin/env bash
# Phase 2 (m7i): same-regime small shape 15k x 40k quick pass (both targets), then the
# full protocol (300 trees, 3 runs, accuracy at full size) on the default list with
# 15k x 400k replaced by 15k x 40k (15k x 400k arm A = 2 h/run; left to the user).
set -uo pipefail
cd /home/ubuntu/yggdrasil-oblique-forests
V=/home/ubuntu/verify_gbt_hd
FORK_ARGS='--ensemble_method Boosting --numerical_split_type "Dynamic Random Histogram"'
UP_ARGS='--ensemble_method Boosting --numerical_split_type "Exact"'
CFG="--config=skip_dead_axis_jobs"
FULL_TRUNKS="1500000|4096 150000|40000 15000|40000"
ph() { echo "=== $(date -u +%FT%TZ) $*"; }
rt() { # rt <suffix> <runs> ; env: EXTRA_BAZEL_CONFIGS EXTRA_TRAIN_ARGS NUM_TREES_DIVISOR CSV/TRUNK overrides
  bash benchmarks/evaluation/runtime.sh --runs=$2 "$1" > "$V/$1.log" 2>&1; echo "rt $1 exit $?"; }
acc() { bash benchmarks/evaluation/accuracy.sh "$1" > "$V/acc_$1.log" 2>&1; echo "acc $1 exit $?"; }

# ---- small shape, upstream (tree currently on the fix branch) ----
export NUM_TREES_DIVISOR=10 CSV_DATASETS_OVERRIDE=none TRUNK_DATASETS_OVERRIDE="15000|40000"
git checkout -q upstream-main-benchmarks || exit 1
ph "SMALL upstream A"; EXTRA_BAZEL_CONFIGS="" EXTRA_TRAIN_ARGS="$UP_ARGS" rt m7i_upb_gbt_hd_a_exact_t30_r1_small 1
git checkout -q upstream-main-benchmarks-gbt-high-dim-fix || exit 1
ph "SMALL upstream B"; EXTRA_BAZEL_CONFIGS="$CFG" EXTRA_TRAIN_ARGS="$UP_ARGS" rt m7i_upb_gbt_hd_b_exact_t30_r1_small 1
# ---- small shape, fork ----
git checkout -q gbt-high-dim-speedup || exit 1
ph "SMALL fork A"; EXTRA_BAZEL_CONFIGS="" EXTRA_TRAIN_ARGS="$FORK_ARGS" rt m7i_gbt_hd_a_dyn_t30_r1_small 1
ph "SMALL fork B"; EXTRA_BAZEL_CONFIGS="$CFG" EXTRA_TRAIN_ARGS="$FORK_ARGS" rt m7i_gbt_hd_b_dyn_t30_r1_small 1

# ---- full protocol, fork ----
export NUM_TREES_DIVISOR=1; unset CSV_DATASETS_OVERRIDE; export TRUNK_DATASETS_OVERRIDE="$FULL_TRUNKS"
ph "FULL fork B accuracy"; EXTRA_BAZEL_CONFIGS="$CFG" EXTRA_TRAIN_ARGS="$FORK_ARGS" acc m7i_gbt_hd_b_dyn_t300
ph "FULL fork B runtime";  EXTRA_BAZEL_CONFIGS="$CFG" EXTRA_TRAIN_ARGS="$FORK_ARGS" rt m7i_gbt_hd_b_dyn_t300_r3 3
ph "FULL fork A accuracy"; EXTRA_BAZEL_CONFIGS="" EXTRA_TRAIN_ARGS="$FORK_ARGS" acc m7i_gbt_hd_a_dyn_t300
ph "FULL fork A runtime";  EXTRA_BAZEL_CONFIGS="" EXTRA_TRAIN_ARGS="$FORK_ARGS" rt m7i_gbt_hd_a_dyn_t300_r3 3
# ---- full protocol, upstream ----
git checkout -q upstream-main-benchmarks || exit 1
ph "FULL upstream A accuracy"; EXTRA_BAZEL_CONFIGS="" EXTRA_TRAIN_ARGS="$UP_ARGS" acc m7i_upb_gbt_hd_a_exact_t300
ph "FULL upstream A runtime";  EXTRA_BAZEL_CONFIGS="" EXTRA_TRAIN_ARGS="$UP_ARGS" rt m7i_upb_gbt_hd_a_exact_t300_r3 3
git checkout -q upstream-main-benchmarks-gbt-high-dim-fix || exit 1
ph "FULL upstream B accuracy"; EXTRA_BAZEL_CONFIGS="$CFG" EXTRA_TRAIN_ARGS="$UP_ARGS" acc m7i_upb_gbt_hd_b_exact_t300
ph "FULL upstream B runtime";  EXTRA_BAZEL_CONFIGS="$CFG" EXTRA_TRAIN_ARGS="$UP_ARGS" rt m7i_upb_gbt_hd_b_exact_t300_r3 3
# ---- 15k x 400k, upstream only, full size, ONE run per arm (user decision 2026-09-07) ----
export CSV_DATASETS_OVERRIDE=none TRUNK_DATASETS_OVERRIDE="15000|400000"
ph "HUGE upstream B runtime (1 run)"; EXTRA_BAZEL_CONFIGS="$CFG" EXTRA_TRAIN_ARGS="$UP_ARGS" rt m7i_upb_gbt_hd_b_exact_t300_r1_huge 1
git checkout -q upstream-main-benchmarks || exit 1
ph "HUGE upstream A runtime (1 run, ~2 h)"; EXTRA_BAZEL_CONFIGS="" EXTRA_TRAIN_ARGS="$UP_ARGS" rt m7i_upb_gbt_hd_a_exact_t300_r1_huge 1
git checkout -q gbt-high-dim-speedup
ph DONE
