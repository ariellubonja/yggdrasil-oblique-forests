#!/usr/bin/env bash
# Upstream quick pass: upstream-main-benchmarks (arm A) vs upstream-main-benchmarks-gbt-high-dim-fix (+config, arm B), m7i.
set -uo pipefail
cd /home/ubuntu/yggdrasil-oblique-forests
V=/home/ubuntu/verify_gbt_hd
S=/home/ubuntu/verify_gbt_hd/skill_scripts     # copied from the fork branch (absent on upstream branches)
export EXTRA_TRAIN_ARGS='--ensemble_method Boosting --numerical_split_type "Exact"'
export NUM_TREES_DIVISOR=10
git checkout -q upstream-main-benchmarks-gbt-high-dim-fix || exit 1
echo "=== $(date -u +%FT%TZ) BUILD B (ported branch, +config)"
EXTRA_BAZEL_CONFIGS="--config=skip_dead_axis_jobs" bash $V/build_arm.sh $V/bin_B_up
git checkout -q upstream-main-benchmarks || exit 1
echo "=== $(date -u +%FT%TZ) BUILD A (upstream-main-benchmarks)"
EXTRA_BAZEL_CONFIGS="" bash $V/build_arm.sh $V/bin_A_up
echo "=== $(date -u +%FT%TZ) BITID"
EXTRA_BAZEL_CONFIGS="" bash $S/ydf_bitid_cc18.sh $V/bin_A_up $V/bin_B_up $V/bitid_up > $V/up_bitid.md 2> $V/bitid_up.err; echo "bitid exit $?"; cat $V/up_bitid.md
echo "=== $(date -u +%FT%TZ) ACCURACY A"
EXTRA_BAZEL_CONFIGS="" bash benchmarks/evaluation/accuracy.sh m7i_upb_gbt_hd_a_exact_t30 > $V/up_acc_a.log 2>&1; echo "acc A exit $?"
echo "=== $(date -u +%FT%TZ) RUNTIME A"
EXTRA_BAZEL_CONFIGS="" bash benchmarks/evaluation/runtime.sh --runs=1 m7i_upb_gbt_hd_a_exact_t30_r1 > $V/up_rt_a.log 2>&1; echo "rt A exit $?"
git checkout -q upstream-main-benchmarks-gbt-high-dim-fix || exit 1
echo "=== $(date -u +%FT%TZ) ACCURACY B"
EXTRA_BAZEL_CONFIGS="--config=skip_dead_axis_jobs" bash benchmarks/evaluation/accuracy.sh m7i_upb_gbt_hd_b_exact_t30 > $V/up_acc_b.log 2>&1; echo "acc B exit $?"
echo "=== $(date -u +%FT%TZ) RUNTIME B"
EXTRA_BAZEL_CONFIGS="--config=skip_dead_axis_jobs" bash benchmarks/evaluation/runtime.sh --runs=1 m7i_upb_gbt_hd_b_exact_t30_r1 > $V/up_rt_b.log 2>&1; echo "rt B exit $?"
echo "=== $(date -u +%FT%TZ) DONE"
