#!/usr/bin/env bash
# Fork quick pass: gbt-high-dim-speedup (flag skip_dead_axis_jobs) vs rebased-main tip, m7i.
set -uo pipefail
cd /home/ubuntu/yggdrasil-oblique-forests
V=/home/ubuntu/verify_gbt_hd
S=.claude/skills/verify-speedup/scripts
OUT=benchmarks/results/verify/gbt-high-dim-speedup/vs-rebased-main
mkdir -p $OUT
export EXTRA_TRAIN_ARGS='--ensemble_method Boosting --numerical_split_type "Dynamic Random Histogram"'
export NUM_TREES_DIVISOR=10
echo "=== $(date -u +%FT%TZ) BITID"
EXTRA_BAZEL_CONFIGS="" bash $S/ydf_bitid_cc18.sh $V/bin_A_fork $V/bin_B_fork $V/bitid_fork > $OUT/m7i_bitid.md 2> $V/bitid_fork.err; echo "bitid exit $?"
cat $OUT/m7i_bitid.md
echo "=== $(date -u +%FT%TZ) ACCURACY A"
EXTRA_BAZEL_CONFIGS="" bash benchmarks/evaluation/accuracy.sh m7i_gbt_hd_a_dyn_t30 > $V/acc_a.log 2>&1; echo "acc A exit $?"
echo "=== $(date -u +%FT%TZ) RUNTIME A"
EXTRA_BAZEL_CONFIGS="" bash benchmarks/evaluation/runtime.sh --runs=1 m7i_gbt_hd_a_dyn_t30_r1 > $V/rt_a.log 2>&1; echo "rt A exit $?"
echo "=== $(date -u +%FT%TZ) ACCURACY B"
EXTRA_BAZEL_CONFIGS="--config=skip_dead_axis_jobs" bash benchmarks/evaluation/accuracy.sh m7i_gbt_hd_b_dyn_t30 > $V/acc_b.log 2>&1; echo "acc B exit $?"
echo "=== $(date -u +%FT%TZ) RUNTIME B"
EXTRA_BAZEL_CONFIGS="--config=skip_dead_axis_jobs" bash benchmarks/evaluation/runtime.sh --runs=1 m7i_gbt_hd_b_dyn_t30_r1 > $V/rt_b.log 2>&1; echo "rt B exit $?"
echo "=== $(date -u +%FT%TZ) DONE"
