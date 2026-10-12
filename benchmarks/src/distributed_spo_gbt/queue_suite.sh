#!/usr/bin/env bash
# Runs after the 50-tree grid and its single-machine references (queue_single50.sh):
# rebuild the driver (new --sparse_oblique_max_num_projections flag), W=16 smoke on a
# 4-feature dataset, then the Table-9 suite. Holds the training lock throughout.
set -uo pipefail
REPO=/home/ubuntu/yggdrasil-oblique-forests; B=$REPO/benchmarks/src/distributed_spo_gbt
O=$REPO/benchmarks/results/runtime/distributed_spo_gbt/logs
until grep -q "SINGLE50_DONE" "$O/single50.out" 2>/dev/null; do sleep 60; done
exec 9>/home/ubuntu/spo_vs_gbt/run_all.lock; flock 9 || exit 1
cd "$REPO"
echo "### $(date -u +%FT%TZ) BUILD"
set +u; source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1; set -u
bazel build -c opt --cxxopt=-O3 --cxxopt=-march=native --repo_env=CC=icx --repo_env=CXX=icpx //examples:train_distributed_gbt > "$O/suite_build.out" 2>&1 || { echo "### BUILD FAILED"; exit 1; }
echo "### $(date -u +%FT%TZ) SMOKE W=16 on blood-transfusion-service-center (F=4)"
DATASETS="blood-transfusion-service-center" CELLS="16:SPARSE_OBLIQUE 16:AXIS_ALIGNED" RESULTS=/home/ubuntu/dgbt_work/suite_smoke/smoke16.csv "$B/run_dgbt_suite.sh"
echo "### $(date -u +%FT%TZ) SUITE"
"$B/run_dgbt_suite.sh"
echo "### QUEUE_SUITE_DONE $(date -u +%FT%TZ)"
