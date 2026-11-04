#!/usr/bin/env bash
# run_barrier_cell.sh AXIS [TREES] [WORKERS] [THREADS]
# One in-process distributed GBT cell with the manager's barrier recorder on (DGBT_BARRIER_CSV).
# Default shape = the 2026-10-22 smoke: 100k x 256 trunk (gen_trunk.py, seed 1), 50 trees, depth 6,
# 4 workers x 2 threads (8-vCPU box). TRUNK_DIR must hold <rows>x256/train/shard_00.csv and
# test_100000x256/shard_00.csv; make it with:
#   python gen_trunk.py --out $TRUNK_DIR --rows 100000 --cols 256 --shards 1 --test_shards 1
# Values: params/kdd28_params.json profile barrier_smoke.
set -euo pipefail
PARAMS=$(cd "$(dirname "$0")" && pwd)/params/params.py
. "$(dirname "$PARAMS")/drift.sh"  # run_cell: pre-flight + drift stop (ALLOW_DRIFT=1 warns)
eval "$(python3 "$PARAMS" export barrier_smoke Q_)"
AXIS=$1; TREES=${2:-$Q_TREES}; W=${3:-$Q_WORKERS}; THREADS=${4:-$Q_THREADS}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=$REPO/bazel-bin/examples/train_distributed_gbt
TRUNK_DIR=${TRUNK_DIR:?set TRUNK_DIR to the gen_trunk.py --out directory}
ROWS=${ROWS:-100000}
OUT=${OUT:-$REPO/benchmarks/results/runtime/distributed_spo_gbt/barriers}
WORK=${WORK:-/home/ubuntu/dgbt_work/trunk${ROWS}x256}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
TAG="trunk${ROWS}x256_W${W}_${AXIS}_T${TREES}_d6_${STAMP}"
mkdir -p "$OUT" "$WORK"
export DGBT_BARRIER_CSV=$OUT/$TAG.csv
PROF=barrier_smoke.spo; [ "$AXIS" = AXIS_ALIGNED ] && PROF=barrier_smoke.aa
# SPO flags on SPO cells only: AXIS_ALIGNED fails on them (train_distributed_gbt.cc, never ignore a flag).
S1=(); S2=()  # S2 = legacy per-worker P: the first barrier measurement predates the split budget
[ "$AXIS" = AXIS_ALIGNED ] || { S1=(--sparse_oblique_projection_density_factor $Q_SPO_DENSITY --sparse_oblique_num_projections_exponent $Q_SPO_EXPONENT)
  S2=(--sparse_oblique_max_num_projections $Q_SPO_MAX_PROJECTIONS --sparse_oblique_split_budget=$Q_SPLIT_BUDGET); }
DRIFT_EXPECT="trees=$TREES" run_cell $PROF "$OUT/$TAG.log" "$BIN" --alsologtostderr --train_csv "$TRUNK_DIR/${ROWS}x256/train/shard_00.csv" \
  --test_csv "$TRUNK_DIR/test_100000x256/shard_00.csv" --label label \
  --work_dir "$WORK" --num_workers "$W" --num_threads "$THREADS" --num_trees "$TREES" --max_depth $Q_MAX_DEPTH \
  --split_axis "$AXIS" ${S1[@]+"${S1[@]}"} --worker_logs=false ${S2[@]+"${S2[@]}"}
grep -E "Training \(incl|^Accuracy:|effective-params:" "$OUT/$TAG.log" | cut -c1-160
PY=$REPO/.venv/bin/python; [ -x "$PY" ] || PY=python3  # pandas lives in the repo venv
"$PY" "$REPO/benchmarks/src/distributed_spo_gbt/summarize_barriers.py" "$DGBT_BARRIER_CSV"
