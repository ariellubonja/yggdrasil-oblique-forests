#!/usr/bin/env bash
# run_barrier_cell.sh AXIS [TREES] [WORKERS] [THREADS]
# One in-process distributed GBT cell with the manager's barrier recorder on (DGBT_BARRIER_CSV).
# Default shape = the 2026-10-22 smoke: 100k x 256 trunk (gen_trunk.py, seed 1), 50 trees, depth 6,
# 4 workers x 2 threads (8-vCPU box). TRUNK_DIR must hold <rows>x256/train/shard_00.csv and
# test_100000x256/shard_00.csv; make it with:
#   python gen_trunk.py --out $TRUNK_DIR --rows 100000 --cols 256 --shards 1 --test_shards 1
set -euo pipefail
AXIS=$1; TREES=${2:-50}; W=${3:-4}; THREADS=${4:-2}
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
"$BIN" --alsologtostderr --train_csv "$TRUNK_DIR/${ROWS}x256/train/shard_00.csv" \
  --test_csv "$TRUNK_DIR/test_100000x256/shard_00.csv" --label label \
  --work_dir "$WORK" --num_workers "$W" --num_threads "$THREADS" --num_trees "$TREES" --max_depth 6 \
  --split_axis "$AXIS" --sparse_oblique_projection_density_factor 1.5 \
  --sparse_oblique_num_projections_exponent 0.5 --worker_logs=false > "$OUT/$TAG.log" 2>&1
grep -E "Training \(incl|^Accuracy:" "$OUT/$TAG.log" | cut -c1-160
PY=$REPO/.venv/bin/python; [ -x "$PY" ] || PY=python3  # pandas lives in the repo venv
"$PY" "$REPO/benchmarks/src/distributed_spo_gbt/summarize_barriers.py" "$DGBT_BARRIER_CSV"
