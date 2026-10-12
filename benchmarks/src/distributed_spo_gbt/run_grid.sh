#!/usr/bin/env bash
# Dataset-outer queue of run_dgbt.sh cells (holds the shared training lock).
#   DATASETS="SUSY EPSILON HIGGS YOUTUBE8M" CELLS="1:SPARSE_OBLIQUE 4:SPARSE_OBLIQUE 8:SPARSE_OBLIQUE 4:AXIS_ALIGNED" TREES=300 run_grid.sh
# Each CELL is WORKERS:SPLIT_AXIS. Set EXTRA for extra driver flags.
set -uo pipefail
REPO=/home/ubuntu/yggdrasil-oblique-forests
DATASETS=${DATASETS:-"SUSY EPSILON HIGGS YOUTUBE8M"}
CELLS=${CELLS:-"1:SPARSE_OBLIQUE 4:SPARSE_OBLIQUE 8:SPARSE_OBLIQUE 4:AXIS_ALIGNED"}
TREES=${TREES:-300}
EXTRA=${EXTRA:-}
exec 9>/home/ubuntu/spo_vs_gbt/run_all.lock
flock 9 || exit 1
for ds in $DATASETS; do
  for cell in $CELLS; do
    w=${cell%%:*}; axis=${cell##*:}
    echo "### $(date -u +%FT%TZ) START $ds W=$w $axis T=$TREES"
    "$REPO/benchmarks/src/distributed_spo_gbt/run_dgbt.sh" "$ds" "$w" "$axis" "$TREES" $EXTRA
    echo "### $(date -u +%FT%TZ) END   $ds W=$w $axis rc=$?"
  done
done
echo "### GRID_DONE $(date -u +%FT%TZ)"
