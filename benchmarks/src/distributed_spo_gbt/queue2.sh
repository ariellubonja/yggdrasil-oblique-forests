#!/usr/bin/env bash
# Runs after pass 1 releases the training lock: single-machine anchors, then pass 2 (4-worker axis-aligned; 8-worker oblique cells dropped 2026-10-12 per user).
set -uo pipefail
B=/home/ubuntu/yggdrasil-oblique-forests/benchmarks/src/distributed_spo_gbt
LOCK=/home/ubuntu/spo_vs_gbt/run_all.lock
for ds in SUSY EPSILON HIGGS YOUTUBE8M; do
  echo "### $(date -u +%FT%TZ) ANCHOR $ds"
  flock "$LOCK" "$B/run_single_anchor.sh" "$ds" 300
done
echo "### $(date -u +%FT%TZ) ANCHORS_DONE"
DATASETS="SUSY EPSILON HIGGS" CELLS="4:AXIS_ALIGNED" TREES=300 "$B/run_grid.sh"
echo "### QUEUE2_DONE $(date -u +%FT%TZ)"
