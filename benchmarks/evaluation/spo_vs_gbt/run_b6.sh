#!/usr/bin/env bash
# B6 (2026-09-07): trunk row-column map, cells_b6.json (12 shapes x 3 arms,
# unlimited depth). Appends to speedup_map.csv (skip-existing). Waits for the
# study lock (run_256.sh / run_all.sh) so it never overlaps another training.
set -uo pipefail
WORK_DIR=/home/ubuntu/spo_vs_gbt
REPO_DIR=/home/ubuntu/yggdrasil-oblique-forests
SCRIPT_DIR="$REPO_DIR/benchmarks/evaluation/spo_vs_gbt"
PY=/home/ubuntu/gbt_venv/bin/python
LOG_DIR="$WORK_DIR/logs"
exec 9>"$WORK_DIR/run_all.lock"
echo "[run_b6] $(date -u +%FT%TZ) waiting for run_all.lock" | tee -a "$LOG_DIR/run_b6.log"
flock 9   # blocks until run_256.sh releases it
echo "[run_b6] $(date -u +%FT%TZ) start" | tee -a "$LOG_DIR/run_b6.log"
"$PY" "$SCRIPT_DIR/run_speedup_map.py" --cells-file "$SCRIPT_DIR/cells_b6.json" \
  --threads 48 --timeout 14400 --out "$WORK_DIR/results/speedup_map.csv" > "$LOG_DIR/b6.log" 2>&1
echo "[run_b6] $(date -u +%FT%TZ) b6 exit $?" | tee -a "$LOG_DIR/run_b6.log"
cp -f "$WORK_DIR/results/speedup_map.csv" "$REPO_DIR/benchmarks/results/spo_vs_gbt/"
touch "$WORK_DIR/b6_DONE"
echo "[run_b6] $(date -u +%FT%TZ) ALL DONE" | tee -a "$LOG_DIR/run_b6.log"
