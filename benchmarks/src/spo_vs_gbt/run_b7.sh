#!/usr/bin/env bash
# B7 (2026-09-08): overall depth table. 19 selected entries x 6 SPO-RF arms x
# depths {6,10,16,24,full}, min_examples 1, 240 trees, 48 threads, seed 1.
# Depth-major so each depth completes as a unit; touches b7_<tag>_DONE per depth.
# Rep 1 appends to speedup_map.csv (skip-existing: purity cells already there
# are reused). Usage: run_b7.sh [REP]   (REP>1 -> separate out file, no skip).
set -uo pipefail
REP="${1:-1}"
WORK_DIR=/home/ubuntu/spo_vs_gbt
REPO_DIR=/home/ubuntu/yggdrasil-oblique-forests
SCRIPT_DIR="$REPO_DIR/benchmarks/src/spo_vs_gbt"
PY=/home/ubuntu/gbt_venv/bin/python
LOG_DIR="$WORK_DIR/logs"
if [ "$REP" = "1" ]; then OUT="$WORK_DIR/results/speedup_map.csv"; SKIP=""; SUF="";
else OUT="$WORK_DIR/results/speedup_map_rep${REP}.csv"; SKIP="--no-skip-existing"; SUF="_rep${REP}"; fi
exec 9>"$WORK_DIR/run_all.lock"
echo "[run_b7$SUF] $(date -u +%FT%TZ) waiting for run_all.lock" | tee -a "$LOG_DIR/run_b7.log"
flock 9
echo "[run_b7$SUF] $(date -u +%FT%TZ) start (out=$OUT)" | tee -a "$LOG_DIR/run_b7.log"
for tag in d6 d10 d16 d24 full; do
  echo "[run_b7$SUF] $(date -u +%FT%TZ) depth $tag start" | tee -a "$LOG_DIR/run_b7.log"
  "$PY" "$SCRIPT_DIR/run_speedup_map.py" --cells-file "$SCRIPT_DIR/cells_b7_${tag}.json" \
    --threads 48 --timeout 14400 --out "$OUT" $SKIP > "$LOG_DIR/b7_${tag}${SUF}.log" 2>&1
  rc=$?
  cp -f "$OUT" "$REPO_DIR/benchmarks/results/runtime/speedup_map_by_dataset/"
  echo "[run_b7$SUF] $(date -u +%FT%TZ) depth $tag DONE exit $rc" | tee -a "$LOG_DIR/run_b7.log"
  touch "$WORK_DIR/b7_${tag}${SUF}_DONE"
done
touch "$WORK_DIR/b7${SUF}_DONE"
echo "[run_b7$SUF] $(date -u +%FT%TZ) ALL DONE" | tee -a "$LOG_DIR/run_b7.log"
