#!/usr/bin/env bash
# 2026-09-24: the one arm missing from the YouTube-8M depth-table row — spo_rf_dyn256_vec
# (Dynamic Random Histogram, 256 bins, AVX-512 binner, --dynamic_split_threshold 1000, HWY exact
# below it) x depths {6,10,16,24,purity}, 240 trees, min_examples 1, 48 threads, seed 1, 1 run
# -> speedup_map.csv. Holds run_all.lock; commits + pushes at the end.
set -uo pipefail
WORK_DIR=/home/ubuntu/spo_vs_gbt; REPO_DIR=/home/ubuntu/yggdrasil-oblique-forests
SCRIPT_DIR="$REPO_DIR/benchmarks/src/spo_vs_gbt"; PY=/home/ubuntu/gbt_venv/bin/python
LOG_DIR="$WORK_DIR/logs"; RES_DIR="$REPO_DIR/benchmarks/results/runtime/speedup_map_by_dataset"
OUT="$WORK_DIR/results/speedup_map.csv"; L="$LOG_DIR/run_b7_youtube8m_dyn256.log"
exec 9>"$WORK_DIR/run_all.lock"
echo "[yt8m-dyn256] $(date -u +%FT%TZ) waiting for run_all.lock" | tee -a "$L"
flock 9
echo "[yt8m-dyn256] $(date -u +%FT%TZ) start" | tee -a "$L"
"$PY" "$SCRIPT_DIR/run_speedup_map.py" --cells-file "$SCRIPT_DIR/cells_b7_youtube8m_dyn256.json" \
  --threads 48 --timeout 14400 --out "$OUT" > "$LOG_DIR/b7_youtube8m_dyn256.log" 2>&1
rc=$?; cp -f "$OUT" "$RES_DIR/"
echo "[yt8m-dyn256] $(date -u +%FT%TZ) DONE exit $rc" | tee -a "$L"
cd "$REPO_DIR" && git add "$RES_DIR/speedup_map.csv" "$SCRIPT_DIR/cells_b7_youtube8m_dyn256.json" \
  "$SCRIPT_DIR/run_b7_youtube8m_dyn256.sh" \
  && git commit -q -m "[results] YouTube-8M: spo_rf_dyn256_vec (AVX-512 256-bin dynamic, threshold 1000), depths 6/10/16/24/purity

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01R4q8nNbev2HUa6WD56ciQm" \
  && (git push -q origin rebased-main || echo "[yt8m-dyn256] push rejected; resolve by hand" | tee -a "$L")
touch "$WORK_DIR/b7_youtube8m_dyn256_DONE"
echo "[yt8m-dyn256] $(date -u +%FT%TZ) ALL DONE" | tee -a "$L"
