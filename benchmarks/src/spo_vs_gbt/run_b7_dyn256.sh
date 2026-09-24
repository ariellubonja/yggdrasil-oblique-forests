#!/usr/bin/env bash
# 2026-09-24: spo_rf_dyn256_vec (Dynamic Random Histogram, 256 bins, AVX-512 binner,
# --dynamic_split_threshold 1000, Highway exact below it) on the B7 map grid
# (19 datasets x depths {6,10,16,24,full}), 3 reps. Rep 1 -> speedup_map.csv (same
# file as every other arm's rep 1), reps 2,3 -> speedup_map_rep{2,3}.csv. Binary:
# bin/default @ 6885c576 (HEAD differs only in chrono scopes, compiled out).
# Commits + pushes the result files at the end. Holds run_all.lock throughout.
set -uo pipefail
WORK_DIR=/home/ubuntu/spo_vs_gbt; REPO_DIR=/home/ubuntu/yggdrasil-oblique-forests
SCRIPT_DIR="$REPO_DIR/benchmarks/src/spo_vs_gbt"; PY=/home/ubuntu/gbt_venv/bin/python
LOG_DIR="$WORK_DIR/logs"; RES_DIR="$REPO_DIR/benchmarks/results/runtime/speedup_map_by_dataset"
L="$LOG_DIR/run_b7_dyn256.log"
exec 9>"$WORK_DIR/run_all.lock"
echo "[dyn256] $(date -u +%FT%TZ) waiting for run_all.lock" | tee -a "$L"
flock 9
echo "[dyn256] $(date -u +%FT%TZ) start" | tee -a "$L"
for rep in 1 2 3; do
  if [ "$rep" = 1 ]; then OUT="$WORK_DIR/results/speedup_map.csv"; else OUT="$WORK_DIR/results/speedup_map_rep${rep}.csv"; fi
  for tag in d6 d10 d16 d24 full; do
    echo "[dyn256] $(date -u +%FT%TZ) rep $rep depth $tag start" | tee -a "$L"
    "$PY" "$SCRIPT_DIR/run_speedup_map.py" --cells-file "$SCRIPT_DIR/cells_b7_dyn256_${tag}.json" \
      --threads 48 --timeout 14400 --out "$OUT" > "$LOG_DIR/b7_dyn256_rep${rep}_${tag}.log" 2>&1
    rc=$?
    cp -f "$OUT" "$RES_DIR/"
    echo "[dyn256] $(date -u +%FT%TZ) rep $rep depth $tag DONE exit $rc" | tee -a "$L"
  done
  touch "$WORK_DIR/b7_dyn256_rep${rep}_DONE"
done
cd "$REPO_DIR" && git add "$RES_DIR/speedup_map.csv" "$RES_DIR/speedup_map_rep2.csv" "$RES_DIR/speedup_map_rep3.csv" \
  && git commit -q -m "[results] spo_rf_dyn256_vec (AVX-512 256-bin dynamic, threshold 1000), B7 map grid, 3 reps

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01R4q8nNbev2HUa6WD56ciQm" \
  && (git push -q origin rebased-main || (git fetch -q origin && git rebase -q origin/rebased-main && git push -q origin rebased-main))
echo "[dyn256] $(date -u +%FT%TZ) git push exit $?" | tee -a "$L"
touch "$WORK_DIR/b7_dyn256_DONE"
echo "[dyn256] $(date -u +%FT%TZ) ALL DONE" | tee -a "$L"
