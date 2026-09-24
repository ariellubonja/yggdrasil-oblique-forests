#!/usr/bin/env bash
# 2026-09-24 (user directive): add YouTube-8M video-level (3,888,919 x 1152, class = entity 0
# "Game" vs rest) to every row of the overall depth table: the 6 SPO-RF table arms x depths
# {6,10,16,24,purity}, 240 trees, min_examples 1, 1 run -> speedup_map.csv (dataset YOUTUBE8M).
# Fast arms (exact_hwy, rand_vec, rand256_vec, dyn_vec) over all depths first, scalar arms last.
# Holds run_all.lock throughout; copies + commits + pushes after each stage.
set -uo pipefail
WORK_DIR=/home/ubuntu/spo_vs_gbt; REPO_DIR=/home/ubuntu/yggdrasil-oblique-forests
SCRIPT_DIR="$REPO_DIR/benchmarks/src/spo_vs_gbt"; PY=/home/ubuntu/gbt_venv/bin/python
LOG_DIR="$WORK_DIR/logs"; RES_DIR="$REPO_DIR/benchmarks/results/runtime/speedup_map_by_dataset"
OUT="$WORK_DIR/results/speedup_map.csv"
L="$LOG_DIR/run_b7_youtube8m.log"
exec 9>"$WORK_DIR/run_all.lock"
echo "[yt8m] $(date -u +%FT%TZ) waiting for run_all.lock" | tee -a "$L"
flock 9
echo "[yt8m] $(date -u +%FT%TZ) start" | tee -a "$L"
commit_push() {
  cd "$REPO_DIR" && git add "$RES_DIR/speedup_map.csv" "$SCRIPT_DIR/cells_b7_youtube8m_fast.json" \
    "$SCRIPT_DIR/cells_b7_youtube8m_slow.json" "$SCRIPT_DIR/run_b7_youtube8m.sh" "$SCRIPT_DIR/arms.py" \
    && git commit -q -m "[results] $1

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01R4q8nNbev2HUa6WD56ciQm" \
    && (git push -q origin rebased-main || (git fetch -q origin && git rebase -q origin/rebased-main && git push -q origin rebased-main))
  echo "[yt8m] $(date -u +%FT%TZ) git push exit $?" | tee -a "$L"
}
for stage in fast slow; do
  echo "[yt8m] $(date -u +%FT%TZ) stage $stage start" | tee -a "$L"
  "$PY" "$SCRIPT_DIR/run_speedup_map.py" --cells-file "$SCRIPT_DIR/cells_b7_youtube8m_${stage}.json" \
    --threads 48 --timeout 14400 --out "$OUT" > "$LOG_DIR/b7_youtube8m_${stage}.log" 2>&1
  rc=$?; cp -f "$OUT" "$RES_DIR/"
  echo "[yt8m] $(date -u +%FT%TZ) stage $stage DONE exit $rc" | tee -a "$L"
  touch "$WORK_DIR/b7_youtube8m_${stage}_DONE"
  commit_push "YouTube-8M on the depth table, $stage arms, depths 6/10/16/24/purity"
done
touch "$WORK_DIR/b7_youtube8m_DONE"
echo "[yt8m] $(date -u +%FT%TZ) ALL DONE" | tee -a "$L"
