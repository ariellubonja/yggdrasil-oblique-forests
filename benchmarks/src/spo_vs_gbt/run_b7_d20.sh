#!/usr/bin/env bash
# SPO-RF depth-20 row for the overall depth table (user request 2026-09-25): the table's 7 RF
# arms (exact_hwy, rand_scalar, rand256_scalar, rand_vec, rand256_vec, dyn_vec, dyn256_vec;
# 240 trees, min_examples 1, 48 threads, 1 run) x the 23 depth-table datasets = 161 cells,
# dataset-outer (cheapest dataset first), arms inner. Split over three m7i boxes by estimated
# runtime (BOX=1|2|3 -> cells_b7_d20_box$BOX.json; ~4 h each):
#   box 1 (the main m7i, has every CSV): all 8 CSV datasets + the small trunks (119 cells)
#   box 2 (trunk only, 1.5M x 40k needs ~250 GB RAM): 1.5M x 40k, 150k x 160k, 150k x 400k (21)
#   box 3 (trunk only): 4.5M x 4096, 1.5M x 16384, 1.5M x 4096 (21)
# Bins: /home/ubuntu/spo_vs_gbt/bin/{default,scalar} built by build_bins.sh at 6885c576 (icx).
#   tmux new-session -d -s d20 'BOX=2 COMMIT=1 bash benchmarks/src/spo_vs_gbt/run_b7_d20.sh'
# Box 1 writes straight into speedup_map.csv (OUT=.../results/speedup_map.csv); boxes 2/3 write
# results/speedup_map_d20_box$BOX.csv, committed as-is and merged into speedup_map.csv by hand.
# Resumable (skip-existing per cell). Holds run_all.lock.
set -uo pipefail
WORK_DIR=${WORK_DIR:-/home/ubuntu/spo_vs_gbt}
REPO_DIR=${REPO_DIR:-/home/ubuntu/yggdrasil-oblique-forests}
PY=${PY:-/home/ubuntu/gbt_venv/bin/python}
SCRIPT_DIR="$REPO_DIR/benchmarks/src/spo_vs_gbt"
LOG_DIR="$WORK_DIR/logs"; mkdir -p "$LOG_DIR" "$WORK_DIR/results"
BOX=${BOX:-1}
OUT=${OUT:-$WORK_DIR/results/speedup_map_d20_box$BOX.csv}
L="$LOG_DIR/run_b7_d20.log"
mkdir -p "$(dirname "$OUT")"
exec 9>"$WORK_DIR/run_all.lock"
echo "[d20 box$BOX] $(date -u +%FT%TZ) waiting for run_all.lock" | tee -a "$L"
flock 9
echo "[d20 box$BOX] $(date -u +%FT%TZ) start" | tee -a "$L"
"$PY" "$SCRIPT_DIR/run_speedup_map.py" --cells-file "$SCRIPT_DIR/cells_b7_d20_box$BOX.json" \
  --bin-dir "$WORK_DIR/bin" --threads 48 --timeout 14400 --out "$OUT" > "$LOG_DIR/b7_d20_box$BOX.log" 2>&1
echo "[d20 box$BOX] $(date -u +%FT%TZ) DONE exit $?" | tee -a "$L"
touch "$WORK_DIR/b7_d20_box${BOX}_DONE"
if [ "${COMMIT:-0}" = 1 ]; then
  RES_DIR="$REPO_DIR/benchmarks/results/runtime/speedup_map_by_dataset"; cp -f "$OUT" "$RES_DIR/"
  cd "$REPO_DIR" && git add "$RES_DIR/$(basename "$OUT")" "$SCRIPT_DIR"/cells_b7_d20*.json "$SCRIPT_DIR/run_b7_d20.sh" \
    && git commit -q -m "[results] SPO-RF depth 20 on the depth-table grid, box $BOX

Claude-Session: https://claude.ai/code/session_01R4q8nNbev2HUa6WD56ciQm" \
    && (git push -q origin rebased-main || (git stash -q; git fetch -q origin && git rebase -q origin/rebased-main && git push -q origin rebased-main; git stash pop -q))
  echo "[d20 box$BOX] $(date -u +%FT%TZ) git push exit $?" | tee -a "$L"
fi
echo "[d20 box$BOX] $(date -u +%FT%TZ) ALL DONE -> $OUT" | tee -a "$L"
