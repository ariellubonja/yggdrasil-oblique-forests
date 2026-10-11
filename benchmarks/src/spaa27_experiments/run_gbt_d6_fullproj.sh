#!/usr/bin/env bash
# SPO-GBT depth-6 runtime grid rerun with P = F projections per node (user directive 2026-10-11):
# the table's 5 GBT arms (exact_hwy, rand_vec, rand256_vec, rand_scalar, rand256_scalar; 300 trees,
# depth 6, min_examples 5, 48 threads, 1 run) as their *_fullproj variants (arms.py ARMS_GBT_FULLPROJ:
# --num_projections_exponent 1 --max_num_projections INT_MAX, density still 1.5/F) x the 23
# depth-table datasets = 115 cells (cells_gbt_d6_fullproj.json, dataset-outer cheapest-first).
# The sqrt(F) rows used the harness default exponent 0.5 (P = ceil(sqrt F), cap 1000) while
# aa_gbt_exact / XGBoost / LightGBM / CatBoost scan all F. Per-cell cost scales ~F/P, so the first
# pass (BOX=1|2|3 -> cells_gbt_d6_fullproj_box$BOX.json) covers the datasets estimated <= 60 h each:
#   box 1 (main m7i, has the CSVs): the 8 CSV datasets (40 cells, <= ~140 h upper bound)
#   box 2 (trunk only): 15k x 40k, 1M x 512, 150k x 40k, 1M x 2048, 15k x 400k (25 cells, <= ~105 h)
#   box 3 (trunk only): 15k x 4096, 1M x 32, 150k x 4096, 1.5M x 4096, 150k x 160k (25 cells, <= ~104 h)
# Deferred (est. 97-581 h each): 1.5M x 16384, 150k x 400k, 4.5M x 4096, 1.5M x 40k, 15k x 1.6M.
# Bins: /home/ubuntu/spo_vs_gbt/bin/{default,scalar} built by build_bins.sh at 6885c576 (icx).
#   tmux new-session -d -s gbtfp 'BOX=2 COMMIT=1 bash benchmarks/src/spaa27_experiments/run_gbt_d6_fullproj.sh'
# Box 1 writes straight into speedup_map.csv (OUT=.../results/speedup_map.csv); boxes 2/3 write
# results/speedup_map_gbt_fullproj_box$BOX.csv, committed as-is and merged into speedup_map.csv by hand.
# Resumable (skip-existing per cell). Holds run_all.lock. Per-cell timeout 24 h.
set -uo pipefail
WORK_DIR=${WORK_DIR:-/home/ubuntu/spo_vs_gbt}
REPO_DIR=${REPO_DIR:-/home/ubuntu/yggdrasil-oblique-forests}
PY=${PY:-/home/ubuntu/gbt_venv/bin/python}
SCRIPT_DIR="$REPO_DIR/benchmarks/src/spaa27_experiments"
LOG_DIR="$WORK_DIR/logs"; mkdir -p "$LOG_DIR" "$WORK_DIR/results"
BOX=${BOX:-1}
OUT=${OUT:-$WORK_DIR/results/speedup_map_gbt_fullproj_box$BOX.csv}
L="$LOG_DIR/run_gbt_d6_fullproj.log"
mkdir -p "$(dirname "$OUT")"
exec 9>"$WORK_DIR/run_all.lock"
echo "[gbtfp box$BOX] $(date -u +%FT%TZ) waiting for run_all.lock" | tee -a "$L"
flock 9
echo "[gbtfp box$BOX] $(date -u +%FT%TZ) start" | tee -a "$L"
"$PY" "$SCRIPT_DIR/run_speedup_map.py" --cells-file "$SCRIPT_DIR/cells_gbt_d6_fullproj_box$BOX.json" \
  --bin-dir "$WORK_DIR/bin" --threads 48 --timeout 86400 --out "$OUT" > "$LOG_DIR/gbt_d6_fullproj_box$BOX.log" 2>&1
echo "[gbtfp box$BOX] $(date -u +%FT%TZ) DONE exit $?" | tee -a "$L"
touch "$WORK_DIR/gbt_d6_fullproj_box${BOX}_DONE"
if [ "${COMMIT:-0}" = 1 ]; then
  RES_DIR="$REPO_DIR/benchmarks/results/runtime/speedup_map_by_dataset"; cp -f "$OUT" "$RES_DIR/"
  cd "$REPO_DIR" && git add "$RES_DIR/$(basename "$OUT")" "$SCRIPT_DIR"/cells_gbt_d6_fullproj*.json "$SCRIPT_DIR/run_gbt_d6_fullproj.sh" \
    && git commit -q -m "[results] SPO-GBT depth 6 with P = F projections (*_fullproj), box $BOX

Claude-Session: https://claude.ai/code/session_01R4q8nNbev2HUa6WD56ciQm" \
    && (git push -q origin rebased-main || (git stash -q; git fetch -q origin && git rebase -q origin/rebased-main && git push -q origin rebased-main; git stash pop -q))
  echo "[gbtfp box$BOX] $(date -u +%FT%TZ) git push exit $?" | tee -a "$L"
fi
echo "[gbtfp box$BOX] $(date -u +%FT%TZ) ALL DONE -> $OUT" | tee -a "$L"
