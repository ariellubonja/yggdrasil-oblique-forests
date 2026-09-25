#!/usr/bin/env bash
# SPO-GBT depth-6 sweep for the overall depth table (user directive 2026-09-24): 6 GBT arms
# (exact_hwy, rand_vec, rand256_vec, rand_scalar, rand256_scalar; 300 trees, depth 6,
# min_examples 5 = the study's GBT defaults; NO dynamic arms, user directive 2026-09-24 evening)
# x the 19 B7 datasets + YouTube-8M, 1 run.
# Meant for a SECOND box (runs in parallel with the RF work on the m7i). Output
# $WORK_DIR/results/speedup_map_gbt_d6.csv, merged into speedup_map.csv by hand afterwards.
#
# Prereqs on the box (benchmarks/SETUP_FRESH_BOX.md): bins built WITH --config=skip_dead_axis_jobs
# into $WORK_DIR/bin/{default,scalar} (GBT on wide data is very slow without it), the venv,
# HIGGS/SUSY/Epsilon/GiveMeSomeCredit CSVs under benchmarks/data, and YouTube-8M
# (`python benchmarks/data/download_youtube8m.py --partitions train`, ~5 min, 56 GB + 18 GB shards).
# Trunk cells are generated in-process (no data needed); 1.5M x 40k needs ~250 GB RAM.
#   tmux new-session -d -s gbt_d6 'bash benchmarks/src/spo_vs_gbt/run_b7_gbt_d6.sh'
# Resumable (skip-existing per cell). Stages: fast arms (dataset-outer) then scalar arms.
set -uo pipefail
WORK_DIR=${WORK_DIR:-/home/ubuntu/spo_vs_gbt}
REPO_DIR=${REPO_DIR:-/home/ubuntu/yggdrasil-oblique-forests}
PY=${PY:-/home/ubuntu/gbt_venv/bin/python}
SCRIPT_DIR="$REPO_DIR/benchmarks/src/spo_vs_gbt"
LOG_DIR="$WORK_DIR/logs"; mkdir -p "$LOG_DIR" "$WORK_DIR/results"
OUT=${OUT:-$WORK_DIR/results/speedup_map_gbt_d6.csv}   # m7i: OUT=.../speedup_map.csv COMMIT=1
L="$LOG_DIR/run_b7_gbt_d6.log"
exec 9>"$WORK_DIR/run_all.lock"
echo "[gbt_d6] $(date -u +%FT%TZ) waiting for run_all.lock" | tee -a "$L"
flock 9
for stage in fast slow; do
  echo "[gbt_d6] $(date -u +%FT%TZ) stage $stage start" | tee -a "$L"
  "$PY" "$SCRIPT_DIR/run_speedup_map.py" --cells-file "$SCRIPT_DIR/cells_b7_gbt_d6_${stage}.json" \
    --bin-dir "$WORK_DIR/bin" --threads 48 --timeout 14400 --out "$OUT" > "$LOG_DIR/b7_gbt_d6_${stage}.log" 2>&1
  echo "[gbt_d6] $(date -u +%FT%TZ) stage $stage DONE exit $?" | tee -a "$L"
  touch "$WORK_DIR/b7_gbt_d6_${stage}_DONE"
  if [ "${COMMIT:-0}" = 1 ]; then
    RES_DIR="$REPO_DIR/benchmarks/results/runtime/speedup_map_by_dataset"; cp -f "$OUT" "$RES_DIR/"
    cd "$REPO_DIR" && git add "$RES_DIR/$(basename "$OUT")" "$SCRIPT_DIR/cells_b7_gbt_d6_fast.json" \
      "$SCRIPT_DIR/cells_b7_gbt_d6_slow.json" "$SCRIPT_DIR/run_b7_gbt_d6.sh" \
      && git commit -q -m "[results] SPO-GBT depth 6 on the B7 grid + YouTube-8M, $stage arms

Claude-Session: https://claude.ai/code/session_01R4q8nNbev2HUa6WD56ciQm" \
      && (git push -q origin rebased-main || (git stash -q; git fetch -q origin && git rebase -q origin/rebased-main && git push -q origin rebased-main; git stash pop -q))
    echo "[gbt_d6] $(date -u +%FT%TZ) git push exit $?" | tee -a "$L"
  fi
done
touch "$WORK_DIR/b7_gbt_d6_DONE"
echo "[gbt_d6] $(date -u +%FT%TZ) ALL DONE -> $OUT" | tee -a "$L"
