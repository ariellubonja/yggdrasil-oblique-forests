#!/usr/bin/env bash
# 2026-09-13: (1) B7 map cells, rep 1, for the two 256-bin Random-histogram arms
# (spo_rf_rand256_scalar = scalar binner, spo_rf_rand256_vec = AVX-512 binner,
# both on the same grid as cells_b7_dynscalar.json: 19 datasets x depths
# {6,10,16,24,full}); appends to speedup_map.csv (skip-existing). (2) Then the
# dynamic-threshold breakeven sweep for AVX-512 256-bin Random vs the Highway
# exact finder, thresholds 100 + k*1000 (k = 0..10), 6 datasets, 1 run each,
# via benchmarks/src/dynamic_histogram_threshold_sweep.sh. Holds run_all.lock
# throughout so nothing else trains concurrently.
set -uo pipefail
WORK_DIR=/home/ubuntu/spo_vs_gbt; REPO_DIR=/home/ubuntu/yggdrasil-oblique-forests
SCRIPT_DIR="$REPO_DIR/benchmarks/src/spo_vs_gbt"; PY=/home/ubuntu/gbt_venv/bin/python
LOG_DIR="$WORK_DIR/logs"; OUT="$WORK_DIR/results/speedup_map.csv"
L="$LOG_DIR/run_b7_rand256.log"
exec 9>"$WORK_DIR/run_all.lock"
echo "[rand256] $(date -u +%FT%TZ) waiting for run_all.lock" | tee -a "$L"
flock 9
echo "[rand256] $(date -u +%FT%TZ) start" | tee -a "$L"
for tag in d6 d10 d16 d24 full; do
  echo "[rand256] $(date -u +%FT%TZ) depth $tag start" | tee -a "$L"
  "$PY" "$SCRIPT_DIR/run_speedup_map.py" --cells-file "$SCRIPT_DIR/cells_b7_rand256_${tag}.json" \
    --threads 48 --timeout 14400 --out "$OUT" > "$LOG_DIR/b7_rand256_${tag}.log" 2>&1
  rc=$?
  cp -f "$OUT" "$REPO_DIR/benchmarks/results/runtime/speedup_map_by_dataset/"
  echo "[rand256] $(date -u +%FT%TZ) depth $tag DONE exit $rc" | tee -a "$L"
  touch "$WORK_DIR/b7_rand256_${tag}_DONE"
done
touch "$WORK_DIR/b7_rand256_DONE"
echo "[rand256] $(date -u +%FT%TZ) map ALL DONE; starting AVX-512 256-bin breakeven sweep" | tee -a "$L"

cd "$REPO_DIR" || exit 1
EXTRA_TRAIN_ARGS='--feature_split_type Oblique --numerical_split_type "Dynamic Random Histogram" --histogram_num_bins 256' \
DYNAMIC_SPLIT_THRESHOLDS_OVERRIDE="100 1100 2100 3100 4100 5100 6100 7100 8100 9100 10100" \
  bash benchmarks/src/dynamic_histogram_threshold_sweep.sh hwysort_avx512_random256_thr100_10100 \
  > "$LOG_DIR/breakeven_avx512_256.log" 2>&1
echo "[rand256] $(date -u +%FT%TZ) breakeven sweep exit $?" | tee -a "$L"
touch "$WORK_DIR/breakeven256_DONE"
echo "[rand256] $(date -u +%FT%TZ) ALL DONE" | tee -a "$L"
