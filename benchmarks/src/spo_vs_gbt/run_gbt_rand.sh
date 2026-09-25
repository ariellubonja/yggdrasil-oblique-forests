#!/usr/bin/env bash
# Pure random-histogram GBT arms (2026-09-25): spo_gbt_rand_vec (64 bins) and
# spo_gbt_rand256_vec (256 bins) on the suite tier only, appended to
# suite_results.csv (skip-existing keyed on dataset,fold,rep,method,seed).
# Shares run_all.sh's flock so it can never overlap other training.
set -uo pipefail
WORK_DIR=/home/ubuntu/spo_vs_gbt
REPO_DIR=/home/ubuntu/yggdrasil-oblique-forests
SCRIPT_DIR="$REPO_DIR/benchmarks/src/spo_vs_gbt"
PY=/home/ubuntu/gbt_venv/bin/python
LOG_DIR="$WORK_DIR/logs"
RESULTS_DIR="$WORK_DIR/results"
exec 9>"$WORK_DIR/run_all.lock"
flock -n 9 || { echo "run_all.lock held; exiting" >&2; exit 1; }

ARMS="spo_gbt_rand_vec,spo_gbt_rand256_vec"
REPO_DATA="$REPO_DIR/benchmarks/data"
echo "[run_gbt_rand] $(date -u +%FT%TZ) start" | tee -a "$LOG_DIR/run_gbt_rand.log"

"$PY" "$SCRIPT_DIR/run_suite.py" --mode suite \
  --datasets-dir "$REPO_DATA/tabarena_binary_csv" --datasets-dir "$REPO_DATA/tabred_binary_csv" \
  --folds 5 --fold-seed 0 --chrono-holdout "ecom-offers,homecredit-default,homesite-insurance" \
  --work-dir "$WORK_DIR/folds" --out "$RESULTS_DIR/suite_results.csv" --threads 48 \
  --arms "$ARMS" > "$LOG_DIR/suite_gbt_rand.log" 2>&1
echo "[run_gbt_rand] $(date -u +%FT%TZ) suite exit $?" | tee -a "$LOG_DIR/run_gbt_rand.log"
cp -f "$RESULTS_DIR/suite_results.csv" "$REPO_DIR/benchmarks/results/runtime/speedup_map_by_dataset/"
touch "$WORK_DIR/gbt_rand_DONE"
echo "[run_gbt_rand] $(date -u +%FT%TZ) ALL DONE" | tee -a "$LOG_DIR/run_gbt_rand.log"
