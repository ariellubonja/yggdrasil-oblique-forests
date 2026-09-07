#!/usr/bin/env bash
# 256-bin follow-up (2026-09-07): the AVX-512 256-threshold histogram arms
# (arms.py ARMS_256, "default" binary) on the same suite + huge tier as the
# 64-bin accuracy comparison. Appends to the same result CSVs (skip-existing
# keyed on dataset,fold,rep,method,seed). Shares run_all.sh's flock so it can
# never overlap the main chain. Smoke test (AVX-512 vs scalar bit-identity at
# 256 bins, compare_models.sh) was done by hand before this: TREES IDENTICAL.
set -uo pipefail
WORK_DIR=/home/ubuntu/spo_vs_gbt
REPO_DIR=/home/ubuntu/yggdrasil-oblique-forests
SCRIPT_DIR="$REPO_DIR/benchmarks/evaluation/spo_vs_gbt"
PY=/home/ubuntu/gbt_venv/bin/python
LOG_DIR="$WORK_DIR/logs"
RESULTS_DIR="$WORK_DIR/results"
exec 9>"$WORK_DIR/run_all.lock"
flock -n 9 || { echo "run_all.lock held; exiting" >&2; exit 1; }

ARMS256="spo_rf_rand256_vec,spo_rf_dyn256_vec,spo_gbt_dyn256_vec"
REPO_DATA="$REPO_DIR/benchmarks/data"
echo "[run_256] $(date -u +%FT%TZ) start" | tee -a "$LOG_DIR/run_256.log"

"$PY" "$SCRIPT_DIR/run_suite.py" --mode suite \
  --datasets-dir "$REPO_DATA/tabarena_binary_csv" --datasets-dir "$REPO_DATA/tabred_binary_csv" \
  --folds 5 --fold-seed 0 --chrono-holdout "ecom-offers,homecredit-default,homesite-insurance" \
  --work-dir "$WORK_DIR/folds" --out "$RESULTS_DIR/suite_results.csv" --threads 48 \
  --arms "$ARMS256" > "$LOG_DIR/suite256.log" 2>&1
echo "[run_256] $(date -u +%FT%TZ) suite exit $?" | tee -a "$LOG_DIR/run_256.log"

"$PY" "$SCRIPT_DIR/run_suite.py" --mode large \
  --dataset "HIGGS=$REPO_DATA/HIGGS_train_10500k.csv:$REPO_DATA/HIGGS_test_500k.csv:class" \
  --dataset "SUSY=$REPO_DATA/SUSY_train_4500k.csv:$REPO_DATA/SUSY_test_500k.csv:class" \
  --dataset "EPSILON=$REPO_DATA/epsilon_normalized_train.csv:$WORK_DIR/data/epsilon_test_100k.csv:label" \
  --seed 1 --timeout-s 10800 --out "$RESULTS_DIR/large_results.csv" --threads 48 \
  --arms "$ARMS256" > "$LOG_DIR/large256.log" 2>&1
echo "[run_256] $(date -u +%FT%TZ) large exit $?" | tee -a "$LOG_DIR/run_256.log"

cp -f "$RESULTS_DIR/suite_results.csv" "$RESULTS_DIR/large_results.csv" "$REPO_DIR/benchmarks/results/spo_vs_gbt/"
touch "$WORK_DIR/bins256_DONE"
echo "[run_256] $(date -u +%FT%TZ) ALL DONE" | tee -a "$LOG_DIR/run_256.log"
