#!/usr/bin/env bash
# 2026-09-09: re-run every spo_rf_dyn_scalar cell with the corrected threshold (4600,
# arms.py _DYN64_SCALAR). Waits for the study lock (the B7 chain holds it until purity
# is done), strips the threshold-250 rows into results/_tests/, then: large tier,
# B7 map cells (all 5 depths), suite. Appends to the same CSVs (skip-existing).
set -uo pipefail
WORK_DIR=/home/ubuntu/spo_vs_gbt; REPO_DIR=/home/ubuntu/yggdrasil-oblique-forests
SCRIPT_DIR="$REPO_DIR/benchmarks/evaluation/spo_vs_gbt"; PY=/home/ubuntu/gbt_venv/bin/python
LOG_DIR="$WORK_DIR/logs"; RESULTS_DIR="$WORK_DIR/results"; REPO_DATA="$REPO_DIR/benchmarks/data"
L="$LOG_DIR/run_b7_dynscalar.log"
exec 9>"$WORK_DIR/run_all.lock"
echo "[dynscalar] $(date -u +%FT%TZ) waiting for run_all.lock" | tee -a "$L"
flock 9
echo "[dynscalar] $(date -u +%FT%TZ) start" | tee -a "$L"
"$PY" "$SCRIPT_DIR/strip_dyn_scalar_thr250.py" "$RESULTS_DIR/speedup_map.csv" \
  "$RESULTS_DIR/large_results.csv" "$RESULTS_DIR/suite_results.csv" 2>&1 | tee -a "$L"

"$PY" "$SCRIPT_DIR/run_suite.py" --mode large \
  --dataset "HIGGS=$REPO_DATA/HIGGS_train_10500k.csv:$REPO_DATA/HIGGS_test_500k.csv:class" \
  --dataset "SUSY=$REPO_DATA/SUSY_train_4500k.csv:$REPO_DATA/SUSY_test_500k.csv:class" \
  --dataset "EPSILON=$REPO_DATA/epsilon_normalized_train.csv:$WORK_DIR/data/epsilon_test_100k.csv:label" \
  --seed 1 --timeout-s 10800 --out "$RESULTS_DIR/large_results.csv" --threads 48 \
  --arms spo_rf_dyn_scalar > "$LOG_DIR/large_dynscalar.log" 2>&1
echo "[dynscalar] $(date -u +%FT%TZ) large exit $?" | tee -a "$L"
cp -f "$RESULTS_DIR/large_results.csv" "$REPO_DIR/benchmarks/results/spo_vs_gbt/"

"$PY" "$SCRIPT_DIR/run_speedup_map.py" --cells-file "$SCRIPT_DIR/cells_b7_dynscalar.json" \
  --threads 48 --timeout 14400 --out "$RESULTS_DIR/speedup_map.csv" > "$LOG_DIR/b7_dynscalar.log" 2>&1
echo "[dynscalar] $(date -u +%FT%TZ) map exit $?" | tee -a "$L"
cp -f "$RESULTS_DIR/speedup_map.csv" "$REPO_DIR/benchmarks/results/spo_vs_gbt/"

"$PY" "$SCRIPT_DIR/run_suite.py" --mode suite \
  --datasets-dir "$REPO_DATA/tabarena_binary_csv" --datasets-dir "$REPO_DATA/tabred_binary_csv" \
  --folds 5 --fold-seed 0 --chrono-holdout "ecom-offers,homecredit-default,homesite-insurance" \
  --work-dir "$WORK_DIR/folds" --out "$RESULTS_DIR/suite_results.csv" --threads 48 \
  --arms spo_rf_dyn_scalar > "$LOG_DIR/suite_dynscalar.log" 2>&1
echo "[dynscalar] $(date -u +%FT%TZ) suite exit $?" | tee -a "$L"
cp -f "$RESULTS_DIR/suite_results.csv" "$REPO_DIR/benchmarks/results/spo_vs_gbt/"
touch "$WORK_DIR/b7_dynscalar_DONE"
echo "[dynscalar] $(date -u +%FT%TZ) ALL DONE" | tee -a "$L"
