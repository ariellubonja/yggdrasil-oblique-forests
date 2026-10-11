#!/usr/bin/env bash
# Table 9 fairness re-run (2026-10-10): SPO-GBT accuracy arms with P = F projections
# per node (arms.py ARMS_GBT_FULLPROJ, --num_projections_exponent 1 + an INT_MAX
# cap) so SPO-GBT scans as many candidates as aa_gbt_exact / XGBoost / LightGBM /
# CatBoost (all F features). Same suite protocol as run_gbt_rand.sh. New arm names
# (*_fullproj) coexist with the sqrt(F) rows in suite_results.csv.
#   tmux new-session -d -s spo_fullproj 'bash run_gbt_fullproj.sh'
set -uo pipefail
WORK_DIR=/home/ubuntu/spo_vs_gbt
REPO_DIR=/home/ubuntu/yggdrasil-oblique-forests
SCRIPT_DIR="$REPO_DIR/benchmarks/src/spaa27_experiments"
PY=/home/ubuntu/gbt_venv/bin/python
LOG_DIR="$WORK_DIR/logs"
RESULTS_DIR="$WORK_DIR/results"
exec 9>"$WORK_DIR/run_all.lock"
flock -n 9 || { echo "run_all.lock held; exiting" >&2; exit 1; }
ARMS="spo_gbt_exact_hwy_fullproj,spo_gbt_rand_vec_fullproj,spo_gbt_rand256_vec_fullproj,spo_gbt_rand_scalar_fullproj,spo_gbt_rand256_scalar_fullproj"
REPO_DATA="$REPO_DIR/benchmarks/data"
echo "[run_gbt_fullproj] $(date -u +%FT%TZ) start" | tee -a "$LOG_DIR/run_gbt_fullproj.log"
"$PY" "$SCRIPT_DIR/run_suite.py" --mode suite \
  --datasets-dir "$REPO_DATA/tabarena_binary_csv" --datasets-dir "$REPO_DATA/tabred_binary_csv" \
  --folds 5 --fold-seed 0 --chrono-holdout "ecom-offers,homecredit-default,homesite-insurance" \
  --work-dir "$WORK_DIR/folds" --out "$RESULTS_DIR/suite_results.csv" --threads 48 \
  --arms "$ARMS" > "$LOG_DIR/suite_gbt_fullproj.log" 2>&1
echo "[run_gbt_fullproj] $(date -u +%FT%TZ) suite exit $?" | tee -a "$LOG_DIR/run_gbt_fullproj.log"
cp -f "$RESULTS_DIR/suite_results.csv" "$REPO_DIR/benchmarks/results/runtime/speedup_map_by_dataset/"
touch "$WORK_DIR/gbt_fullproj_DONE"
echo "[run_gbt_fullproj] $(date -u +%FT%TZ) ALL DONE" | tee -a "$LOG_DIR/run_gbt_fullproj.log"
