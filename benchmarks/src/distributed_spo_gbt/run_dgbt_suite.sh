#!/usr/bin/env bash
# Table-9-style accuracy suite for the in-worker distributed GBT (2026-10-12).
#   run_dgbt_suite.sh            (env: DATASETS, CELLS="W:AXIS ...", TREES=300, THREADS=6)
# Datasets outer, folds next, cells inner; the fold CSVs are the paper's
# (/home/ubuntu/spo_vs_gbt/folds/<ds>/fold<k>_{train,test}.csv, label col from folds.json;
# chrono datasets have one fold). Settings match the paper's GBT arms (300 trees, depth 6,
# min_examples 5, shrinkage 0.1, no validation split) and Table 9's SPO-GBT arm: P = F
# projections per node in total (--sparse_oblique_num_projections_exponent 1 with
# --sparse_oblique_local_projection_count=true, so W workers sample F/W each from their own
# F/W columns), cap lifted (INT_MAX; Bioresponse F=1776 > proto default 1000), density 1.5,
# BINARY weights. Non-default, stated: --max_unique_values_for_discretized_numerical=1 (exact
# float columns; the cache's default discretisation (<=16000 uniques -> bin midpoints) cost
# SPO 0.01-0.035 AUC on the qsar-biodeg fold-0 smoke and does nothing for AA). Threads per
# worker = driver default 6. Appends to suite_results.csv.
# Values: params/kdd28_params.json profile table8 (arms suite_spo, suite_aa).
set -uo pipefail
PARAMS=$(cd "$(dirname "$0")" && pwd)/params/params.py
. "$(dirname "$PARAMS")/drift.sh"  # run_cell: pre-flight + per-cell drift stop (ALLOW_DRIFT=1 warns)
eval "$(python3 "$PARAMS" export table8.suite_spo Q_)"
REPO=/home/ubuntu/yggdrasil-oblique-forests
BIN=$REPO/bazel-bin/examples/train_distributed_gbt
OUT=$REPO/benchmarks/results/runtime/distributed_spo_gbt
FOLDS=/home/ubuntu/spo_vs_gbt/folds
WORK=${WORK:-/home/ubuntu/dgbt_work/suite}
RESULTS=${RESULTS:-$OUT/suite_results.csv}
TREES=${TREES:-$Q_TREES}; MAX_DEPTH=${MAX_DEPTH:-$Q_MAX_DEPTH}
# Threads per worker: max(6, 48/W) (6 = driver default). Runtime-only: the distributed oblique
# finder seeds its RNG per (seed, node), so results are thread-count independent (verified
# bit-identical at 6 vs 48 threads on qsar-biodeg fold 0, 2026-10-12). At the default 6 the
# 1-worker P=F cells were ~40x slower than the paper's 48-thread arm (APSFailure 683 s/fold).
threads_for() { local t=$((48 / $1)); [ $t -lt 6 ] && t=6; echo $t; }
# Resume: skip a cell whose (dataset, fold, workers, split_axis, trees) row already has rc=0.
CELLS=${CELLS:-"$(for w in $Q_WORKERS; do printf '%s ' $w:SPARSE_OBLIQUE; done)$(for w in $Q_WORKERS; do printf '%s ' $w:AXIS_ALIGNED; done)"}
DATASETS=${DATASETS:-$(ls $FOLDS)}
GITSHA=$(git -C "$REPO" rev-parse --short HEAD)
python3 "$PARAMS" require-stamped "$RESULTS" || exit 2
mkdir -p "$OUT/logs/suite" "$WORK"
[ -f "$RESULTS" ] || echo "dataset,fold,workers,threads_per_worker,split_axis,trees,max_depth,rc,train_took,wall_s,test_acc,test_auc,test_logloss,git,log,$(python3 "$PARAMS" stamp-header dataset,fold,workers,threads_per_worker,split_axis,trees,max_depth,rc,train_took,wall_s,test_acc,test_auc,test_logloss,git,log)" > "$RESULTS"
for ds in $DATASETS; do
  LABEL=$(python3 -c "import json;print(json.load(open('$FOLDS/$ds/folds.json'))['label_col'])")
  NF=$(python3 -c "import json;print(len(json.load(open('$FOLDS/$ds/folds.json'))['indices']))")
  for ((k=0; k<NF; k++)); do
    TRAIN=$FOLDS/$ds/fold${k}_train.csv; TEST=$FOLDS/$ds/fold${k}_test.csv
    for cell in $CELLS; do
      W=${cell%%:*}; AXIS=${cell##*:}; THREADS=${THREADS_OVERRIDE:-$(threads_for $W)}
      if grep -q "^$ds,$k,$W,[0-9]*,$AXIS,$TREES,$MAX_DEPTH,0," "$RESULTS"; then echo "### skip $ds f$k W=$W $AXIS (done)"; continue; fi
      STAMP=$(date -u +%Y%m%dT%H%M%SZ); TAG="${ds}_f${k}_W${W}_${AXIS}_T${TREES}"
      LOG=$OUT/logs/suite/${TAG}.log
      WD=$WORK/${ds}_f${k}_W${W}
      rm -rf "$WD"
      echo "### $STAMP START $TAG"
      T0=$(date +%s.%N)
      PROF=table8.suite_spo; [ $AXIS = AXIS_ALIGNED ] && PROF=table8.suite_aa
      # Legacy per-worker local count (recorded Table 9 cells): pin split budget off, else the guard rejects it.
      # SPO flags on SPO cells only: AXIS_ALIGNED fails on them (train_distributed_gbt.cc, never ignore a flag).
      S1=(); [ $AXIS = AXIS_ALIGNED ] || S1=(--sparse_oblique_projection_density_factor $Q_SPO_DENSITY
        --sparse_oblique_num_projections_exponent $Q_SPO_EXPONENT --sparse_oblique_max_num_projections $Q_SPO_MAX_PROJECTIONS
        --sparse_oblique_local_projection_count=$Q_LOCAL_PROJECTION_COUNT --sparse_oblique_split_budget=$Q_SPLIT_BUDGET)
      DRIFT_EXPECT="trees=$TREES max_depth=$MAX_DEPTH" run_cell $PROF "$LOG" "$BIN" --alsologtostderr --train_csv "$TRAIN" --test_csv "$TEST" --label "$LABEL" \
        --work_dir "$WD" --num_workers "$W" --num_threads "$THREADS" \
        --num_trees "$TREES" --max_depth "$MAX_DEPTH" --split_axis "$AXIS" ${S1[@]+"${S1[@]}"} \
        --max_unique_values_for_discretized_numerical=$Q_MAX_UNIQUE_DISCRETIZED \
        --worker_logs=false
      RC=$?
      T1=$(date +%s.%N); WALL=$(python3 -c "print(round($T1-$T0,2))")
      TRAIN_S=$(grep -o "Training (incl. cache creation) took [0-9.a-zµ]*" "$LOG" | tail -1 | awk '{print $NF}')
      ACC=$(grep -m1 -E "^Accuracy:" "$LOG" | awk '{print $2}')
      AUC=$(grep -m1 -E "^\s*auc:" "$LOG" | awk '{print $2}')
      LOGLOSS=$(grep -m1 -E "^LogLoss:" "$LOG" | sed -E 's/LogLoss: *:? *//' | awk '{print $1}')
      echo "$ds,$k,$W,$THREADS,$AXIS,$TREES,$MAX_DEPTH,$RC,$TRAIN_S,$WALL,$ACC,$AUC,$LOGLOSS,$GITSHA,$(basename "$LOG"),$(python3 "$PARAMS" stamp "$PROF$DRIFT" "$LOG")" >> "$RESULTS"
      echo "### $(date -u +%FT%TZ) END   $TAG rc=$RC wall=${WALL}s acc=$ACC auc=$AUC"
      rm -rf "$WD"
    done
  done
done
echo "### SUITE_DONE $(date -u +%FT%TZ)"
