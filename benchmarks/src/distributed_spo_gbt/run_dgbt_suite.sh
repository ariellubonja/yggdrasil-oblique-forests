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
set -uo pipefail
REPO=/home/ubuntu/yggdrasil-oblique-forests
BIN=$REPO/bazel-bin/examples/train_distributed_gbt
OUT=$REPO/benchmarks/results/runtime/distributed_spo_gbt
FOLDS=/home/ubuntu/spo_vs_gbt/folds
WORK=${WORK:-/home/ubuntu/dgbt_work/suite}
RESULTS=${RESULTS:-$OUT/suite_results.csv}
TREES=${TREES:-300}; MAX_DEPTH=${MAX_DEPTH:-6}
# Threads per worker: max(6, 48/W) (6 = driver default). Runtime-only: the distributed oblique
# finder seeds its RNG per (seed, node), so results are thread-count independent (verified
# bit-identical at 6 vs 48 threads on qsar-biodeg fold 0, 2026-10-12). At the default 6 the
# 1-worker P=F cells were ~40x slower than the paper's 48-thread arm (APSFailure 683 s/fold).
threads_for() { local t=$((48 / $1)); [ $t -lt 6 ] && t=6; echo $t; }
# Resume: skip a cell whose (dataset, fold, workers, split_axis, trees) row already has rc=0.
CELLS=${CELLS:-"1:SPARSE_OBLIQUE 4:SPARSE_OBLIQUE 8:SPARSE_OBLIQUE 16:SPARSE_OBLIQUE 1:AXIS_ALIGNED 4:AXIS_ALIGNED 8:AXIS_ALIGNED 16:AXIS_ALIGNED"}
DATASETS=${DATASETS:-$(ls $FOLDS)}
GITSHA=$(git -C "$REPO" rev-parse --short HEAD)
mkdir -p "$OUT/logs/suite" "$WORK"
[ -f "$RESULTS" ] || echo "dataset,fold,workers,threads_per_worker,split_axis,trees,max_depth,rc,train_took,wall_s,test_acc,test_auc,test_logloss,git,log" > "$RESULTS"
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
      "$BIN" --alsologtostderr --train_csv "$TRAIN" --test_csv "$TEST" --label "$LABEL" \
        --work_dir "$WD" --num_workers "$W" --num_threads "$THREADS" \
        --num_trees "$TREES" --max_depth "$MAX_DEPTH" --split_axis "$AXIS" \
        --sparse_oblique_projection_density_factor 1.5 \
        --sparse_oblique_num_projections_exponent 1 \
        --sparse_oblique_max_num_projections 2147483647 \
        --sparse_oblique_local_projection_count=true \
        --max_unique_values_for_discretized_numerical=1 \
        --worker_logs=false > "$LOG" 2>&1
      RC=$?
      T1=$(date +%s.%N); WALL=$(python3 -c "print(round($T1-$T0,2))")
      TRAIN_S=$(grep -o "Training (incl. cache creation) took [0-9.a-zµ]*" "$LOG" | tail -1 | awk '{print $NF}')
      ACC=$(grep -m1 -E "^Accuracy:" "$LOG" | awk '{print $2}')
      AUC=$(grep -m1 -E "^\s*auc:" "$LOG" | awk '{print $2}')
      LOGLOSS=$(grep -m1 -E "^LogLoss:" "$LOG" | sed -E 's/LogLoss: *:? *//' | awk '{print $1}')
      echo "$ds,$k,$W,$THREADS,$AXIS,$TREES,$MAX_DEPTH,$RC,$TRAIN_S,$WALL,$ACC,$AUC,$LOGLOSS,$GITSHA,$(basename "$LOG")" >> "$RESULTS"
      echo "### $(date -u +%FT%TZ) END   $TAG rc=$RC wall=${WALL}s acc=$ACC auc=$AUC"
      rm -rf "$WD"
    done
  done
done
echo "### SUITE_DONE $(date -u +%FT%TZ)"
