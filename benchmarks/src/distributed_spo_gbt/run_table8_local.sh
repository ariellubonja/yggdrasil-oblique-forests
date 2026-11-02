#!/bin/bash
# KDD28 Table 8 rerun, in-process (MULTI_THREAD workers) on one machine: accuracy only, runtime not comparable.
# Arms = run_dgbt_suite.sh distributed cells (W up to 32) + the two single-machine arms of speedup_map suite_results.csv
# (recorded argv, threads aside). Datasets outer, folds, arms inner. Resumable.
# Usage: run_table8_local.sh <out_dir>   (out_dir holds folds/, bin/, logs/, results.csv; env DATASETS, FOLDS, ARMS,
#        NCPU, DIST_EXTRA = extra flags for the distributed arms)
set -u
OUT=${1:?out_dir}
FOLDS_DIR=$OUT/folds
CSV=$OUT/results.csv
NCPU=${NCPU:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}
ARMS=${ARMS:-"single:SPO single:AA 1:SPO 4:SPO 8:SPO 16:SPO 32:SPO 1:AA 4:AA 8:AA 16:AA 32:AA"}
DATASETS=${DATASETS:-$(ls $FOLDS_DIR)}
GIT=$(git -C "$(dirname "$0")" rev-parse --short HEAD)$(git -C "$(dirname "$0")" diff --quiet HEAD || echo -dirty)
mkdir -p $OUT/logs $OUT/work
[ -f $CSV ] || echo "dataset,fold,arm,workers,split_axis,threads,rc,test_acc,test_auc,test_logloss,wall_s,git" > $CSV
for ds in $DATASETS; do
  NF=$(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))['indices']))" $FOLDS_DIR/$ds/folds.json)
  for ((k=0; k<NF; k++)); do
    [ -n "${FOLDS:-}" ] && [[ " $FOLDS " != *" $k "* ]] && continue
    TRAIN=$FOLDS_DIR/$ds/fold${k}_train.csv; TEST=$FOLDS_DIR/$ds/fold${k}_test.csv
    for arm in $ARMS; do
      W=${arm%%:*}; A=${arm##*:}
      grep -q "^$ds,$k,$arm,[^,]*,[^,]*,[^,]*,0," $CSV && continue
      LOG=$OUT/logs/${ds}_f${k}_${W}_${A}.log; t0=$(date +%s)
      if [ $W = single ]; then
        T=$NCPU; AXIS=$A
        if [ $A = SPO ]; then X=(--feature_split_type Oblique --num_projections_exponent 1 --max_num_projections 2147483647)
        else X=(--feature_split_type "Axis Aligned" --num_candidate_attributes -1); fi
        $OUT/bin/train_oblique_forest --input_mode csv --train_csv $TRAIN --label_col label --test_csv $TEST \
          --num_trees=300 --tree_depth=6 --min_examples=5 --num_threads=$T --seed=1 --numerical_split_type Exact \
          --ensemble_method Boosting --compute_oob_performances=false "${X[@]}" > $LOG 2>&1
        rc=$?
        m=$(grep -o "test-accuracy:[^ ]* test-auc:[^ ]* test-logloss:[^ ]*" $LOG | tail -1 | sed 's/test-[a-z]*://g; s/ /,/g')
      else
        T=$(( NCPU / W )); [ $T -lt 1 ] && T=1
        AXIS=SPARSE_OBLIQUE; [ $A = AA ] && AXIS=AXIS_ALIGNED
        WD=$OUT/work/${ds}_f${k}_W${W}; rm -rf $WD
        $OUT/bin/train_distributed_gbt --alsologtostderr --train_csv $TRAIN --test_csv $TEST --label label \
          --work_dir $WD --num_workers $W --num_threads $T --num_trees 300 --max_depth 6 --split_axis $AXIS \
          --sparse_oblique_projection_density_factor 1.5 --sparse_oblique_num_projections_exponent 1 \
          --sparse_oblique_max_num_projections 2147483647 --sparse_oblique_local_projection_count=true \
          --max_unique_values_for_discretized_numerical=1 --worker_logs=false ${DIST_EXTRA:-} > $LOG 2>&1
        rc=$?; rm -rf $WD
        acc=$(grep -m1 -E "^Accuracy:" $LOG | awk '{print $2}')
        auc=$(grep -m1 -E "^\s*auc:" $LOG | awk '{print $2}')
        ll=$(grep -m1 -E "^LogLoss:" $LOG | sed -E 's/LogLoss: *:? *//' | awk '{print $1}')
        m="$acc,$auc,$ll"
      fi
      echo "$ds,$k,$arm,$W,$AXIS,$T,$rc,${m:-,,},$(( $(date +%s) - t0 )),$GIT" >> $CSV
      echo "$(date +%H:%M:%S) $ds f$k $arm rc=$rc $m"
    done
  done
done
echo "### DONE $(date -u +%FT%TZ)"
