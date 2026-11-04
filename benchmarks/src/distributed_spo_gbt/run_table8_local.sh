#!/bin/bash
# KDD28 Table 8 rerun, in-process (MULTI_THREAD workers) on one machine: accuracy only, runtime not comparable.
# Arms = run_dgbt_suite.sh distributed cells (W up to 32) + the two single-machine arms of speedup_map suite_results.csv
# (recorded argv, threads aside). Datasets outer, folds, arms inner. Resumable.
# Usage: run_table8_local.sh <out_dir>   (out_dir holds folds/, bin/, logs/, results.csv; env DATASETS, FOLDS, ARMS,
#        NCPU, DIST_EXTRA = extra flags for the distributed arms, DIST_PROFILE = their named override in params/)
# Values: params/kdd28_params.json profile table8 (arms single_spo, single_aa, dist_spo, dist_aa).
set -u
OUT=${1:?out_dir}
FOLDS_DIR=$OUT/folds
CSV=$OUT/results.csv
PARAMS=$(cd "$(dirname "$0")" && pwd)/params/params.py
. "$(dirname "$PARAMS")/drift.sh"  # run_cell: pre-flight + per-cell drift stop (ALLOW_DRIFT=1 warns)
python3 "$PARAMS" require-stamped "$CSV" || exit 2
eval "$(python3 "$PARAMS" export table8.dist_spo D_)"
NCPU=${NCPU:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}
ARMS=${ARMS:-"single:SPO single:AA $(for w in $D_WORKERS; do printf '%s ' $w:SPO; done)$(for w in $D_WORKERS; do printf '%s ' $w:AA; done)"}
DATASETS=${DATASETS:-$(ls $FOLDS_DIR)}
GIT=$(git -C "$(dirname "$0")" rev-parse --short HEAD)$(git -C "$(dirname "$0")" diff --quiet HEAD || echo -dirty)
mkdir -p $OUT/logs $OUT/work
[ -f $CSV ] || echo "dataset,fold,arm,workers,split_axis,threads,rc,test_acc,test_auc,test_logloss,wall_s,git,$(python3 "$PARAMS" stamp-header dataset,fold,arm,workers,split_axis,threads,rc,test_acc,test_auc,test_logloss,wall_s,git)" > $CSV
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
        PROF=table8.single_spo; [ $A = AA ] && PROF=table8.single_aa
        eval "$(python3 "$PARAMS" export $PROF Q_)"
        # Density pinned (2026-11-03): the harness default became YDF's 2, the recorded cells used 1.5.
        if [ $A = SPO ]; then X=(--feature_split_type Oblique --num_projections_exponent $Q_SPO_EXPONENT --max_num_projections $Q_SPO_MAX_PROJECTIONS
                                  --projection_density_factor $Q_SPO_DENSITY)
        else X=(--feature_split_type "Axis Aligned" --num_candidate_attributes $Q_NUM_CANDIDATE_ATTRIBUTES); fi
        run_cell $PROF $LOG $OUT/bin/train_oblique_forest --input_mode csv --train_csv $TRAIN --label_col label --test_csv $TEST \
          --num_trees=$Q_TREES --tree_depth=$Q_MAX_DEPTH --min_examples=$Q_MIN_EXAMPLES --num_threads=$T --seed=$Q_SEEDS --numerical_split_type $Q_SPLIT_FLAG \
          --ensemble_method Boosting --compute_oob_performances=false "${X[@]}"
        rc=$?
        m=$(grep -o "test-accuracy:[^ ]* test-auc:[^ ]* test-logloss:[^ ]*" $LOG | tail -1 | sed 's/test-[a-z]*://g; s/ /,/g')
      else
        T=$(( NCPU / W )); [ $T -lt 1 ] && T=1
        AXIS=SPARSE_OBLIQUE; [ $A = AA ] && AXIS=AXIS_ALIGNED
        PROF=${DIST_PROFILE:-table8.dist_spo}; [ $A = AA ] && PROF=${DIST_AA_PROFILE:-table8.dist_aa}
        WD=$OUT/work/${ds}_f${k}_W${W}; rm -rf $WD
        # Legacy per-worker local count (recorded Table 8 cells); DIST_EXTRA comes last and may override it.
        # SPO flags on SPO cells only: AXIS_ALIGNED fails on them (train_distributed_gbt.cc, never ignore a flag).
        S1=(); [ $A = AA ] || S1=(--sparse_oblique_projection_density_factor $D_SPO_DENSITY --sparse_oblique_num_projections_exponent $D_SPO_EXPONENT
          --sparse_oblique_max_num_projections $D_SPO_MAX_PROJECTIONS --sparse_oblique_local_projection_count=$D_LOCAL_PROJECTION_COUNT
          --sparse_oblique_split_budget=$D_SPLIT_BUDGET)
        run_cell $PROF $LOG $OUT/bin/train_distributed_gbt --alsologtostderr --train_csv $TRAIN --test_csv $TEST --label label \
          --work_dir $WD --num_workers $W --num_threads $T --num_trees $D_TREES --max_depth $D_MAX_DEPTH --split_axis $AXIS ${S1[@]+"${S1[@]}"} \
          --max_unique_values_for_discretized_numerical=$D_MAX_UNIQUE_DISCRETIZED --worker_logs=false ${DIST_EXTRA:-}
        rc=$?; rm -rf $WD
        acc=$(grep -m1 -E "^Accuracy:" $LOG | awk '{print $2}')
        auc=$(grep -m1 -E "^\s*auc:" $LOG | awk '{print $2}')
        ll=$(grep -m1 -E "^LogLoss:" $LOG | sed -E 's/LogLoss: *:? *//' | awk '{print $1}')
        m="$acc,$auc,$ll"
      fi
      echo "$ds,$k,$arm,$W,$AXIS,$T,$rc,${m:-,,},$(( $(date +%s) - t0 )),$GIT,$(python3 "$PARAMS" stamp "$PROF$DRIFT" $LOG)" >> $CSV
      echo "$(date +%H:%M:%S) $ds f$k $arm rc=$rc $m"
    done
  done
done
echo "### DONE $(date -u +%FT%TZ)"
