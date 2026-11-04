#!/bin/bash
# Seed sweep on the Table 8 folds: single-machine SPO (train_oblique_forest, paper flags) vs distributed W=1
# (train_distributed_gbt 10509539, paper flags + numeric 0/1 typing). Only --seed / threads differ from the paper.
# Usage: run_seed_sweep.sh <single|w1> <seed> <threads> <out_dir> [datasets...]   (resumable)
# Values: params/kdd28_params.json table8 arms seed_sweep_single, seed_sweep_w1.
set -u
ARM=$1; SEED=$2; T=$3; OUT=$4; shift 4
PARAMS=$(cd "$(dirname "$0")" && pwd)/../params/params.py
. "$(dirname "$PARAMS")/drift.sh"  # run_cell: pre-flight + per-cell drift stop (ALLOW_DRIFT=1 warns)
eval "$(python3 "$PARAMS" export table8.seed_sweep_single S_)"
eval "$(python3 "$PARAMS" export table8.seed_sweep_w1 W_)"
F=${FOLDS:-$HOME/table8_ydf/folds}
SINGLE=${SINGLE:-$HOME/table8_ydf/bin/train_oblique_forest}
DIST=${DIST:-$HOME/table8_fullfix/bin/train_distributed_gbt}
DATASETS=${*:-$(ls $F)}
CSV=$OUT/results.csv; LOGD=$OUT/logs/${ARM}_s$SEED; mkdir -p $LOGD $OUT/work
python3 "$PARAMS" require-stamped $CSV || exit 2
[ -f $CSV ] || echo "dataset,fold,arm,seed,threads,rc,test_acc,test_auc,test_logloss,wall_s,$(python3 "$PARAMS" stamp-header dataset,fold,arm,seed,threads,rc,test_acc,test_auc,test_logloss,wall_s)" > $CSV
PROF=$S_PROFILE; [ $ARM = single ] || PROF=$W_PROFILE
for ds in $DATASETS; do
  NF=$(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))['indices']))" $F/$ds/folds.json)
  for ((k=0; k<NF; k++)); do
    grep -q "^$ds,$k,$ARM,$SEED,[^,]*,0," $CSV && continue
    TRAIN=$F/$ds/fold${k}_train.csv; TEST=$F/$ds/fold${k}_test.csv; LOG=$LOGD/${ds}_f$k.log; t0=$(date +%s)
    if [ $ARM = single ]; then
      run_cell $PROF $LOG $SINGLE --input_mode csv --train_csv $TRAIN --label_col label --test_csv $TEST --num_trees=$S_TREES --tree_depth=$S_MAX_DEPTH \
        --min_examples=$S_MIN_EXAMPLES --num_threads=$T --seed=$SEED --feature_split_type Oblique --numerical_split_type $S_SPLIT_FLAG \
        --ensemble_method Boosting --num_projections_exponent $S_SPO_EXPONENT --max_num_projections $S_SPO_MAX_PROJECTIONS \
        --projection_density_factor $S_SPO_DENSITY --compute_oob_performances=false  # density pinned: harness default is now 2
      rc=$?
      m=$(grep -o "test-accuracy:[^ ]* test-auc:[^ ]* test-logloss:[^ ]*" $LOG | tail -1 | sed 's/test-[a-z]*://g; s/ /,/g')
    else
      WD=$OUT/work/${ARM}_s${SEED}_${ds}_f$k; rm -rf $WD
      # Legacy per-worker local count (recorded cells); the split-budget pin keeps newer binaries off the guard.
      run_cell $PROF $LOG $DIST --alsologtostderr --train_csv $TRAIN --test_csv $TEST --label label --work_dir $WD --num_workers $W_WORKERS \
        --num_threads $T --num_trees $W_TREES --max_depth $W_MAX_DEPTH --split_axis SPARSE_OBLIQUE \
        --sparse_oblique_projection_density_factor $W_SPO_DENSITY --sparse_oblique_num_projections_exponent $W_SPO_EXPONENT \
        --sparse_oblique_max_num_projections $W_SPO_MAX_PROJECTIONS --sparse_oblique_local_projection_count=$W_LOCAL_PROJECTION_COUNT \
        --sparse_oblique_split_budget=$W_SPLIT_BUDGET \
        --max_unique_values_for_discretized_numerical=$W_MAX_UNIQUE_DISCRETIZED --worker_logs=false --force_numerical_features=$W_FORCE_NUMERICAL \
        --seed $SEED
      rc=$?; rm -rf $WD
      acc=$(grep -m1 -E "^Accuracy:" $LOG | awk '{print $2}')
      auc=$(grep -m1 -E "^\s*auc:" $LOG | awk '{print $2}')
      ll=$(grep -m1 -E "^LogLoss:" $LOG | sed -E 's/LogLoss: *:? *//' | awk '{print $1}')
      m="$acc,$auc,$ll"
    fi
    echo "$ds,$k,$ARM,$SEED,$T,$rc,${m:-,,},$(( $(date +%s) - t0 )),$(python3 "$PARAMS" stamp "$PROF$DRIFT" $LOG)" >> $CSV
  done
done
echo "### DONE $ARM s$SEED $(date -u +%FT%TZ)"
