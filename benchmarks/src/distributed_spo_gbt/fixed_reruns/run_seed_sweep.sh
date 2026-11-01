#!/bin/bash
# Seed sweep on the Table 8 folds: single-machine SPO (train_oblique_forest, paper flags) vs distributed W=1
# (train_distributed_gbt 10509539, paper flags + numeric 0/1 typing). Only --seed / threads differ from the paper.
# Usage: run_seed_sweep.sh <single|w1> <seed> <threads> <out_dir> [datasets...]   (resumable)
set -u
ARM=$1; SEED=$2; T=$3; OUT=$4; shift 4
F=${FOLDS:-$HOME/table8_ydf/folds}
SINGLE=${SINGLE:-$HOME/table8_ydf/bin/train_oblique_forest}
DIST=${DIST:-$HOME/table8_fullfix/bin/train_distributed_gbt}
DATASETS=${*:-$(ls $F)}
CSV=$OUT/results.csv; LOGD=$OUT/logs/${ARM}_s$SEED; mkdir -p $LOGD $OUT/work
[ -f $CSV ] || echo "dataset,fold,arm,seed,threads,rc,test_acc,test_auc,test_logloss,wall_s" > $CSV
for ds in $DATASETS; do
  NF=$(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))['indices']))" $F/$ds/folds.json)
  for ((k=0; k<NF; k++)); do
    grep -q "^$ds,$k,$ARM,$SEED,[^,]*,0," $CSV && continue
    TRAIN=$F/$ds/fold${k}_train.csv; TEST=$F/$ds/fold${k}_test.csv; LOG=$LOGD/${ds}_f$k.log; t0=$(date +%s)
    if [ $ARM = single ]; then
      $SINGLE --input_mode csv --train_csv $TRAIN --label_col label --test_csv $TEST --num_trees=300 --tree_depth=6 \
        --min_examples=5 --num_threads=$T --seed=$SEED --feature_split_type Oblique --numerical_split_type Exact \
        --ensemble_method Boosting --num_projections_exponent 1 --max_num_projections 2147483647 \
        --compute_oob_performances=false > $LOG 2>&1
      rc=$?
      m=$(grep -o "test-accuracy:[^ ]* test-auc:[^ ]* test-logloss:[^ ]*" $LOG | tail -1 | sed 's/test-[a-z]*://g; s/ /,/g')
    else
      WD=$OUT/work/${ARM}_s${SEED}_${ds}_f$k; rm -rf $WD
      $DIST --alsologtostderr --train_csv $TRAIN --test_csv $TEST --label label --work_dir $WD --num_workers 1 \
        --num_threads $T --num_trees 300 --max_depth 6 --split_axis SPARSE_OBLIQUE \
        --sparse_oblique_projection_density_factor 1.5 --sparse_oblique_num_projections_exponent 1 \
        --sparse_oblique_max_num_projections 2147483647 --sparse_oblique_local_projection_count=true \
        --max_unique_values_for_discretized_numerical=1 --worker_logs=false --force_numerical_features=true \
        --seed $SEED > $LOG 2>&1
      rc=$?; rm -rf $WD
      acc=$(grep -m1 -E "^Accuracy:" $LOG | awk '{print $2}')
      auc=$(grep -m1 -E "^\s*auc:" $LOG | awk '{print $2}')
      ll=$(grep -m1 -E "^LogLoss:" $LOG | sed -E 's/LogLoss: *:? *//' | awk '{print $1}')
      m="$acc,$auc,$ll"
    fi
    echo "$ds,$k,$ARM,$SEED,$T,$rc,${m:-,,},$(( $(date +%s) - t0 ))" >> $CSV
  done
done
echo "### DONE $ARM s$SEED $(date -u +%FT%TZ)"
