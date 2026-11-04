#!/bin/bash
# Single-machine SPO-GBT baseline for the Vertex column-replication sweep (CC18 fold 0, 10 trees, depth 6, P=F).
set -u
cd "$(dirname "$0")/../../.."
OUT=ariel_notes/results/2026-10-29_local_spo_gbt_baseline
DATA=ariel_notes/results/2026-10-28_column_replication/data_cc18_fold0
BIN=$OUT/train_oblique_forest
mkdir -p $OUT/logs
CSV=$OUT/results.csv
[ -f $CSV ] || echo "dataset,seed,rc,test_acc,test_auc,test_logloss,wall_s,git" > $CSV
GIT=$(git rev-parse --short HEAD)
for d in $(ls -d $DATA/task_* | xargs -n1 basename); do
  for s in 1 2 3 4 5; do
    grep -q "^$d,$s,0," $CSV && continue
    log=$OUT/logs/${d}__s$s.log; t0=$(date +%s)
    $BIN --stderrthreshold=0 --input_mode csv --train_csv $DATA/$d/train.csv --test_csv $DATA/$d/test.csv --label_col label \
      --ensemble_method Boosting --feature_split_type Oblique --numerical_split_type Exact \
      --num_trees 10 --tree_depth 6 --num_projections_exponent 1 --max_num_projections 2147483647 \
      --projection_density_factor 1.5 --seed $s > $log 2>&1
    rc=$?
    m=$(grep -o "test-accuracy:[^ ]* test-auc:[^ ]* test-logloss:[^ ]*" $log | tail -1 | sed 's/test-[a-z]*://g; s/ /,/g')
    echo "$d,$s,$rc,${m:-,,},$(( $(date +%s) - t0 )),$GIT" >> $CSV
    echo "$d s$s rc=$rc $m"
  done
done
