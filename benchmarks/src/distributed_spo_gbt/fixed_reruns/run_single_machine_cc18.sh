#!/bin/bash
# Table 2 single-machine SPO-GBT row: CC18 fold 0, 10 trees, depth 6, P = F, density 1.5, seeds 1-5 (170 cells).
# The recorded row (paper_refs/table2_single_machine_spo_gbt.csv) was built at commit 2c0990f0 and run on an
# Apple-silicon Mac with the default thread count: bit-reproducible only there (the job split depends on threads).
# Usage: run_single_machine_cc18.sh <train_oblique_forest> <cc18_fold0_dir> <out_dir>   (resumable)
set -u
BIN=${1:?train_oblique_forest}; DATA=${2:?cc18_fold0_dir}; OUT=${3:?out_dir}
mkdir -p $OUT/logs
CSV=$OUT/results.csv
[ -f $CSV ] || echo "dataset,seed,rc,test_acc,test_auc,test_logloss,wall_s,git" > $CSV
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
    echo "$d,$s,$rc,${m:-,,},$(( $(date +%s) - t0 )),${GIT:-}" >> $CSV
    echo "$d s$s rc=$rc $m"
  done
done
