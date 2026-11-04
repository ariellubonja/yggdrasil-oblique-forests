#!/bin/bash
# Table 2 single-machine SPO-GBT row: CC18 fold 0, 10 trees, depth 6, P = F, density 1.5, seeds 1-5 (170 cells).
# The recorded row (paper_refs/table2_single_machine_spo_gbt.csv) was built at commit 2c0990f0 and run on an
# Apple-silicon Mac with the default thread count: bit-reproducible only there (the job split depends on threads).
# Usage: run_single_machine_cc18.sh <train_oblique_forest> <cc18_fold0_dir> <out_dir>   (resumable)
# Values: params/kdd28_params.json profile table2.single_spo.
set -u
BIN=${1:?train_oblique_forest}; DATA=${2:?cc18_fold0_dir}; OUT=${3:?out_dir}
PARAMS=$(cd "$(dirname "$0")" && pwd)/../params/params.py
. "$(dirname "$PARAMS")/drift.sh"  # run_cell: pre-flight + per-cell drift stop (ALLOW_DRIFT=1 warns)
eval "$(python3 "$PARAMS" export table2.single_spo Q_)"
mkdir -p $OUT/logs
CSV=$OUT/results.csv
python3 "$PARAMS" require-stamped $CSV || exit 2
[ -f $CSV ] || echo "dataset,seed,rc,test_acc,test_auc,test_logloss,wall_s,git,$(python3 "$PARAMS" stamp-header dataset,seed,rc,test_acc,test_auc,test_logloss,wall_s,git)" > $CSV
for d in $(ls -d $DATA/task_* | xargs -n1 basename); do
  for s in $Q_SEEDS; do
    grep -q "^$d,$s,0," $CSV && continue
    log=$OUT/logs/${d}__s$s.log; t0=$(date +%s)
    run_cell $Q_PROFILE $log $BIN --stderrthreshold=0 --input_mode csv --train_csv $DATA/$d/train.csv --test_csv $DATA/$d/test.csv --label_col label \
      --ensemble_method Boosting --feature_split_type Oblique --numerical_split_type $Q_SPLIT_FLAG \
      --num_trees $Q_TREES --tree_depth $Q_MAX_DEPTH --num_projections_exponent $Q_SPO_EXPONENT --max_num_projections $Q_SPO_MAX_PROJECTIONS \
      --projection_density_factor $Q_SPO_DENSITY --seed $s
    rc=$?
    m=$(grep -o "test-accuracy:[^ ]* test-auc:[^ ]* test-logloss:[^ ]*" $log | tail -1 | sed 's/test-[a-z]*://g; s/ /,/g')
    echo "$d,$s,$rc,${m:-,,},$(( $(date +%s) - t0 )),${GIT:-},$(python3 "$PARAMS" stamp "$Q_PROFILE$DRIFT" $log)" >> $CSV
    echo "$d s$s rc=$rc $m"
  done
done
