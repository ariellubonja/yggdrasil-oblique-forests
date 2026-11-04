#!/bin/bash
# CC18 accuracy study, single-machine GBT: AA vs SPO (harness defaults) vs SPO (YDF defaults) vs AA+SPO (YDF defaults).
# 34 binary CC18 tasks x 10 folds (numeric copies from prep_cc18_fold0.py --folds 0-9), seed 1. Resumable.
# One run per cell gives both numbers: train_s is timed before the test-set evaluation; the model is saved to models/.
# Run one cell at a time (no parallel shards) when train_s is reported.
# Usage: run_cc18_aa_spo_gbt.sh <out_dir> [bin]   (out_dir holds data/, logs/, results.csv)
# Values: ../distributed_spo_gbt/params/kdd28_params.json profile cc18_aa_spo (one arm per row label).
set -u
OUT=${1:?out_dir}
BIN=${2:-$OUT/train_oblique_forest}
DATA=$OUT/data
CSV=$OUT/results.csv
PARAMS=$(cd "$(dirname "$0")" && pwd)/../distributed_spo_gbt/params/params.py
. "$(dirname "$PARAMS")/drift.sh"  # run_cell: pre-flight per arm + per-cell drift stop (ALLOW_DRIFT=1 warns)
eval "$(python3 "$PARAMS" export cc18_aa_spo Q_)"
eval "$(python3 "$PARAMS" export cc18_aa_spo.spo_harness B_)"
eval "$(python3 "$PARAMS" export cc18_aa_spo.spo_harness_exact300 C_)"
python3 "$PARAMS" require-stamped $CSV || exit 2
FOLDS=${FOLDS:-"$Q_FOLDS"}
ARMS=${ARMS:-"aa_ydf spo_harness spo_ydf aa_spo_ydf"}
mkdir -p $OUT/logs $OUT/models
GIT=$(git -C "$(dirname "$0")" rev-parse --short HEAD)$(git -C "$(dirname "$0")" diff --quiet HEAD || echo -dirty)
BIN_SHA=$(sha256sum $BIN 2>/dev/null || shasum -a 256 $BIN); BIN_SHA=${BIN_SHA:0:12}
[ -f $CSV ] || echo "dataset,fold,arm,rc,test_acc,test_auc,test_logloss,num_trees,train_s,wall_s,git,bin_sha,$(python3 "$PARAMS" stamp-header dataset,fold,arm,rc,test_acc,test_auc,test_logloss,num_trees,train_s,wall_s,git,bin_sha)" > $CSV

arm_args() {
  case $1 in
    aa_ydf)      ARGS=(--ydf_defaults "--feature_split_type=Axis Aligned") ;;
    # Recorded harness SPO (P=F, max 1000, density 1.5, no ES), pinned now that the flag defaults are YDF's.
    # Trees (30) and split type (Dynamic Random Histogram) stay the current harness defaults for Boosting.
    spo_harness) ARGS=(--feature_split_type=Oblique --num_projections_exponent $B_SPO_EXPONENT
                       --max_num_projections $B_SPO_MAX_PROJECTIONS --projection_density_factor $B_SPO_DENSITY) ;;
    spo_harness_exact300) ARGS=(--feature_split_type=Oblique --num_trees $C_TREES "--numerical_split_type=$C_SPLIT_FLAG"
                       --num_projections_exponent $C_SPO_EXPONENT --max_num_projections $C_SPO_MAX_PROJECTIONS
                       --projection_density_factor $C_SPO_DENSITY) ;;
    spo_ydf)     ARGS=(--ydf_defaults --feature_split_type=Oblique) ;;
    aa_spo_ydf)  ARGS=(--ydf_defaults --feature_split_type=Oblique --oblique_also_axis_aligned) ;;
    *) echo "unknown arm $1" >&2; exit 2 ;;
  esac
}

for k in $FOLDS; do
  for dpath in $(ls -d $DATA/*/ | sort); do
    d=$(basename $dpath)
    lbl=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['label_col'])" $dpath/meta.json)
    for arm in $ARMS; do
      grep -q "^$d,$k,$arm,0," $CSV && continue
      log=$OUT/logs/${d}__f${k}__${arm}.log
      arm_args $arm
      t0=$(date +%s)
      run_cell cc18_aa_spo.$arm $log $BIN --stderrthreshold=0 --input_mode csv --train_csv $dpath/fold${k}_train.csv \
        --test_csv $dpath/fold${k}_test.csv --label_col "$lbl" --ensemble_method Boosting \
        --seed $Q_SEEDS --model_out_dir $OUT/models/${d}__f${k}__${arm} "${ARGS[@]}"
      rc=$?
      m=$(grep -o "test-accuracy:[^ ]* test-auc:[^ ]* test-logloss:[^ ]*" $log | tail -1 | sed 's/test-[a-z]*://g; s/ /,/g')
      nt=$(grep -o "Final model num-trees:[0-9]*" $log | tail -1 | cut -d: -f2)
      [ -z "$nt" ] && nt=$(grep -o "Train tree [0-9]*/" $log | tail -1 | tr -dc 0-9)
      ts=$(grep -o "Training block took: [0-9.e+-]*" $log | tail -1 | awk '{print $4}')
      echo "$d,$k,$arm,$rc,${m:-,,},$nt,$ts,$(( $(date +%s) - t0 )),$GIT,$BIN_SHA,$(python3 "$PARAMS" stamp "cc18_aa_spo.$arm$DRIFT" $log)" >> $CSV
      echo "$(date +%H:%M:%S) f$k $d $arm rc=$rc acc=${m%%,*} trees=$nt"
    done
  done
done
