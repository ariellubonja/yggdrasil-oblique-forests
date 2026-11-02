#!/bin/bash
# CC18 accuracy study, single-machine GBT: AA vs SPO (harness defaults) vs SPO (YDF defaults) vs AA+SPO (YDF defaults).
# 34 binary CC18 tasks x 10 folds (numeric copies from prep_cc18_fold0.py --folds 0-9), seed 1. Resumable.
# One run per cell gives both numbers: train_s is timed before the test-set evaluation; the model is saved to models/.
# Run one cell at a time (no parallel shards) when train_s is reported.
# Usage: run_cc18_aa_spo_gbt.sh <out_dir> [bin]   (out_dir holds data/, logs/, results.csv)
set -u
OUT=${1:?out_dir}
BIN=${2:-$OUT/train_oblique_forest}
DATA=$OUT/data
CSV=$OUT/results.csv
FOLDS=${FOLDS:-"0 1 2 3 4 5 6 7 8 9"}
ARMS=${ARMS:-"aa_ydf spo_harness spo_ydf aa_spo_ydf"}
mkdir -p $OUT/logs $OUT/models
GIT=$(git -C "$(dirname "$0")" rev-parse --short HEAD)$(git -C "$(dirname "$0")" diff --quiet HEAD || echo -dirty)
BIN_SHA=$(sha256sum $BIN 2>/dev/null || shasum -a 256 $BIN); BIN_SHA=${BIN_SHA:0:12}
[ -f $CSV ] || echo "dataset,fold,arm,rc,test_acc,test_auc,test_logloss,num_trees,train_s,wall_s,git,bin_sha" > $CSV

arm_args() {
  case $1 in
    aa_ydf)      ARGS=(--ydf_defaults "--feature_split_type=Axis Aligned") ;;
    spo_harness) ARGS=(--feature_split_type=Oblique) ;;
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
      $BIN --stderrthreshold=0 --input_mode csv --train_csv $dpath/fold${k}_train.csv \
        --test_csv $dpath/fold${k}_test.csv --label_col "$lbl" --ensemble_method Boosting \
        --seed 1 --model_out_dir $OUT/models/${d}__f${k}__${arm} "${ARGS[@]}" > $log 2>&1
      rc=$?
      m=$(grep -o "test-accuracy:[^ ]* test-auc:[^ ]* test-logloss:[^ ]*" $log | tail -1 | sed 's/test-[a-z]*://g; s/ /,/g')
      nt=$(grep -o "Final model num-trees:[0-9]*" $log | tail -1 | cut -d: -f2)
      [ -z "$nt" ] && nt=$(grep -o "Train tree [0-9]*/" $log | tail -1 | tr -dc 0-9)
      ts=$(grep -o "Training block took: [0-9.e+-]*" $log | tail -1 | awk '{print $4}')
      echo "$d,$k,$arm,$rc,${m:-,,},$nt,$ts,$(( $(date +%s) - t0 )),$GIT,$BIN_SHA" >> $CSV
      echo "$(date +%H:%M:%S) f$k $d $arm rc=$rc acc=${m%%,*} trees=$nt"
    done
  done
done
