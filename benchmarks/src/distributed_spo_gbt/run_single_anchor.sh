#!/usr/bin/env bash
# Same-boot single-machine SPO-GBT anchor (the paper's spo_gbt_exact_hwy arm, with test
# evaluation) for one natural dataset: run_single_anchor.sh DATASET [TREES] [Oblique|AxisAligned]
# Threads: harness default (-1 = all cores). Oblique weights: harness default (binary, density 1.5).
# Binary: the paper's /home/ubuntu/spo_vs_gbt/bin/default (icx, 6885c576). Appends to anchors.csv.
set -uo pipefail
DATASET=$1; TREES=${2:-300}; SPLIT=${3:-Oblique}
case $SPLIT in Oblique) ARM=spo_gbt_exact_hwy; FST=Oblique;; AxisAligned) ARM=aa_gbt_exact; FST='Axis Aligned';; *) echo bad split; exit 2;; esac
REPO=/home/ubuntu/yggdrasil-oblique-forests
BIN=${BIN:-/home/ubuntu/spo_vs_gbt/bin/default}
OUT=$REPO/benchmarks/results/runtime/distributed_spo_gbt
RESULTS=${RESULTS:-$OUT/anchors.csv}
D=$REPO/benchmarks/data
case $DATASET in
  SUSY)      TRAIN=$D/SUSY_train_4500k.csv;  TEST=$D/SUSY_test_500k.csv;  LABEL=class;;
  HIGGS)     TRAIN=$D/HIGGS_train_10500k.csv; TEST=$D/HIGGS_test_500k.csv; LABEL=class;;
  EPSILON)   TRAIN=$D/epsilon_normalized_train.csv; TEST=/home/ubuntu/spo_vs_gbt/data/epsilon_test_100k.csv; LABEL=label;;
  YOUTUBE8M) TRAIN=$D/youtube8m/youtube8m_video_train.csv; TEST=$D/youtube8m/youtube8m_video_validate.csv; LABEL=class;;
  *) echo "unknown dataset $DATASET"; exit 2;;
esac
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$OUT/logs/ANCHOR_${DATASET}_${ARM}_T${TREES}_${STAMP}.log
echo "=== $STAMP anchor $DATASET $ARM trees=$TREES bin=$BIN ($(cat $BIN.gitsha 2>/dev/null))" | tee "$LOG"
T0=$(date +%s.%N)
"$BIN" --input_mode csv --train_csv "$TRAIN" --test_csv "$TEST" --label_col "$LABEL" \
  --num_trees="$TREES" --seed=1 --feature_split_type "$FST" \
  --numerical_split_type Exact --ensemble_method Boosting --tree_depth=6 --min_examples=5 \
  --compute_oob_performances=false >> "$LOG" 2>&1
RC=$?
WALL=$(python3 -c "print(round($(date +%s.%N)-$T0,2))")
TRAIN_S=$(grep -o "Training block took:[[:space:]]*[0-9.eE+-]*" "$LOG" | tail -1 | awk '{print $NF}')
PRE_S=$(grep -o "Loading/Init (pre-train):[[:space:]]*[0-9.eE+-]*" "$LOG" | tail -1 | awk '{print $NF}')
POST_S=$(grep -o "Training + Post-Processing:[[:space:]]*[0-9.eE+-]*" "$LOG" | tail -1 | awk '{print $NF}')
read ACC AUC LL <<<"$(grep -o "test-accuracy:[0-9.eE+-]* *test-auc:-\?[0-9.eE+-]* *test-logloss:[0-9.eE+-]*" "$LOG" | tail -1 | sed -E 's/test-[a-z]+://g')"
[ -f "$RESULTS" ] || echo "dataset,arm,trees,max_depth,min_examples,threads,rc,train_s,pre_train_s,train_post_s,wall_s,test_acc,test_auc,test_logloss,bin_git,log" > "$RESULTS"
echo "$DATASET,$ARM,$TREES,6,5,-1,$RC,$TRAIN_S,$PRE_S,$POST_S,$WALL,${ACC:-},${AUC:-},${LL:-},$(cat $BIN.gitsha 2>/dev/null),$(basename "$LOG")" >> "$RESULTS"
echo "=== rc=$RC train_s=$TRAIN_S acc=${ACC:-} auc=${AUC:-} wall=${WALL}s" | tee -a "$LOG"
exit $RC
