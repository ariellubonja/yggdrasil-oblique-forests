#!/usr/bin/env bash
# One cell of the in-worker distributed SPO-GBT test on the paper's natural datasets.
#   run_dgbt.sh DATASET NUM_WORKERS SPLIT_AXIS NUM_TREES [extra train_distributed_gbt flags]
# DATASET in SUSY|HIGGS|EPSILON|YOUTUBE8M, SPLIT_AXIS in SPARSE_OBLIQUE|AXIS_ALIGNED.
# Workers are in-process (MULTI_THREAD); threads per worker = 48 / NUM_WORKERS.
# Oblique settings match the single-machine spo_gbt_exact_hwy arm (train_oblique_forest
# defaults): BINARY weights (the harness leaves the proto field unset; the RF/GBT learners'
# SetDefaultHyperParameters then picks binary — training.cc "By default, we use binary weights";
# the driver's flag default is BINARY too), density 1.5 (harness flag default; YDF's is 2.0),
# P = ceil(F^0.5), depth 6, 0.1 shrinkage, min_examples 5, no validation split.
# The seven 300-tree oblique cells of 2026-10-11 ran with --sparse_oblique_weights CONTINUOUS by
# mistake (results.csv rows with extra=="" and trees==300). The dataset cache lives in WORK/DATASET and is
# reused across cells (first cell of a dataset pays its creation).
# Appends one row to RESULTS (CSV) and keeps the full log under logs/.
set -euo pipefail
DATASET=$1; W=$2; AXIS=$3; TREES=$4; shift 4
REPO=/home/ubuntu/yggdrasil-oblique-forests
BIN=$REPO/bazel-bin/examples/train_distributed_gbt
OUT=$REPO/benchmarks/results/runtime/distributed_spo_gbt
WORK=${WORK:-/home/ubuntu/dgbt_work}
RESULTS=${RESULTS:-$OUT/results.csv}
MAX_DEPTH=${MAX_DEPTH:-6}
THREADS=${THREADS:-$((48 / W))}
D=$REPO/benchmarks/data
# SUSY/HIGGS train CSVs spell the label 0.000000000000000000e+00 / 1.000000000000000000e+00 while the
# test CSVs spell it 0 / 1; a CATEGORICAL label compares tokens, so the test copies below re-spell
# the label the train way (features untouched; made 2026-10-11 by awk from benchmarks/data).
FIX=/home/ubuntu/dgbt_work/data
case $DATASET in
  SUSY)      TRAIN=$D/SUSY_train_4500k.csv;  TEST=$FIX/SUSY_test_500k_floatlabel.csv;  LABEL=class;;
  HIGGS)     TRAIN=$D/HIGGS_train_10500k.csv; TEST=$FIX/HIGGS_test_500k_floatlabel.csv; LABEL=class;;
  EPSILON)   TRAIN=$D/epsilon_normalized_train.csv; TEST=/home/ubuntu/spo_vs_gbt/data/epsilon_test_100k.csv; LABEL=label;;
  YOUTUBE8M) TRAIN=$D/youtube8m/youtube8m_video_train.csv; TEST=$D/youtube8m/youtube8m_video_validate.csv; LABEL=class;;
  *) echo "unknown dataset $DATASET"; exit 2;;
esac
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
TAG="${DATASET}_W${W}_${AXIS}_T${TREES}_d${MAX_DEPTH}"
LOG=$OUT/logs/${TAG}_${STAMP}.log
mkdir -p "$OUT/logs" "$WORK/$DATASET"
GITSHA=$(git -C "$REPO" rev-parse --short HEAD)
echo "=== $STAMP $TAG threads/worker=$THREADS extra=[$*] git=$GITSHA" | tee "$LOG"
T0=$(date +%s.%N)
set +e
"$BIN" --alsologtostderr --train_csv "$TRAIN" --test_csv "$TEST" --label "$LABEL" \
  --work_dir "$WORK/$DATASET" --num_workers "$W" --num_threads "$THREADS" \
  --num_trees "$TREES" --max_depth "$MAX_DEPTH" --split_axis "$AXIS" \
  --sparse_oblique_projection_density_factor 1.5 \
  --sparse_oblique_num_projections_exponent 0.5 --worker_logs=false "$@" >> "$LOG" 2>&1
RC=$?
set -e
T1=$(date +%s.%N)
WALL=$(python3 -c "print(round($T1-$T0,2))")
TRAIN_S=$(grep -o "Training (incl. cache creation) took [0-9.a-zµ]*" "$LOG" | tail -1 | awk '{print $NF}' || true)
CACHE_S=$(grep -o "Dataset cache created in [0-9.a-zµ]*" "$LOG" | tail -1 | awk '{print $NF}' || true)
ACC=$(grep -m1 -E "^Accuracy:" "$LOG" | awk '{print $2}' || true)
AUC=$(grep -m1 -E "^\s*auc:" "$LOG" | awk '{print $2}' || true)
# Tree-training seconds = "Start tree training" -> "Training ... took" log timestamps (excludes
# dataset-cache creation and dataspec inference; includes the workers loading their columns).
TREES_S=$(python3 - "$LOG" <<'PY' || true
import re,sys
ts=[]
for pat in (r"Start tree training", r"Training \(incl\. cache creation\) took"):
    for line in open(sys.argv[1], errors="replace"):
        m=re.match(r"I\d{4} (\d\d):(\d\d):(\d\d\.\d+)", line)
        if m and re.search(pat, line):
            ts.append(int(m.group(1))*3600+int(m.group(2))*60+float(m.group(3)))
if len(ts)==2:
    d=ts[1]-ts[0]
    if d<0: d+=86400
    print(round(d,2))
PY
)
LOGLOSS=$(grep -m1 -E "^LogLoss:" "$LOG" | sed -E 's/LogLoss: *:? *//' | awk '{print $1}' || true)
COND=$(grep -o "Conditions: .*" "$LOG" | tail -1 || true)
[ -f "$RESULTS" ] || echo "dataset,workers,threads_per_worker,split_axis,trees,max_depth,rc,train_took,tree_training_s,cache_created,wall_s,test_acc,test_auc,test_logloss,conditions,extra,git,log" > "$RESULTS"
echo "$DATASET,$W,$THREADS,$AXIS,$TREES,$MAX_DEPTH,$RC,$TRAIN_S,$TREES_S,$CACHE_S,$WALL,$ACC,$AUC,$LOGLOSS,\"$COND\",\"$*\",$GITSHA,$(basename "$LOG")" >> "$RESULTS"
echo "=== rc=$RC train_took=$TRAIN_S trees=${TREES_S}s cache=$CACHE_S wall=${WALL}s acc=$ACC auc=$AUC $COND" | tee -a "$LOG"
exit $RC
