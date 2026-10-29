#!/usr/bin/env bash
# Naive column shipping vs disjoint-column distributed SPO-GBT on the CC18 fold-0 numeric copies
# (benchmarks/data/prep_cc18_fold0.py).
#   run_cc18_colship.sh [DATASET ...]
#   env: W=4 THREADS=2 TREES=300 MAX_DEPTH=6 MODE=inproc|grpc ARMS="disjoint ship" PORT0=8101
# Datasets outer, arms inner. Both arms use the paper's SPO-GBT settings: P = F projections per node
# in total (--sparse_oblique_num_projections_exponent 1 + --sparse_oblique_local_projection_count=true,
# so each of the W workers samples F/W), cap lifted, density 1.5, BINARY weights, exact float columns
# (--max_unique_values_for_discretized_numerical=1, non-default, same as run_dgbt_suite.sh), depth 6,
# min_examples 5 / shrinkage 0.1 (YDF defaults). "ship" adds --sparse_oblique_ship_columns.
# MODE=inproc: workers are threads of one process (MULTI_THREAD; shipping = in-memory copies).
# MODE=grpc:   W grpc_worker_main processes on 127.0.0.1:PORT0.. (loopback sockets: real
#              serialization + TCP, still not a network). Neither is a Vertex run (CLAUDE.md).
# One barrier CSV per cell (DGBT_BARRIER_CSV). The dataset cache is built once per dataset and reused
# by both arms; tree_training_s (log timestamps "Start tree training" -> "Training ... took")
# excludes it. Results: $OUT/results_<MODE>.csv; logs/barriers/work: ariel_notes/results/... (never /tmp).
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
BIN=$REPO/bazel-bin/examples/train_distributed_gbt
WBIN=$REPO/bazel-bin/yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc_worker_main
DATA=${DATA:-$REPO/benchmarks/data/cc18_fold0_numeric}
OUT=${OUT:-$REPO/benchmarks/results/runtime/distributed_spo_gbt/naive_column_shipping}
RES_ROOT=${RES_ROOT:-$REPO/ariel_notes/results/20261028_naive_colship}
LOGS=${LOGS:-$RES_ROOT/logs}
WORK=${WORK:-$RES_ROOT/work}
W=${W:-4}; THREADS=${THREADS:-2}; TREES=${TREES:-300}; MAX_DEPTH=${MAX_DEPTH:-6}
MODE=${MODE:-inproc}; ARMS=${ARMS:-"disjoint ship"}; PORT0=${PORT0:-8101}
RESULTS=${RESULTS:-$OUT/results_${MODE}.csv}
mkdir -p "$OUT" "$LOGS" "$WORK"
GITSHA=$(git -C "$REPO" rev-parse --short HEAD)
HEADER="dataset,rows_train,features,mode,workers,threads_per_worker,arm,trees,max_depth,rc,train_took,tree_training_s,wall_s,test_acc,test_auc,test_logloss,ship_bytes_in,ship_bytes_out,ship_cols_in,ship_s_max,ship_s_sum,cross_worker_splits,conditions,git,log"
[ -f "$RESULTS" ] || echo "$HEADER" > "$RESULTS"

meta() { python3 -c "import json,sys;print(json.load(open('$DATA/$1/meta.json'))['$2'])"; }
tree_seconds() {  # log -> seconds between "Start tree training" and "Training (incl. cache creation) took"
  python3 - "$1" <<'PY'
import re,sys
ts=[]
for pat in (r"Start tree training", r"Training \(incl\. cache creation\) took"):
    for line in open(sys.argv[1], errors="replace"):
        m=re.match(r"I\d{4} (\d\d):(\d\d):(\d\d\.\d+)", line)
        if m and re.search(pat, line):
            ts.append(int(m.group(1))*3600+int(m.group(2))*60+float(m.group(3))); break
if len(ts)==2:
    d=ts[1]-ts[0]
    if d<0: d+=86400
    print(round(d,2))
PY
}
field() { echo "$1" | grep -o "$2:[0-9.e+-]*" | head -1 | cut -d: -f2; }

if [ $# -gt 0 ]; then DATASETS=("$@"); else mapfile -t DATASETS < <(ls "$DATA"); fi
for ds in "${DATASETS[@]}"; do
  TRAIN=$DATA/$ds/fold0_train.csv; TEST=$DATA/$ds/fold0_test.csv
  LABEL=$(meta "$ds" label_col); ROWS=$(meta "$ds" rows_train); FEATS=$(meta "$ds" features)
  WD=$WORK/$ds
  for arm in $ARMS; do
    if grep -q "^$ds,[0-9]*,[0-9]*,$MODE,$W,$THREADS,$arm,$TREES,$MAX_DEPTH,0," "$RESULTS"; then
      echo "### skip $ds $arm (done)"; continue
    fi
    EXTRA=(); [ "$arm" = ship ] && EXTRA=(--sparse_oblique_ship_columns)
    TAG="${ds}_${MODE}_W${W}_t${THREADS}_${arm}_T${TREES}_d${MAX_DEPTH}"
    LOG=$LOGS/$TAG.log; BAR=$LOGS/${TAG}_barriers.csv
    GRPC=(); PIDS=()
    if [ "$MODE" = grpc ]; then
      ADDR=""
      for ((i = 0; i < W; i++)); do
        p=$((PORT0 + i))
        "$WBIN" --port=$p --alsologtostderr > "$LOGS/${TAG}_worker$i.log" 2>&1 &
        PIDS+=($!); ADDR="$ADDR${ADDR:+,}127.0.0.1:$p"
      done
      sleep 2; GRPC=(--grpc_worker_addresses "$ADDR")
    fi
    echo "### $(date -u +%FT%TZ) START $TAG"
    T0=$(date +%s.%N)
    DGBT_BARRIER_CSV=$BAR "$BIN" --alsologtostderr --train_csv "$TRAIN" --test_csv "$TEST" --label "$LABEL" \
      --work_dir "$WD" --num_workers "$W" --num_threads "$THREADS" \
      --num_trees "$TREES" --max_depth "$MAX_DEPTH" --split_axis SPARSE_OBLIQUE \
      --sparse_oblique_projection_density_factor 1.5 \
      --sparse_oblique_num_projections_exponent 1 \
      --sparse_oblique_max_num_projections 2147483647 \
      --sparse_oblique_local_projection_count=true \
      --max_unique_values_for_discretized_numerical=1 \
      --worker_logs=false "${EXTRA[@]}" "${GRPC[@]}" > "$LOG" 2>&1
    RC=$?
    T1=$(date +%s.%N); WALL=$(python3 -c "print(round($T1-$T0,2))")
    if [ ${#PIDS[@]} -gt 0 ]; then kill "${PIDS[@]}" 2>/dev/null; wait "${PIDS[@]}" 2>/dev/null; fi
    TRAIN_S=$(grep -o "Training (incl. cache creation) took [0-9.a-zµ]*" "$LOG" | tail -1 | awk '{print $NF}')
    TREES_S=$(tree_seconds "$LOG")
    ACC=$(grep -m1 -E "^Accuracy:" "$LOG" | awk '{print $2}')
    AUC=$(grep -m1 -E "^\s*auc:" "$LOG" | awk '{print $2}')
    LOGLOSS=$(grep -m1 -E "^LogLoss:" "$LOG" | sed -E 's/LogLoss: *:? *//' | awk '{print $1}')
    SHIP=$(grep -o "column-shipping:{[^}]*}" "$LOG" | tail -1)
    S_IN=$(field "$SHIP" bytes-in); S_OUT=$(field "$SHIP" bytes-out); S_COLS=$(field "$SHIP" cols-in)
    S_MAX=$(field "$SHIP" ship-s-max); S_SUM=$(field "$SHIP" ship-s-sum); S_X=$(field "$SHIP" cross-worker-splits)
    COND=$(grep -o "Conditions: .*" "$LOG" | tail -1)
    echo "$ds,$ROWS,$FEATS,$MODE,$W,$THREADS,$arm,$TREES,$MAX_DEPTH,$RC,$TRAIN_S,$TREES_S,$WALL,$ACC,$AUC,$LOGLOSS,${S_IN:-0},${S_OUT:-0},${S_COLS:-0},${S_MAX:-0},${S_SUM:-0},${S_X:-0},\"$COND\",$GITSHA,$(basename "$LOG")" >> "$RESULTS"
    echo "### $(date -u +%FT%TZ) END   $TAG rc=$RC trees=${TREES_S}s wall=${WALL}s acc=$ACC auc=$AUC ship_in=${S_IN:-0}B ship_s_max=${S_MAX:-0} $COND"
  done
  rm -rf "$WD"
done
echo "### SUITE_DONE $(date -u +%FT%TZ)"
