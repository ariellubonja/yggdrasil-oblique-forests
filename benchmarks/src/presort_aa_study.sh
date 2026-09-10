#!/usr/bin/env bash
set -euo pipefail

# Presort benefit for axis-aligned Random Forests on the real datasets
# (HIGGS, SUSY, EPSILON).
#
# Sweeps --aa_sorting_strategy for AA splits under two numerical split types:
#   * "Exact"                    -> the CART/exact finder, the only path that
#                                   consults the presorted feature index
#                                   (training.cc FindSplitLabelClassification-
#                                   FeatureNumericalCart / EffectiveStrategy).
#   * "Dynamic Random Histogram" -> the harness default (64 bins). For AA there
#                                   is no dynamic downgrade (that lives in
#                                   oblique.cc), so this is a plain random
#                                   histogram and presort can only cost: the
#                                   index is built inside the training block
#                                   (random_forest.cc PreprocessTrainingDataset
#                                   runs after begin_training) but never read.
# Strategies: in_node (no index) vs presorted (index built; used at nodes with
# >= 12.5 % of the examples and >= 25 rows, in-node sort below -- YDF's AUTO).
# Under the histogram the index is built but never consulted, so that pair
# measures the pure presort cost.
#
# Everything else is the harness default (train_oblique_forest.cc: Bagging,
# growing_strategy Local, min_examples 1, num_candidate_attributes 0 = sqrt(F),
# all threads) and runtime.sh's tree count (5 x P-cores).
#
# Usage:  $0 [--runs=N] [--smoke] <suffix>
#   --runs=N   repetitions per cell (default 3; median reported)
#   --smoke    one tiny trunk (20000 x 16), 1 run per cell -- checks plumbing
# Environment (recorded in the provenance header):
#   EXTRA_BAZEL_CONFIGS, EXTRA_TRAIN_ARGS  as in runtime.sh
#   CSV_DATASETS_OVERRIDE                  space-separated "path|label" subset
#   SPLIT_TYPES_OVERRIDE                   '|'-separated, e.g. "Exact"
#   STRATEGIES_OVERRIDE                    e.g. "in_node presorted force_presorted"
#
# Output: benchmarks/results/misc/axis_aligned_presort/presort_aa_<suffix>.csv, one
# row per (dataset, split_type, strategy), appended as cells finish. presort_s
# is the median "Feature index computed in" time (inside the training block).

NUM_RUNS=3
MODE="full"
SUFFIX=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs=*)
      NUM_RUNS="${1#*=}"
      [[ "$NUM_RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --runs must be a positive integer" >&2; exit 2; }
      shift ;;
    --smoke) MODE="smoke"; NUM_RUNS=1; shift ;;
    -h|--help) echo "Usage: $0 [--runs=N] [--smoke] <suffix>" >&2; exit 0 ;;
    --*) echo "ERROR: unknown flag '$1'" >&2; exit 2 ;;
    *)
      [[ -n "$SUFFIX" ]] && { echo "ERROR: unexpected positional '$1'" >&2; exit 2; }
      SUFFIX="${1,,}"; shift ;;
  esac
done
[[ -z "$SUFFIX" ]] && { echo "Usage: $0 [--runs=N] [--smoke] <suffix>" >&2; exit 2; }

###### Parameters

CSV_DATASETS=(
  "benchmarks/data/HIGGS_with_header.csv|class"
  "benchmarks/data/SUSY_with_header.csv|class"
  "benchmarks/data/epsilon_normalized_train.csv|label"
)
if [[ -n "${CSV_DATASETS_OVERRIDE:-}" ]]; then
  read -r -a CSV_DATASETS <<<"$CSV_DATASETS_OVERRIDE"
fi
SPLIT_TYPES=("Exact" "Dynamic Random Histogram")
if [[ -n "${SPLIT_TYPES_OVERRIDE:-}" ]]; then
  IFS='|' read -r -a SPLIT_TYPES <<<"$SPLIT_TYPES_OVERRIDE"
fi
# force_presorted is excluded from the real-dataset sweep: the presorted scan
# (splitter_scanner.h ScanSplitsPresortedSparse) rebuilds a mask over ALL N
# examples and walks the whole sorted column at every node, i.e. O(N) per
# (node, feature) regardless of node size. On purity-depth trees with millions
# of nodes and N = 11M that never finishes (7x slower already on the 20k-row
# smoke trunk). 'presorted' = YDF AUTO: index used only where the node holds
# >= 12.5 % of the examples (top ~3 levels), in-node sort elsewhere.
strategies_for() {
  echo "${STRATEGIES_OVERRIDE:-in_node presorted}"
}

BUILD_TARGET="//examples:train_oblique_forest"
BAZEL_FLAGS=(-c opt --cxxopt="-O3" --cxxopt="-march=native")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/utils/bench_common.sh"
bench_restore_ecores_on_exit

logdir="benchmarks/results/misc/axis_aligned_presort"
mkdir -p "$logdir"
logfile="${logdir}/presort_aa_${SUFFIX}.log"
csvfile="${logdir}/presort_aa_${SUFFIX}.csv"
BENCH_LOGFILE="$logfile"
confirm_overwrite "$logfile" "$csvfile"

bazel_build "${BAZEL_FLAGS[@]}" "$BUILD_TARGET"
BINARY="./bazel-bin/examples/train_oblique_forest"

NUM_TREES="$(bench_num_trees)"
BASE_ARGS="--num_trees=$NUM_TREES"

bench_provenance_to_csv "$csvfile" \
  "dataset,split_type,strategy,num_trees,median_s,stddev_s,n_samples,all_samples_s,presort_s,pre_train_s" \
  "NUM_TREES: $NUM_TREES  NUM_RUNS: $NUM_RUNS  MODE: $MODE" \
  "CSV_DATASETS: ${CSV_DATASETS[*]}" \
  "SPLIT_TYPES: ${SPLIT_TYPES[*]}"

# Like bench_repeat_cmd, plus the presort and pre-train wall times per run.
run_cell() {
  local dataset_label=$1 input_args=$2 split=$3 strategy=$4
  local cmd="$BINARY $input_args --feature_split_type \"Axis Aligned\" \
    --numerical_split_type \"$split\" --aa_sorting_strategy=$strategy \
    $BASE_ARGS ${EXTRA_TRAIN_ARGS:-}"
  bench_log "=== dataset=$dataset_label split=\"$split\" strategy=$strategy ==="
  bench_log "$cmd"
  local times=() presorts=() pretrains=()
  local i out rc t p q
  for ((i=1; i<=NUM_RUNS; i++)); do
    bench_log "----- Run $i/$NUM_RUNS -----"
    rc=0
    out=$(bash -c "$cmd" 2>&1) || rc=$?
    bench_log "$out"
    (( rc != 0 )) && bench_log "WARNING: command exited with status $rc on run $i"
    t=$(echo "$out" | grep -oE 'Training block took:[[:space:]]*[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?' \
        | grep -oE '[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?' | tail -1) || true
    p=$(echo "$out" | grep -oE 'Feature index computed in [0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?' \
        | grep -oE '[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$' | tail -1) || true
    q=$(echo "$out" | grep -oE 'Loading/Init \(pre-train\):[[:space:]]*[0-9]+(\.[0-9]+)?' \
        | grep -oE '[0-9]+(\.[0-9]+)?$' | tail -1) || true
    if [[ -n "$t" ]]; then times+=("$t"); else bench_log "WARNING: no 'Training block took' in run $i"; fi
    [[ -n "$p" ]] && presorts+=("$p")
    [[ -n "$q" ]] && pretrains+=("$q")
    (( rc == 137 )) && { bench_log "WARNING: OOM-killed; skipping remaining runs of this cell"; break; }
  done
  local median stddev pmed pstd qmed qstd n=${#times[@]} samples=""
  bench_median_stddev times median stddev
  bench_median_stddev presorts pmed pstd
  bench_median_stddev pretrains qmed qstd
  (( n > 0 )) && samples=$(IFS=';'; echo "${times[*]}")
  bench_log "RESULT dataset=$dataset_label split=\"$split\" strategy=$strategy median=${median}s stddev=${stddev}s presort=${pmed}s n=$n"
  echo "${dataset_label},${split// /_},${strategy},${NUM_TREES},${median},${stddev},${n},${samples},${pmed},${qmed}" >> "$csvfile"
}

banner "AA PRESORT STUDY  mode: $MODE  configs: ${EXTRA_BAZEL_CONFIGS:-<none>}  args: ${EXTRA_TRAIN_ARGS:-<none>}"

if [[ "$MODE" == "smoke" ]]; then
  for split in "${SPLIT_TYPES[@]}"; do
    for strategy in $(strategies_for "$split"); do
      run_cell "trunk_20000_x_16" "--input_mode trunk --rows 20000 --cols 16" "$split" "$strategy"
    done
  done
else
  for entry in "${CSV_DATASETS[@]}"; do
    IFS='|' read -r path label <<<"$entry"
    dataset_label=$(basename "$path" .csv)
    for split in "${SPLIT_TYPES[@]}"; do
      for strategy in $(strategies_for "$split"); do
        run_cell "$dataset_label" "--input_mode csv --train_csv \"$path\" --label_col \"$label\"" "$split" "$strategy"
      done
    done
  done
fi

echo
echo "Done. CSV: $csvfile"
echo "Log: $logfile (kept)"
