#!/usr/bin/env bash
set -euo pipefail

# Runtime evaluation
#
# --runs controls only the number of repetitions per command (median is
# reported)
#
# Usage:  $0 [--runs=N] <suffix>     (default --runs=3)
#   <suffix> becomes part of the result filename, e.g. 'AWS_m7i' ->
#   aws_m7i.csv (the run count lives in the provenance header).
#
# The script is agnostic about WHAT is benchmarked: it builds once and runs the
# binary over the datasets below. Everything else comes from the environment:
#   EXTRA_BAZEL_CONFIGS  build configs, e.g. "--config=symmetric_optimized"
#   EXTRA_TRAIN_ARGS     binary flags, e.g. '--numerical_split_type "Dynamic
#                        Random Histogram" --histogram_num_bins=64
#                        --dynamic_split_threshold=250'
# Both are recorded in the CSV provenance header. The CSV's `algorithm` column
# is derived by the parser from the command line, so pass the flags that
# identify the experiment (feature_split_type / numerical_split_type /
# ensemble_method / dynamic_split_threshold) in EXTRA_TRAIN_ARGS.
#
# Ariel - ENSURE compute_oob_performances===== - it has an equal sign, not a blank space

NUM_RUNS=3
SUFFIX=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs=*)
      NUM_RUNS="${1#*=}"
      if ! [[ "$NUM_RUNS" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: --runs must be a positive integer, got '$NUM_RUNS'" >&2
        exit 2
      fi
      shift ;;
    -h|--help)
      echo "Usage: $0 [--runs=N] <suffix>" >&2
      exit 0 ;;
    --*)
      echo "ERROR: unknown flag '$1'" >&2
      echo "Usage: $0 [--runs=N] <suffix>" >&2
      exit 2 ;;
    *)
      if [[ -n "$SUFFIX" ]]; then
        echo "ERROR: unexpected positional argument '$1' (suffix already set to '$SUFFIX')" >&2
        exit 2
      fi
      SUFFIX="${1,,}"; shift ;;
  esac
done
if [[ -z "$SUFFIX" ]]; then
  echo "Usage: $0 [--runs=N] <suffix>" >&2
  echo "  e.g. '$0 AWS_m7i' -> aws_m7i.csv" >&2
  exit 2
fi

###### Parameters

# CSV datasets. Entries are "path|label_col".
CSV_DATASETS=(
  "benchmarks/data/HIGGS_with_header.csv|class"
  # "benchmarks/data/SUSY_with_header.csv|class"
  # "benchmarks/data/epsilon_normalized_train.csv|label"
  )
# Synthetic trunk datasets as "rows|cols" pairs.
TRUNK_DATASETS=(
  # "30000000|4" # OOMs for Symmetric trees - comment out
  # "3000000|4096"
  "1500000|4096"
  # "300000|40000"
  "150000|40000"
  #"30000|400000"
  "15000|400000"
)

# =========================
# Main Script
# =========================

BUILD_TARGET="//examples:train_oblique_forest"
BAZEL_FLAGS=(-c opt --cxxopt="-O3" --cxxopt="-march=native" --repo_env=CC=icx --repo_env=CXX=icpx)

# Always: enable -> build -> disable -> run -> re-enable at end.
# All bazel builds are wrapped by bench_common's bazel_build() so CPU E features
# are only enabled during the build itself, and disabled for every run.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/bench_common.sh"
bench_restore_ecores_on_exit

logdir="benchmarks/results"
mkdir -p "$logdir"
logfile="${logdir}/${SUFFIX}.log"
csvfile="${logdir}/${SUFFIX}.csv"
BENCH_LOGFILE="$logfile"
# Set to 1 by run_cmd when any dataset OOM'd or errored; finalize_log then keeps
# the log (instead of deleting it on success) so the failure can be inspected.
DEGRADED=0

confirm_overwrite "$logfile"
confirm_overwrite "$csvfile"

# Parse log -> CSV. On parser success the log is deleted; if the parser fails
# the log is kept for debugging. The log is also kept implicitly when the
# script aborts mid-run (set -e), since this function is never reached.
finalize_log() {
  echo "Parsing log -> CSV..."
  if python3 benchmarks/utils/parse_log_to_csv.py "$logfile" "$csvfile" runtime; then
    # Prepend the provenance to the top of the CSV. This makes the result
    # self-describing; the leading lines are intentionally not well-formed CSV.
    bench_prepend_provenance "$csvfile" "$metafile"
    if (( DEGRADED == 1 )); then
      echo "WARNING: one or more datasets reported OOM/ERROR -- KEEPING log for inspection." >&2
      echo "CSV: $csvfile  (log kept at $logfile due to OOM/ERROR rows)"
    else
      rm -f "$logfile"
      echo "CSV: $csvfile  (log deleted on success)"
    fi
  else
    echo "ERROR: parser failed; log kept at $logfile" >&2
    return 1
  fi
}

# bazel_build appends EXTRA_BAZEL_CONFIGS and leaves e-cores disabled.
bazel_build "${BAZEL_FLAGS[@]}" "$BUILD_TARGET"
BINARY="./bazel-bin/examples/train_oblique_forest"

# E-cores are now disabled, so nproc reflects the P-core count.
# 5x cores to prevent skewness, or a fixed count under Boosting.
NUM_TREES="$(bench_num_trees)"
BASE_ARGS="--num_trees=$NUM_TREES"

# Provenance: written to the log AND a temp file; the temp file is prepended to
# the top of the CSV on a successful parse (see finalize_log) and then removed.
metafile="$(mktemp)"
bench_provenance_block \
  "NUM_TREES: $NUM_TREES  NUM_RUNS: $NUM_RUNS" \
  | tee -a "$logfile" "$metafile"

run_cmd() {
  bench_log "$*"
  local times=()
  bench_repeat_cmd "$*" times
  if [[ "${#times[@]}" -gt 0 ]]; then
    local median stddev
    bench_median_stddev times median stddev
    bench_log "MEDIAN of ${#times[@]}/${NUM_RUNS} runs: ${median} s  STDDEV: ${stddev} s  (samples: ${times[*]})"
  else
    # No run produced a timing. Record OOM (process was OOM-killed) or ERROR
    # (any other failure) so this dataset surfaces as a labelled cell in the CSV
    # instead of crashing the run. DEGRADED tells finalize_log to keep the log.
    local status="ERROR"
    (( BENCH_OOM == 1 )) && status="OOM"
    bench_log "MEDIAN of 0/${NUM_RUNS} runs: ${status}"
    DEGRADED=1
  fi
}

banner "RUNTIME EXPERIMENTS  configs: ${EXTRA_BAZEL_CONFIGS:-<none>}  args: ${EXTRA_TRAIN_ARGS:-<none>}"

for entry in "${CSV_DATASETS[@]}"; do
  IFS='|' read -r path label <<<"$entry"
  run_cmd "$BINARY --input_mode csv --train_csv \"$path\" --label_col \"$label\" $BASE_ARGS $EXTRA_TRAIN_ARGS"
done

for entry in "${TRUNK_DATASETS[@]}"; do
  IFS='|' read -r rows cols <<<"$entry"
  run_cmd "$BINARY --input_mode trunk --rows $rows --cols $cols $BASE_ARGS $EXTRA_TRAIN_ARGS"
done

# CPU features re-enabled by trap on exit
finalize_log
