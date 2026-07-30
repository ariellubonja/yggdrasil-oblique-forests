#!/usr/bin/env bash
set -euo pipefail

# Sweep --dynamic_split_threshold. The dynamic splitter uses Exact below the
# threshold and a histogram finder above it; this script times one build across
# the threshold ladder below, on every dataset.
#
# Usage:  $0 <suffix>
#   <suffix> becomes part of the result filename, e.g. 'AWS_m7i' ->
#   dynamic_threshold_sweep_aws_m7i.csv.
#
# The script is agnostic about WHAT is swept: it builds once and varies only the
# threshold. Everything else comes from the environment:
#   EXTRA_BAZEL_CONFIGS  build configs, e.g. "--config=symmetric_optimized"
#   EXTRA_TRAIN_ARGS     binary flags, e.g. '--feature_split_type "Oblique"
#                        --numerical_split_type "Dynamic Random Histogram"
#                        --histogram_num_bins=64'
# Both are recorded in the CSV provenance header. Pass the flags that identify
# the experiment in EXTRA_TRAIN_ARGS; --num_trees and --dynamic_split_threshold
# are the two the script owns.
#
# Note: the AVX2 (64 bins) / AVX-512 (256 bins) / scalar histogram binner is
# picked at RUNTIME from cpuid + the bin count, so the ISA follows
# --histogram_num_bins in EXTRA_TRAIN_ARGS, never a build config.

SUFFIX=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      echo "Usage: $0 <suffix>" >&2
      exit 0 ;;
    --*)
      echo "ERROR: unknown flag '$1'" >&2
      echo "Usage: $0 <suffix>" >&2
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
  echo "Usage: $0 <suffix>" >&2
  echo "  e.g. '$0 AWS_m7i' -> dynamic_threshold_sweep_aws_m7i.csv" >&2
  exit 2
fi

###### Parameters

NUM_RUNS=3

DYNAMIC_SPLIT_THRESHOLDS=(
  # 100
  350
  600
  850
  1100
  1350
  1600
  1850
  2100
  2350
  2600
  2850
  3100
)

# Optional CSV datasets as "path|label_col" entries.
CSV_DATASETS=(
  # "benchmarks/data/HIGGS_with_header.csv|class"
  # "benchmarks/data/SUSY_with_header.csv|class"
  # "benchmarks/data/epsilon_normalized_train.csv|label"
)

# Synthetic trunk datasets as "rows|cols" pairs.
TRUNK_DATASETS=(
  "50000000|4"
  "3000000|4096"
  "3000|4000000"
)

# =========================
# Main Script
# =========================

BUILD_TARGET="//examples:train_oblique_forest"
BAZEL_FLAGS=(-c opt --cxxopt="-O3" --cxxopt="-march=native")

# Shared plumbing: ensure_icx, bazel_build (+EXTRA_BAZEL_CONFIGS), e-core
# toggling, EXTRA_TRAIN_ARGS, provenance, banner/csv_escape, timing helpers.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/bench_common.sh"
bench_restore_ecores_on_exit

logdir="benchmarks/results"
mkdir -p "$logdir"
logfile="${logdir}/dynamic_threshold_sweep_${SUFFIX}.log"
csvfile="${logdir}/dynamic_threshold_sweep_${SUFFIX}.csv"
BENCH_LOGFILE="$logfile"

confirm_overwrite "$logfile"
confirm_overwrite "$csvfile"

dataset_name_for_csv() {
  local path="$1"
  local base
  base="$(basename "$path")"
  echo "${base%.csv}"
}

run_cmd() {
  local dataset="$1"
  local threshold="$2"
  local cmd="$3"

  bench_log "$cmd"
  local times=()
  bench_repeat_cmd "$cmd" times

  local median stddev samples
  bench_median_stddev times median stddev
  samples=$(IFS=';'; echo "${times[*]}")
  bench_log "RESULT dataset=$dataset threshold=$threshold median=${median}s stddev=${stddev}s n=${#times[@]}"
  {
    csv_escape "$dataset"
    printf ',%s,%s,%s,%s,%s,' "$threshold" "$NUM_TREES" "$median" "$stddev" "${#times[@]}"
    csv_escape "$samples"
    printf '\n'
  } >> "$csvfile"
}

# bazel_build appends EXTRA_BAZEL_CONFIGS and leaves e-cores disabled, so the
# nproc behind NUM_TREES reflects the runtime (P-core) topology.
bazel_build "${BAZEL_FLAGS[@]}" "$BUILD_TARGET"
BINARY="./bazel-bin/examples/train_oblique_forest"

# 5x cores to prevent skewness, or a fixed count under Boosting.
NUM_TREES="$(bench_num_trees)"
BASE_ARGS="--num_trees=$NUM_TREES"

# Provenance, then the header. Rows are appended as the sweep runs, so both must
# land before the first run_cmd; the leading lines are intentionally not CSV.
bench_provenance_block \
  "NUM_TREES: $NUM_TREES  NUM_RUNS: $NUM_RUNS" \
  "DYNAMIC_SPLIT_THRESHOLDS: ${DYNAMIC_SPLIT_THRESHOLDS[*]}" \
  | tee -a "$logfile" > "$csvfile"
echo "dataset,dynamic_split_threshold,num_trees,median_s,stddev_s,n_samples,all_samples_s" >> "$csvfile"

banner "DYNAMIC THRESHOLD SWEEP  configs: ${EXTRA_BAZEL_CONFIGS:-<none>}  args: ${EXTRA_TRAIN_ARGS:-<none>}"

for threshold in "${DYNAMIC_SPLIT_THRESHOLDS[@]}"; do
  banner "Threshold ${threshold}"
  threshold_arg="--dynamic_split_threshold=$threshold"

  for entry in "${CSV_DATASETS[@]}"; do
    IFS='|' read -r path label <<<"$entry"
    dataset="$(dataset_name_for_csv "$path")"
    run_cmd "$dataset" "$threshold" \
      "$BINARY --input_mode csv --train_csv \"$path\" --label_col \"$label\" $BASE_ARGS $threshold_arg $EXTRA_TRAIN_ARGS"
  done

  for entry in "${TRUNK_DATASETS[@]}"; do
    IFS='|' read -r rows cols <<<"$entry"
    run_cmd "trunk_${rows}_x_${cols}" "$threshold" \
      "$BINARY --input_mode trunk --rows $rows --cols $cols $BASE_ARGS $threshold_arg $EXTRA_TRAIN_ARGS"
  done
done

echo
echo "CSV: $csvfile"
echo "Log: $logfile"
