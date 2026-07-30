#!/usr/bin/env bash
set -euo pipefail

# Sweep Dynamic Random Histogram thresholds. The dynamic splitter uses Exact
# below the threshold and Random Histogram above it; this script selects which
# Random implementation is benchmarked.
#
# Usage:  $0 [--random|--random-avx2|--random-avx512|--all] <suffix>
#   --random        scalar Random Histogram, 64 bins
#   --random-avx2   AVX2 Random Histogram, 64 bins
#   --random-avx512 AVX512 Random Histogram, 256 bins
#   --all           run all three Random implementations
#
# Honors EXTRA_BAZEL_CONFIGS / EXTRA_TRAIN_ARGS like runtime.sh, and writes the
# same PROVENANCE header (to the log and the top of the CSV).

RANDOM_IMPLS=()
SUFFIX=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --random) RANDOM_IMPLS+=("random"); shift ;;
    --random-avx2) RANDOM_IMPLS+=("random-avx2"); shift ;;
    --random-avx512) RANDOM_IMPLS+=("random-avx512"); shift ;;
    --all) RANDOM_IMPLS=("random" "random-avx2" "random-avx512"); shift ;;
    -h|--help)
      echo "Usage: $0 [--random|--random-avx2|--random-avx512|--all] <suffix>" >&2
      exit 0 ;;
    --*)
      echo "ERROR: unknown flag '$1'" >&2
      echo "Usage: $0 [--random|--random-avx2|--random-avx512|--all] <suffix>" >&2
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
  echo "Usage: $0 [--random|--random-avx2|--random-avx512|--all] <suffix>" >&2
  exit 2
fi
if [[ "${#RANDOM_IMPLS[@]}" -eq 0 ]]; then
  RANDOM_IMPLS=("random")
fi

###### Parameters

NUM_RUNS=3
NUM_THREADS=-1
COMPUTE_OOB_PERFORMANCES=false

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

# Synthetic trunk datasets.
TRUNK_DATASETS=(
  "50000000|4"
  "3000000|4096"
  "3000|4000000"
)

BUILD_TARGET="//examples:train_oblique_forest"
BAZEL_FLAGS=(-c opt --cxxopt="-O3" --cxxopt="-march=native" --repo_env=CC=icx --repo_env=CXX=icpx)
# SIMD histogram binning is default-ON; the AVX2 (64 bins) / AVX-512 (256 bins)
# path is chosen at RUNTIME from the bin count, so the avx2/avx512 impls below
# use the same default build and differ only by histogram_num_bins.

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

bench_require_absent "$logfile" "$csvfile"

dataset_name_for_csv() {
  local path="$1"
  local base
  base="$(basename "$path")"
  echo "${base%.csv}"
}

run_cmd() {
  local dataset="$1"
  local random_impl="$2"
  local bins="$3"
  local threshold="$4"
  local cmd="$5"

  bench_log "$cmd"
  local times=()
  bench_repeat_cmd "$cmd" times

  local median stddev samples
  bench_median_stddev times median stddev
  samples=$(IFS=';'; echo "${times[*]}")
  bench_log "RESULT dataset=$dataset random_impl=$random_impl bins=$bins threshold=$threshold median=${median}s stddev=${stddev}s n=${#times[@]}"
  {
    csv_escape "$dataset"
    printf ','
    csv_escape "$random_impl"
    printf ',%s,%s,%s,%s,%s,%s,' "$bins" "$threshold" "$NUM_TREES" "$median" "$stddev" "${#times[@]}"
    csv_escape "$samples"
    printf '\n'
  } >> "$csvfile"
}

# E-cores off before nproc so NUM_TREES reflects the runtime (P-core) topology;
# bazel_build re-enables them for the build itself only.
bench_ecores --disable
if [[ "$EXTRA_TRAIN_ARGS" == *"ensemble_method=Boosting"* ]]; then
  NUM_TREES=300   # boosting is sequential; the 5x-cores heuristic doesn't apply
else
  NUM_TREES=$(( $(bench_nproc) * 5 ))
fi
BASE_ARGS="--num_trees=$NUM_TREES --num_threads=$NUM_THREADS --compute_oob_performances=$COMPUTE_OOB_PERFORMANCES"

# Provenance, then the header. Rows are appended as the sweep runs, so both must
# land before the first run_cmd; the leading lines are intentionally not CSV.
bench_provenance_block \
  "NUM_TREES: $NUM_TREES  NUM_RUNS: $NUM_RUNS  NUM_THREADS: $NUM_THREADS" \
  "RANDOM_IMPLS: ${RANDOM_IMPLS[*]}" \
  "DYNAMIC_SPLIT_THRESHOLDS: ${DYNAMIC_SPLIT_THRESHOLDS[*]}" \
  | tee -a "$logfile" > "$csvfile"
echo "dataset,random_impl,histogram_num_bins,dynamic_split_threshold,num_trees,median_s,stddev_s,n_samples,all_samples_s" >> "$csvfile"

for random_impl in "${RANDOM_IMPLS[@]}"; do
  case "$random_impl" in
    random)
      histogram_num_bins=64
      build_args=("${BAZEL_FLAGS[@]}")
      impl_label="Random"
      ;;
    random-avx2)
      histogram_num_bins=64
      build_args=("${BAZEL_FLAGS[@]}")   # AVX2 selected at runtime from bins=64
      impl_label="Random AVX2"
      ;;
    random-avx512)
      histogram_num_bins=256
      build_args=("${BAZEL_FLAGS[@]}")   # AVX-512 selected at runtime from bins=256
      impl_label="Random AVX512"
      ;;
    *)
      echo "ERROR: unknown random implementation '$random_impl'" >&2
      exit 2
      ;;
  esac

  bazel_build "${build_args[@]}" "$BUILD_TARGET"
  BINARY="./bazel-bin/examples/train_oblique_forest"

  banner "DYNAMIC THRESHOLD SWEEP [Exact below threshold, ${impl_label} above] histogram_num_bins=${histogram_num_bins}"

  for threshold in "${DYNAMIC_SPLIT_THRESHOLDS[@]}"; do
    threshold_arg="--dynamic_split_threshold=$threshold"
    extra="--histogram_num_bins=$histogram_num_bins"

    banner "Threshold ${threshold} [${impl_label}]"

    for entry in "${CSV_DATASETS[@]}"; do
      IFS='|' read -r path label <<<"$entry"
      dataset="$(dataset_name_for_csv "$path")"
      cmd="$BINARY --input_mode csv --train_csv \"$path\" --label_col \"$label\" --feature_split_type \"Oblique\" --numerical_split_type \"Dynamic Random Histogram\" $BASE_ARGS $extra $threshold_arg $EXTRA_TRAIN_ARGS"
      run_cmd "$dataset" "$random_impl" "$histogram_num_bins" "$threshold" "$cmd"
    done

    for entry in "${TRUNK_DATASETS[@]}"; do
      IFS='|' read -r rows cols <<<"$entry"
      dataset="trunk_${rows}_x_${cols}"
      cmd="$BINARY --input_mode trunk --rows $rows --cols $cols --feature_split_type \"Oblique\" --numerical_split_type \"Dynamic Random Histogram\" $BASE_ARGS $extra $threshold_arg $EXTRA_TRAIN_ARGS"
      run_cmd "$dataset" "$random_impl" "$histogram_num_bins" "$threshold" "$cmd"
    done
  done
done

echo
echo "CSV: $csvfile"
echo "Log: $logfile"
