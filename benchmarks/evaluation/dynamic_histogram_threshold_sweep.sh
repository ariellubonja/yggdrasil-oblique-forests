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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SET_CPU_E_FEATURES="$(cd "$SCRIPT_DIR/../.." && pwd)/benchmarks/utils/set_cpu_e_features.sh"
trap 'sudo "$SET_CPU_E_FEATURES" --enable' EXIT

ensure_icx() {
  # .bazelrc only pins CC=icx under build:linux; macOS has no such pin
  # (Intel never shipped icx/icpx for macOS/Apple Silicon) and just uses
  # the default toolchain, so there's nothing to ensure there.
  if [[ "$(uname -s)" != "Linux" ]]; then
    return 0
  fi
  if ! command -v icx >/dev/null 2>&1; then
    local setvars="${ONEAPI_SETVARS:-/opt/intel/oneapi/setvars.sh}"
    if [[ -r "$setvars" ]]; then
      set +u +e
      # shellcheck disable=SC1090
      source "$setvars" >/dev/null 2>&1
      set -u -e
    fi
  fi
  if ! command -v icx >/dev/null 2>&1 || ! command -v icpx >/dev/null 2>&1; then
    echo "ERROR: icx/icpx not on PATH and could not source oneAPI." >&2
    echo "       Run 'source /opt/intel/oneapi/setvars.sh' or set ONEAPI_SETVARS." >&2
    exit 1
  fi
  echo "Compiler: $(command -v icx)"
}

bazel_build() {
  ensure_icx
  sudo "$SET_CPU_E_FEATURES" --enable
  bazel build "$@"
  sudo "$SET_CPU_E_FEATURES" --disable
}

logdir="benchmarks/results"
mkdir -p "$logdir"
logfile="${logdir}/dynamic_threshold_sweep_${SUFFIX}.log"
csvfile="${logdir}/dynamic_threshold_sweep_${SUFFIX}.csv"

if [[ -e "$logfile" ]]; then
  echo "ERROR: $logfile already exists. Use a different suffix or remove it." >&2
  exit 1
fi
if [[ -e "$csvfile" ]]; then
  echo "ERROR: $csvfile already exists. Use a different suffix or remove it." >&2
  exit 1
fi

banner() {
  echo -e "\n\n======== $* ========\n" | tee -a "$logfile"
}

dataset_name_for_csv() {
  local path="$1"
  local base
  base="$(basename "$path")"
  echo "${base%.csv}"
}

median_and_stddev() {
  local -n samples_ref=$1
  local -n median_ref=$2
  local -n stddev_ref=$3
  local n=${#samples_ref[@]}
  median_ref="N/A"
  stddev_ref="N/A"
  if (( n == 0 )); then
    return
  fi
  local sorted
  mapfile -t sorted < <(printf '%s\n' "${samples_ref[@]}" | sort -g)
  local mid=$(( n / 2 ))
  if (( n % 2 == 1 )); then
    median_ref="${sorted[$mid]}"
  else
    median_ref=$(awk -v a="${sorted[$((mid-1))]}" -v b="${sorted[$mid]}" 'BEGIN{printf "%.6f", (a+b)/2.0}')
  fi
  if (( n >= 2 )); then
    stddev_ref=$(printf '%s\n' "${samples_ref[@]}" | awk '
      {x[NR]=$1; s+=$1}
      END{m=s/NR; ss=0; for(i=1;i<=NR;i++){d=x[i]-m; ss+=d*d}
          printf "%.6f", sqrt(ss/(NR-1))}')
  fi
}

csv_escape() {
  local value="$1"
  value="${value//\"/\"\"}"
  printf '"%s"' "$value"
}

run_cmd() {
  local dataset="$1"
  local random_impl="$2"
  local bins="$3"
  local threshold="$4"
  local cmd="$5"

  echo "$cmd" | tee -a "$logfile"
  local times=()
  local i out t rc
  for ((i=1; i<=NUM_RUNS; i++)); do
    echo "----- Run $i/$NUM_RUNS -----" | tee -a "$logfile"
    rc=0
    out=$(bash -c "$cmd" 2>&1) || rc=$?
    echo "$out" | tee -a "$logfile"
    if (( rc != 0 )); then
      echo "WARNING: command exited with status $rc on run $i (continuing)" | tee -a "$logfile"
    fi
    t=$(echo "$out" | grep -oE 'Training block took:[[:space:]]*[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?' \
        | grep -oE '[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?' | tail -1)
    if [[ -n "$t" ]]; then
      times+=("$t")
    else
      echo "WARNING: Could not parse 'Training block took' from run $i" | tee -a "$logfile"
    fi
  done

  local median stddev samples
  median_and_stddev times median stddev
  samples=$(IFS=';'; echo "${times[*]}")
  echo "RESULT dataset=$dataset random_impl=$random_impl bins=$bins threshold=$threshold median=${median}s stddev=${stddev}s n=${#times[@]}" | tee -a "$logfile"
  {
    csv_escape "$dataset"
    printf ','
    csv_escape "$random_impl"
    printf ',%s,%s,%s,%s,%s,%s,' "$bins" "$threshold" "$NUM_TREES" "$median" "$stddev" "${#times[@]}"
    csv_escape "$samples"
    printf '\n'
  } >> "$csvfile"
}

echo "dataset,random_impl,histogram_num_bins,dynamic_split_threshold,num_trees,median_s,stddev_s,n_samples,all_samples_s" > "$csvfile"

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

  # E-cores are now disabled, so nproc reflects the runtime CPU topology.
  NUM_TREES=$(( $(nproc) * 5 ))
  BASE_ARGS="--num_trees=$NUM_TREES --num_threads=$NUM_THREADS --compute_oob_performances=$COMPUTE_OOB_PERFORMANCES"

  banner "DYNAMIC THRESHOLD SWEEP [Exact below threshold, ${impl_label} above] histogram_num_bins=${histogram_num_bins}"

  for threshold in "${DYNAMIC_SPLIT_THRESHOLDS[@]}"; do
    threshold_arg="--dynamic_split_threshold=$threshold"
    extra="--histogram_num_bins=$histogram_num_bins"

    banner "Threshold ${threshold} [${impl_label}]"

    for entry in "${CSV_DATASETS[@]}"; do
      IFS='|' read -r path label <<<"$entry"
      dataset="$(dataset_name_for_csv "$path")"
      cmd="$BINARY --input_mode csv --train_csv \"$path\" --label_col \"$label\" --feature_split_type \"Oblique\" --numerical_split_type \"Dynamic Random Histogram\" $BASE_ARGS $extra $threshold_arg"
      run_cmd "$dataset" "$random_impl" "$histogram_num_bins" "$threshold" "$cmd"
    done

    for entry in "${TRUNK_DATASETS[@]}"; do
      IFS='|' read -r rows cols <<<"$entry"
      dataset="trunk_${rows}_x_${cols}"
      cmd="$BINARY --input_mode trunk --rows $rows --cols $cols --feature_split_type \"Oblique\" --numerical_split_type \"Dynamic Random Histogram\" $BASE_ARGS $extra $threshold_arg"
      run_cmd "$dataset" "$random_impl" "$histogram_num_bins" "$threshold" "$cmd"
    done
  done
done

echo
echo "CSV: $csvfile"
echo "Log: $logfile"
