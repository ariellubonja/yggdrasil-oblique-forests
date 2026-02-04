#!/usr/bin/env bash
set -euo pipefail

# =========================
# Configuration (edit/comment here)
# =========================

NUM_TREES=100
NUM_THREADS=-1
BASE_ARGS="--num_trees $NUM_TREES --num_threads $NUM_THREADS"

histogram_num_bins=64
# histogram_num_bins=256   # Uncomment to switch; AVX512 will be used on Vectorized method

# Numerical split method - comment out any you don't want
METHODS=(
  "Exact"
  "Random"
  "Dynamic Random Histogram"
# "Equal Width"
# "Dynamic Equal Width Histogram"
)

# Optional per-method extra args
declare -A METHOD_EXTRA_ARGS
METHOD_EXTRA_ARGS["Exact"]=""
METHOD_EXTRA_ARGS["Equal Width"]="--histogram_num_bins $histogram_num_bins"
METHOD_EXTRA_ARGS["Random"]="--histogram_num_bins $histogram_num_bins"
METHOD_EXTRA_ARGS["Dynamic Equal Width Histogram"]="--histogram_num_bins $histogram_num_bins"
METHOD_EXTRA_ARGS["Dynamic Random Histogram"]="--histogram_num_bins $histogram_num_bins"

# CSV datasets as "path|label_col" (comment out lines to skip)
CSV_DATASETS=(
  # Big
  "benchmarks/data/HIGGS_with_header.csv|class"
  "benchmarks/data/SUSY_with_header.csv|class"
  "benchmarks/data/epsilon_normalized_train.csv|label"

  # Small
  "benchmarks/data/cc18_binary_csv/task_14965_bank-marketing/repeat0_fold0_sample0_train.csv|Class"
  "benchmarks/data/cc18_binary_csv/task_14952_PhishingWebsites/repeat0_fold0_sample0_train.csv|Result"
  "benchmarks/data/cc18_binary_csv/task_29_credit-approval/repeat0_fold0_sample0_train.csv|class"
  "benchmarks/data/cc18_binary_csv/task_167125_Internet-Advertisements/repeat0_fold0_sample0_train.csv|class"
)

# Synthetic trunk rows (comment out values to skip)
TRUNK_ROWS=(
  10000
  100000
  1000000
)

# =========================
# Main Script
# =========================

# Build target and base flags
BUILD_TARGET="//examples:train_oblique_forest"
BAZEL_FLAGS=(-c opt --cxxopt="-O3" --cxxopt="-march=native")
# Vectorized build configs (adjust if your repo uses different config names)
VEC_CONFIG_AVX2="--config=enable_std_upper_bound_avx2"
VEC_CONFIG_AVX512="--config=enable_std_upper_bound_avx512"

# Vectorization applies only to these methods
VECTORIZE_METHODS=("Random" "Dynamic Random Histogram")

# Always: enable -> build -> disable -> run -> re-enable at end
sudo benchmarks/src/utils/set_cpu_e_features.sh --enable
trap 'sudo benchmarks/src/utils/set_cpu_e_features.sh --enable' EXIT

logdir="benchmarks/results"
mkdir -p "$logdir"
ts=$(date +%Y%m%d_%H%M%S)
logfile="${logdir}/e2e_runtime_${ts}.log"

# Normal build
bazel build "${BAZEL_FLAGS[@]}" "$BUILD_TARGET"
BINARY="./bazel-bin/examples/train_oblique_forest"

# Disable for experiments
sudo benchmarks/src/utils/set_cpu_e_features.sh --disable

run_cmd() {
  echo "$*" | tee -a "$logfile"
  bash -c "$*" 2>&1 | tee -a "$logfile"
}

banner() {
  echo -e "\n\n======== $* ========\n" | tee -a "$logfile"
}

# -------------------------
# Normal (non-vectorized) experiments
# -------------------------
banner "NORMAL EXPERIMENTS (no explicit vector ISA) histogram_num_bins=${histogram_num_bins}"

for method in "${METHODS[@]}"; do
  extra="${METHOD_EXTRA_ARGS[$method]:-}"
  if [[ -n "$extra" ]]; then
    banner "Running $method with $histogram_num_bins bins"
  else
    banner "Running $method"
  fi

  # CSV datasets
  for entry in "${CSV_DATASETS[@]}"; do
    IFS='|' read -r path label <<<"$entry"
    cmd="$BINARY --input_mode csv --train_csv \"$path\" --label_col \"$label\" --numerical_split_type \"$method\" $BASE_ARGS $extra"
    run_cmd "$cmd"
  done

  # Trunk rows
  for rows in "${TRUNK_ROWS[@]}"; do
    cmd="$BINARY --input_mode trunk --rows $rows --numerical_split_type \"$method\" $BASE_ARGS $extra"
    run_cmd "$cmd"
  done
done

# -------------------------
# Vectorized experiments (Random, Dynamic Random Histogram only)
# -------------------------

# Determine which vectorizable methods are selected by the user
selected_vec_methods=()
for m in "${METHODS[@]}"; do
  for v in "${VECTORIZE_METHODS[@]}"; do
    if [[ "$m" == "$v" ]]; then
      selected_vec_methods+=("$m")
      break
    fi
  done
done

if [[ "${#selected_vec_methods[@]}" -eq 0 ]]; then
  banner "No vectorizable methods selected; skipping vectorized experiments"
  exit 0
fi

# Determine ISA based on histogram_num_bins
vec_cfg=""
vec_name=""
if [[ "$histogram_num_bins" -eq 64 ]]; then
  vec_cfg="$VEC_CONFIG_AVX2"
  vec_name="AVX2"
elif [[ "$histogram_num_bins" -eq 256 ]]; then
  vec_cfg="$VEC_CONFIG_AVX512"
  vec_name="AVX512"
else
  banner "Vectorized experiments require histogram_num_bins to be 64 (AVX2) or 256 (AVX512). Current: $histogram_num_bins. Skipping vectorized experiments."
  exit 0
fi

bazel build "${BAZEL_FLAGS[@]}" "$vec_cfg" "$BUILD_TARGET"

banner "VECTORIZED EXPERIMENTS [$vec_name] histogram_num_bins=${histogram_num_bins}"

for method in "${selected_vec_methods[@]}"; do
  extra="${METHOD_EXTRA_ARGS[$method]:-}"
  banner "Running $method [VECTORIZE: $vec_name] with $histogram_num_bins bins"

  # CSV datasets
  for entry in "${CSV_DATASETS[@]}"; do
    IFS='|' read -r path label <<<"$entry"
    cmd="$BINARY --input_mode csv --train_csv \"$path\" --label_col \"$label\" --numerical_split_type \"$method\" $BASE_ARGS $extra"
    run_cmd "$cmd"
  done

  # Trunk rows
  for rows in "${TRUNK_ROWS[@]}"; do
    cmd="$BINARY --input_mode trunk --rows $rows --numerical_split_type \"$method\" $BASE_ARGS $extra"
    run_cmd "$cmd"
  done
done

# CPU features re-enabled by trap on exit