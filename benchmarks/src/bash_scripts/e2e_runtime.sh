#!/usr/bin/env bash
set -euo pipefail

###### Parameters

NUM_TREES=240 # Good number for 48-core AWS machine to prevent skewness
NUM_THREADS=-1
COMPUTE_OOB_PERFORMANCES=false  # set true to compute OOB metrics
# Ariel - ENSURE compute_oob_performances===== - it has an equal sign, not a blank space
BASE_ARGS="--num_trees=$NUM_TREES --num_threads=$NUM_THREADS --compute_oob_performances=$COMPUTE_OOB_PERFORMANCES"

# histogram_num_bins=64
histogram_num_bins=256   # Uncomment to switch; AVX512 will be used on Vectorized method

# Which feature split types to run (comment out any you don't want)
SPLIT_TYPES=(
  "Oblique"
  "Axis Aligned"
)

# Numerical split methods (comment out any you don't want)
METHODS=(
  "Exact"
  "Random"
  "Dynamic Random Histogram"
# "Equal Width"
# "Dynamic Equal Width Histogram"
)

# Dynamic split threshold (only affects Dynamic methods)
DYNAMIC_SPLIT_THRESHOLDS=(
  100
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

# Optional per-method extra args
declare -A METHOD_EXTRA_ARGS
METHOD_EXTRA_ARGS["Exact"]=""
METHOD_EXTRA_ARGS["Equal Width"]="--histogram_num_bins=$histogram_num_bins"
METHOD_EXTRA_ARGS["Random"]="--histogram_num_bins=$histogram_num_bins"
METHOD_EXTRA_ARGS["Dynamic Equal Width Histogram"]="--histogram_num_bins=$histogram_num_bins"
METHOD_EXTRA_ARGS["Dynamic Random Histogram"]="--histogram_num_bins=$histogram_num_bins"

# Build target and base flags
BUILD_TARGET="//examples:train_oblique_forest"
BAZEL_FLAGS=(-c opt --cxxopt="-O3" --cxxopt="-march=native")
# Vectorized build configs (adjust if your repo uses different config names)
VEC_CONFIG_AVX2="--config=enable_std_upper_bound_avx2"
VEC_CONFIG_AVX512="--config=enable_std_upper_bound_avx512"

# Vectorization applies only to these methods (Oblique only)
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

is_dynamic_method() {
  [[ "$1" == Dynamic* ]]
}

# -------------------------
# Normal experiments (Oblique and/or Axis Aligned per SPLIT_TYPES)
# -------------------------
banner "NORMAL EXPERIMENTS (no explicit vector ISA) histogram_num_bins=${histogram_num_bins}"

oblique_selected=false
for split in "${SPLIT_TYPES[@]}"; do
  # Select compatible methods for this split type
  methods_to_run=()
  if [[ "$split" == "Axis Aligned" ]]; then
    allowed=("Exact" "Random" "Equal Width")
    for m in "${METHODS[@]}"; do
      for a in "${allowed[@]}"; do
        if [[ "$m" == "$a" ]]; then
          methods_to_run+=("$m")
          break
        fi
      done
    done
    if [[ "${#methods_to_run[@]}" -eq 0 ]]; then
      banner "AXIS ALIGNED: No compatible methods selected (need Exact, Random, or Equal Width). Skipping."
      continue
    fi
    banner "AXIS ALIGNED EXPERIMENTS feature_split_type=Axis Aligned histogram_num_bins=${histogram_num_bins}"
    feature_arg='--feature_split_type "Axis Aligned"'
  else
    # Oblique: use METHODS as-is, no feature_split_type flag
    methods_to_run=("${METHODS[@]}")
    oblique_selected=true
    banner "OBLIQUE EXPERIMENTS histogram_num_bins=${histogram_num_bins}"
    feature_arg='--feature_split_type "Oblique"'
  fi

  for method in "${methods_to_run[@]}"; do
    extra="${METHOD_EXTRA_ARGS[$method]:-}"

    # Build list of threshold values to iterate over
    if is_dynamic_method "$method"; then
      thresholds=("${DYNAMIC_SPLIT_THRESHOLDS[@]}")
    else
      thresholds=("")  # single empty entry so the loop runs once
    fi

    for thresh in "${thresholds[@]}"; do
      thresh_arg=""
      thresh_label=""
      if [[ -n "$thresh" ]]; then
        thresh_arg="--dynamic_split_threshold=$thresh"
        thresh_label=" threshold=$thresh"
      fi

      if [[ -n "$extra" ]]; then
        banner "Running $method [$split] with $histogram_num_bins bins${thresh_label}"
      else
        banner "Running $method [$split]${thresh_label}"
      fi

      # CSV datasets
      for entry in "${CSV_DATASETS[@]}"; do
        IFS='|' read -r path label <<<"$entry"
        cmd="$BINARY --input_mode csv --train_csv \"$path\" --label_col \"$label\" $feature_arg --numerical_split_type \"$method\" $BASE_ARGS $extra $thresh_arg"
        run_cmd "$cmd"
      done

      # Trunk rows
      for rows in "${TRUNK_ROWS[@]}"; do
        cmd="$BINARY --input_mode trunk --rows $rows $feature_arg --numerical_split_type \"$method\" $BASE_ARGS $extra $thresh_arg"
        run_cmd "$cmd"
      done
    done
  done
done

# -------------------------
# Vectorized experiments (Oblique only; Random, Dynamic Random Histogram)
# -------------------------

# Only run if Oblique was selected and vectorizable methods are present
selected_vec_methods=()
if [[ "$oblique_selected" == "true" ]]; then
  for m in "${METHODS[@]}"; do
    for v in "${VECTORIZE_METHODS[@]}"; do
      if [[ "$m" == "$v" ]]; then
        selected_vec_methods+=("$m")
        break
      fi
    done
  done
fi

if [[ "${#selected_vec_methods[@]}" -eq 0 ]]; then
  banner "No vectorizable Oblique methods selected; skipping vectorized experiments"
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

banner "VECTORIZED EXPERIMENTS [${vec_name}] (Oblique only) histogram_num_bins=${histogram_num_bins}"
echo "USING INSTRUCTION SET: ${vec_name}" | tee -a "$logfile"

for method in "${selected_vec_methods[@]}"; do
  extra="${METHOD_EXTRA_ARGS[$method]:-}"

  # Build list of threshold values to iterate over
  if is_dynamic_method "$method"; then
    thresholds=("${DYNAMIC_SPLIT_THRESHOLDS[@]}")
  else
    thresholds=("")  # single empty entry so the loop runs once
  fi

  for thresh in "${thresholds[@]}"; do
    thresh_arg=""
    thresh_label=""
    if [[ -n "$thresh" ]]; then
      thresh_arg="--dynamic_split_threshold=$thresh"
      thresh_label=" threshold=$thresh"
    fi

    banner "Running $method [VECTORIZE: ${vec_name}] with $histogram_num_bins bins${thresh_label}"

    # CSV datasets
    for entry in "${CSV_DATASETS[@]}"; do
      IFS='|' read -r path label <<<"$entry"
      cmd="$BINARY --input_mode csv --train_csv \"$path\" --label_col \"$label\" --numerical_split_type \"$method\" $BASE_ARGS $extra $thresh_arg"
      run_cmd "$cmd"
    done

    # Trunk rows
    for rows in "${TRUNK_ROWS[@]}"; do
      cmd="$BINARY --input_mode trunk --rows $rows --numerical_split_type \"$method\" $BASE_ARGS $extra $thresh_arg"
      run_cmd "$cmd"
    done
  done
done

# CPU features re-enabled by trap on exit