#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <suffix>" >&2
  echo "  Suffix becomes the result filename, e.g. 'AWS_m7i' -> e2e_runtime_aws_m7i.csv" >&2
  exit 2
fi
SUFFIX="${1,,}"  # lowercase

###### Parameters

NUM_RUNS=7   # Number of repetitions per command; median runtime is reported
NUM_TREES=$(( $(nproc) * 5 )) # 5x cores to prevent skewness
NUM_THREADS=-1
COMPUTE_OOB_PERFORMANCES=false  # set true to compute OOB metrics
# Ariel - ENSURE compute_oob_performances===== - it has an equal sign, not a blank space
BASE_ARGS="--num_trees=$NUM_TREES --num_threads=$NUM_THREADS --compute_oob_performances=$COMPUTE_OOB_PERFORMANCES"

histogram_num_bins=64  # NOTE! AVX512 will be used on Vectorized method

RUN_CPU=true          # set false to skip the normal (CPU) experiments section
RUN_VECTORIZED=false  # set true to run AVX2/AVX512 vectorized experiments

# GPU experiments. Only applies to Oblique + HISTOGRAM_RANDOM-style splits.
# The Oblique path requires --config=oblique_gpu (compiles in the GPU
# dispatch). Nodewise mode additionally requires --config=dfs_node_queue so
# each node is processed individually by the BFS driver (one GPU kernel per
# node). Depthwise mode uses the default BFS driver which batches sibling
# nodes into one kernel per BFS depth level.
RUN_GPU=false
GPU_MODES=(
  "depthwise"   # BFS; one kernel launch per depth level, batching siblings
  # "nodewise"    # DFS; one kernel launch per node
)

# Which feature split types to run (comment out any you don't want)
SPLIT_TYPES=(
  "Oblique"
  # "Axis Aligned"
)

# Numerical split methods (comment out any you don't want)
METHODS=(
  "Exact"
  "Random"
  # "Dynamic Random Histogram"
# "Equal Width"
# "Dynamic Equal Width Histogram"
)

# Dynamic split threshold (only affects Dynamic methods)
# Set true to sweep over all values below; false to use fixed defaults
USE_THRESHOLD_SWEEP=false
DYNAMIC_SPLIT_THRESHOLD_DEFAULT=1350             # For Dynamic Random (normal)
DYNAMIC_SPLIT_THRESHOLD_VECTORIZED_DEFAULT=500   # For Dynamic Random (vectorized)

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
  # "benchmarks/data/HIGGS_with_header.csv|class"
  # "benchmarks/data/SUSY_with_header.csv|class"
  # "benchmarks/data/epsilon_normalized_train.csv|label"

  # # Small
  # "benchmarks/data/cc18_binary_csv/task_14965_bank-marketing/repeat0_fold0_sample0_train.csv|Class"
  # "benchmarks/data/cc18_binary_csv/task_14952_PhishingWebsites/repeat0_fold0_sample0_train.csv|Result"
  # "benchmarks/data/cc18_binary_csv/task_29_credit-approval/repeat0_fold0_sample0_train.csv|class"
  # "benchmarks/data/cc18_binary_csv/task_167125_Internet-Advertisements/repeat0_fold0_sample0_train.csv|class"
)

# Synthetic trunk rows (comment out values to skip)
TRUNK_ROWS=(
  10000
  20000
  # 40000
  # 80000
  # 100000
  # 1000000
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

# Always: enable -> build -> disable -> run -> re-enable at end.
# All bazel builds are wrapped by bazel_build() so CPU E features are only
# enabled during the build itself, and disabled for every experiment run.
trap 'sudo benchmarks/src/utils/set_cpu_e_features.sh --enable' EXIT

bazel_build() {
  sudo benchmarks/src/utils/set_cpu_e_features.sh --enable
  bazel build "$@"
  sudo benchmarks/src/utils/set_cpu_e_features.sh --disable
}

logdir="benchmarks/results"
mkdir -p "$logdir"
logfile="${logdir}/e2e_runtime_${SUFFIX}.log"
csvfile="${logdir}/e2e_runtime_${SUFFIX}.csv"

if [[ -e "$logfile" ]]; then
  echo "ERROR: $logfile already exists. Use a different suffix or remove it." >&2
  exit 1
fi
if [[ -e "$csvfile" ]]; then
  echo "ERROR: $csvfile already exists. Use a different suffix or remove it." >&2
  exit 1
fi

# Parse log -> CSV. Log file is preserved for debugging.
finalize_log() {
  echo "Parsing log -> CSV..."
  if python3 benchmarks/src/utils/parse_log_to_csv.py "$logfile" "$csvfile"; then
    echo "CSV: $csvfile  (log kept at $logfile)"
  else
    echo "ERROR: parser failed; log kept at $logfile" >&2
    return 1
  fi
}

# Normal build (plain CPU binary). Skipped when only GPU experiments will
# run, since the GPU section does its own builds with --config=oblique_gpu.
if [[ "$RUN_CPU" == "true" || "$RUN_VECTORIZED" == "true" ]]; then
  bazel_build "${BAZEL_FLAGS[@]}" "$BUILD_TARGET"
else
  # Ensure features are disabled for experiments even when no initial build runs.
  sudo benchmarks/src/utils/set_cpu_e_features.sh --disable
fi
BINARY="./bazel-bin/examples/train_oblique_forest"

run_cmd() {
  echo "$*" | tee -a "$logfile"
  local times=()
  local i out t rc
  for ((i=1; i<=NUM_RUNS; i++)); do
    echo "----- Run $i/$NUM_RUNS -----" | tee -a "$logfile"
    rc=0
    out=$(bash -c "$*" 2>&1) || rc=$?
    echo "$out" | tee -a "$logfile"
    if (( rc != 0 )); then
      echo "WARNING: command exited with status $rc on run $i (continuing)" | tee -a "$logfile"
    fi
    # Parse "random_forest.cc Training block took: <X> s"
    t=$(echo "$out" | grep -oE 'Training block took:[[:space:]]*[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?' \
        | grep -oE '[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?' | tail -1)
    if [[ -n "$t" ]]; then
      times+=("$t")
    else
      echo "WARNING: Could not parse 'Training block took' from run $i" | tee -a "$logfile"
    fi
  done
  if [[ "${#times[@]}" -gt 0 ]]; then
    local sorted
    mapfile -t sorted < <(printf '%s\n' "${times[@]}" | sort -g)
    local n=${#sorted[@]}
    local mid=$(( n / 2 ))
    local median
    if (( n % 2 == 1 )); then
      median="${sorted[$mid]}"
    else
      median=$(awk -v a="${sorted[$((mid-1))]}" -v b="${sorted[$mid]}" 'BEGIN{printf "%.6f", (a+b)/2.0}')
    fi
    echo "MEDIAN of ${#times[@]}/${NUM_RUNS} runs: ${median} s  (samples: ${times[*]})" | tee -a "$logfile"
  else
    echo "MEDIAN of 0/${NUM_RUNS} runs: N/A" | tee -a "$logfile"
  fi
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
oblique_selected=false
if [[ "$RUN_CPU" != "true" ]]; then
  banner "Skipping CPU experiments (RUN_CPU=false)"
  # We still need `oblique_selected` for the GPU/Vectorized gates below, so
  # pre-compute it from SPLIT_TYPES without running any CPU experiments.
  for split in "${SPLIT_TYPES[@]}"; do
    if [[ "$split" != "Axis Aligned" ]]; then
      oblique_selected=true
    fi
  done
fi

if [[ "$RUN_CPU" == "true" ]]; then
banner "NORMAL EXPERIMENTS (no explicit vector ISA) histogram_num_bins=${histogram_num_bins}"

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
      if [[ "$USE_THRESHOLD_SWEEP" == "true" ]]; then
        thresholds=("${DYNAMIC_SPLIT_THRESHOLDS[@]}")
      else
        thresholds=("$DYNAMIC_SPLIT_THRESHOLD_DEFAULT")
      fi
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
fi  # RUN_CPU

# -------------------------
# GPU experiments (Oblique only). One build per GPU mode because `nodewise`
# requires --config=dfs_node_queue while `depthwise` uses the default BFS
# driver. Only HISTOGRAM_RANDOM / DYNAMIC_RANDOM_HISTOGRAM exercise the full
# GPU split pipeline; other methods fall back to GPU-apply + CPU-split.
# -------------------------

if [[ "$RUN_GPU" == "true" && "$oblique_selected" == "true" ]]; then
  for gpu_mode in "${GPU_MODES[@]}"; do
    case "$gpu_mode" in
      depthwise)
        gpu_extra_configs=(--config=oblique_gpu)
        ;;
      nodewise)
        gpu_extra_configs=(--config=oblique_gpu --config=dfs_node_queue)
        ;;
      *)
        banner "Unknown GPU mode '$gpu_mode'. Skipping."
        continue
        ;;
    esac

    bazel_build "${BAZEL_FLAGS[@]}" "${gpu_extra_configs[@]}" "$BUILD_TARGET"

    banner "GPU EXPERIMENTS [$gpu_mode] (Oblique only) histogram_num_bins=${histogram_num_bins}"
    echo "GPU MODE: $gpu_mode" | tee -a "$logfile"

    for method in "${METHODS[@]}"; do
      extra="${METHOD_EXTRA_ARGS[$method]:-}"

      # Build list of threshold values to iterate over
      if is_dynamic_method "$method"; then
        if [[ "$USE_THRESHOLD_SWEEP" == "true" ]]; then
          thresholds=("${DYNAMIC_SPLIT_THRESHOLDS[@]}")
        else
          thresholds=("$DYNAMIC_SPLIT_THRESHOLD_DEFAULT")
        fi
      else
        thresholds=("")
      fi

      for thresh in "${thresholds[@]}"; do
        thresh_arg=""
        thresh_label=""
        if [[ -n "$thresh" ]]; then
          thresh_arg="--dynamic_split_threshold=$thresh"
          thresh_label=" threshold=$thresh"
        fi

        banner "Running $method [GPU: $gpu_mode] with $histogram_num_bins bins${thresh_label}"

        # CSV datasets
        for entry in "${CSV_DATASETS[@]}"; do
          IFS='|' read -r path label <<<"$entry"
          cmd="$BINARY --input_mode csv --train_csv \"$path\" --label_col \"$label\" --feature_split_type \"Oblique\" --numerical_split_type \"$method\" --use_gpu=true $BASE_ARGS $extra $thresh_arg"
          run_cmd "$cmd"
        done

        # Trunk rows
        for rows in "${TRUNK_ROWS[@]}"; do
          cmd="$BINARY --input_mode trunk --rows $rows --feature_split_type \"Oblique\" --numerical_split_type \"$method\" --use_gpu=true $BASE_ARGS $extra $thresh_arg"
          run_cmd "$cmd"
        done
      done
    done
  done

  # Rebuild the CPU binary so any subsequent non-GPU sections run with the
  # pure-CPU binary (otherwise BINARY still points at the last GPU build).
  # Only needed if the Vectorized section will run after this.
  if [[ "$RUN_VECTORIZED" == "true" ]]; then
    bazel_build "${BAZEL_FLAGS[@]}" "$BUILD_TARGET"
  fi
fi

# -------------------------
# Vectorized experiments (Oblique only; Random, Dynamic Random Histogram)
# -------------------------

# Only run if Oblique was selected, vectorizable methods are present, and toggle is on
selected_vec_methods=()
if [[ "$RUN_VECTORIZED" == "true" && "$oblique_selected" == "true" ]]; then
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
  finalize_log
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
  finalize_log
  exit 0
fi

bazel_build "${BAZEL_FLAGS[@]}" "$vec_cfg" "$BUILD_TARGET"

banner "VECTORIZED EXPERIMENTS [${vec_name}] (Oblique only) histogram_num_bins=${histogram_num_bins}"
echo "USING INSTRUCTION SET: ${vec_name}" | tee -a "$logfile"

for method in "${selected_vec_methods[@]}"; do
  extra="${METHOD_EXTRA_ARGS[$method]:-}"

  # Build list of threshold values to iterate over
  if is_dynamic_method "$method"; then
    if [[ "$USE_THRESHOLD_SWEEP" == "true" ]]; then
      thresholds=("${DYNAMIC_SPLIT_THRESHOLDS[@]}")
    else
      thresholds=("$DYNAMIC_SPLIT_THRESHOLD_VECTORIZED_DEFAULT")
    fi
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
finalize_log