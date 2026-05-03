#!/usr/bin/env bash
set -euo pipefail

# Quick (default) vs Full evaluation.
#
# Workflow: use Quick to test whether a code change impacted accuracy. When
# Quick agrees with the runtime evaluation that a change is worth keeping
# (e.g. >20% runtime improvement on runtime.sh's Quick), run Full to confirm.
#
# Quick: small dataset surface (all CC18 train folds; no trunk).
# Full:  Quick datasets + the large CSVs (HIGGS/SUSY/epsilon) and trunk
#        10k/20k/40k/80k/100k/1M.
# Seeds: 10 seeds run in BOTH modes (accuracy needs the variance regardless).
#
# Usage:  $0 [--full] <suffix>
#   <suffix> becomes part of the result filename, e.g. 'AWS_m7i' ->
#   accuracy_quick_aws_m7i.csv (or accuracy_full_aws_m7i.csv).

MODE="quick"
SUFFIX=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)  MODE="full";  shift ;;
    -h|--help)
      echo "Usage: $0 [--full] <suffix>" >&2
      exit 0 ;;
    --*)
      echo "ERROR: unknown flag '$1'" >&2
      echo "Usage: $0 [--full] <suffix>" >&2
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
  echo "Usage: $0 [--full] <suffix>" >&2
  echo "  e.g. '$0 AWS_m7i' -> accuracy_quick_aws_m7i.csv" >&2
  exit 2
fi

###### Parameters

if [[ "$MODE" == "quick" ]]; then
  SEEDS=(1)
else
  SEEDS=(1 2 3 4 5 6 7 8 9 10)
fi
if grep -q "Ultra 9 185H" /proc/cpuinfo 2>/dev/null; then
  NUM_TREES=30
else
  NUM_TREES=$(( $(nproc) * 5 )) # 5x cores to prevent skewness
fi
# OOB metrics must be on for accuracy parsing; not exposed as a toggle.
BASE_ARGS="--num_trees=$NUM_TREES --num_threads=-1 --compute_oob_performances=true"

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

# CSV datasets are built from the CC18 binary tasks (always) plus the large
# benchmark CSVs (Full mode only). Datasets entries are "path|label_col".
CC18_DIR="benchmarks/data/cc18_binary_csv"
if [[ ! -d "$CC18_DIR" ]]; then
  echo "ERROR: $CC18_DIR not found. Run from repo root." >&2
  exit 1
fi

CSV_DATASETS=()
# Only enumerate task_*/ folders; datasets renamed to issue_<name>/ are
# skipped (used to mark folds the binary cannot train on, e.g. an
# all-missing column that aborts dataspec creation).
for d in "$CC18_DIR"/task_*/; do
  csv="${d}repeat0_fold0_sample0_train.csv"
  [[ -f "$csv" ]] || continue
  # Label is always the last header column; strip BOM/CR/spaces defensively.
  label=$(head -n 1 "$csv" | awk -F',' '{print $NF}' | tr -d '\r\n ' | sed 's/^\xef\xbb\xbf//')
  CSV_DATASETS+=("$csv|$label")
done
if [[ "${#CSV_DATASETS[@]}" -eq 0 ]]; then
  echo "ERROR: found no CC18 datasets under $CC18_DIR" >&2
  echo "Run: python3 benchmarks/utils/download_cc18_datasets.py" >&2
  exit 1
fi

# Full-mode-only large CSVs (added on top of CC18).
FULL_EXTRA_CSV_DATASETS=(
  "benchmarks/data/HIGGS_with_header.csv|class"
  "benchmarks/data/SUSY_with_header.csv|class"
  "benchmarks/data/epsilon_normalized_train.csv|label"
)
if [[ "$MODE" == "full" ]]; then
  CSV_DATASETS+=("${FULL_EXTRA_CSV_DATASETS[@]}")
fi

# Synthetic trunk rows. Full mode only -- Quick is CC18-only.
TRUNK_ROWS=()
if [[ "$MODE" == "full" ]]; then
  TRUNK_ROWS=(10000 20000 40000 80000 100000 1000000)
fi

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

# CPU E features must stay enabled the whole time (build + run) so accuracy
# numbers are taken under a consistent CPU configuration. No
# enable/disable dance like runtime.sh.
sudo benchmarks/utils/set_cpu_e_features.sh --enable

bazel_build() {
  bazel build "$@"
}

logdir="benchmarks/results"
mkdir -p "$logdir"
logfile="${logdir}/accuracy_${MODE}_${SUFFIX}.log"
csvfile="${logdir}/accuracy_${MODE}_${SUFFIX}.csv"

if [[ -e "$logfile" ]]; then
  echo "ERROR: $logfile already exists. Use a different suffix or remove it." >&2
  exit 1
fi
if [[ -e "$csvfile" ]]; then
  echo "ERROR: $csvfile already exists. Use a different suffix or remove it." >&2
  exit 1
fi

# Parse log -> CSV. On parser success the log is deleted; if the parser fails
# the log is kept for debugging. The log is also kept implicitly when the
# script aborts mid-sweep (set -e), since this function is never reached.
finalize_log() {
  echo "Parsing log -> CSV..."
  if python3 benchmarks/utils/parse_log_to_csv.py "$logfile" "$csvfile"; then
    rm -f "$logfile"
    echo "CSV: $csvfile  (log deleted on success)"
  else
    echo "ERROR: parser failed; log kept at $logfile" >&2
    return 1
  fi
}

# Normal build (plain CPU binary). Skipped when only GPU experiments will
# run, since the GPU section does its own builds with --config=oblique_gpu.
if [[ "$RUN_CPU" == "true" || "$RUN_VECTORIZED" == "true" ]]; then
  bazel_build "${BAZEL_FLAGS[@]}" "$BUILD_TARGET"
fi
BINARY="./bazel-bin/examples/train_oblique_forest"

run_cmd() {
  echo "$*" | tee -a "$logfile"
  local seed out rc total=${#SEEDS[@]} i=0
  for seed in "${SEEDS[@]}"; do
    i=$((i+1))
    echo "----- Run $i/$total (seed=$seed) -----" | tee -a "$logfile"
    rc=0
    out=$(bash -c "$* --seed=$seed" 2>&1) || rc=$?
    echo "$out" | tee -a "$logfile"
    if (( rc != 0 )); then
      echo "WARNING: command exited with status $rc on seed $seed (continuing)" | tee -a "$logfile"
    fi
  done
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

finalize_log
