#!/usr/bin/env bash
set -euo pipefail

# Accuracy evaluation: 10-fold cross-validation over all CC18 binary tasks.
# Each task ships a matching train/test split per fold
# (repeat0_fold{0..9}_sample0_{train,test}.csv); every fold is trained at the
# binary's fixed --seed and evaluated on its held-out _test.csv. The per-fold
# 'test-accuracy:' values are the samples; mean +/- std over folds is the CV
# estimate. RF and GBT are run as separate invocations (GBT via
# EXTRA_TRAIN_ARGS="--ensemble_method Boosting ...") over the SAME folds, so the
# preferred held-out 'test-accuracy:' number is directly comparable between them.
#
# Usage:  $0 <suffix>
#   <suffix> becomes part of the result filename, e.g. 'AWS_m7i' ->
#   accuracy_aws_m7i.csv.

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
  echo "  e.g. '$0 AWS_m7i' -> accuracy_aws_m7i.csv" >&2
  exit 2
fi

###### Parameters

# Number of CV folds per task (repeat0_fold{0..NUM_FOLDS-1}_sample0).
NUM_FOLDS=10

NUM_TREES=300
  
# Every CC18 fold ships a matching _test.csv, so the binary is given --test_csv
# and the parser prefers the held-out 'test-accuracy:' line -- the only number
# comparable across RF and GBT. OOB (RF) / train-accuracy (GBT) stays on as a
# fallback, used only if a test fold is missing / fails to load. The model seed
# is fixed in the binary (--seed default = 1), so folds -- not seeds -- are the
# sample axis: exactly one training run per fold, no seed sweep.
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

# CSV datasets are built from the CC18 binary tasks. Entries are
# "path|label_col".
CC18_DIR="benchmarks/data/cc18_binary_csv"
if [[ ! -d "$CC18_DIR" ]]; then
  echo "ERROR: $CC18_DIR not found. Run from repo root." >&2
  exit 1
fi

CSV_DATASETS=()
# Only enumerate task_*/ folders; datasets renamed to issue_<name>/ are
# skipped (used to mark folds the binary cannot train on, e.g. an
# all-missing column that aborts dataspec creation). Each entry is
# "task_dir|label"; run_cv derives the per-fold train/test CSV paths from the
# task dir. Fold 0's train CSV must exist for the task to be included; the label
# is read from it (identical header across all folds).
for d in "$CC18_DIR"/task_*/; do
  csv="${d}repeat0_fold0_sample0_train.csv"
  [[ -f "$csv" ]] || continue
  # Label is always the last header column; strip BOM/CR/spaces defensively.
  label=$(head -n 1 "$csv" | awk -F',' '{print $NF}' | tr -d '\r\n ' | sed 's/^\xef\xbb\xbf//')
  CSV_DATASETS+=("$d|$label")
done
if [[ "${#CSV_DATASETS[@]}" -eq 0 ]]; then
  echo "ERROR: found no CC18 datasets under $CC18_DIR" >&2
  echo "Run: python3 benchmarks/data/download_cc18_datasets.py" >&2
  exit 1
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
# Optional space-separated extra build configs/flags injected into every
# bazel_build (e.g. EXTRA_BAZEL_CONFIGS="--config=symmetric_nodewise_control").
# shellcheck disable=SC2206
EXTRA_BAZEL_CONFIGS_ARR=(${EXTRA_BAZEL_CONFIGS:-})
# Optional extra train_oblique_forest flags injected into every binary command
# (e.g. EXTRA_TRAIN_ARGS="--ensemble_method Boosting --shrinkage=0.1").
EXTRA_TRAIN_ARGS=${EXTRA_TRAIN_ARGS:-}
# Vectorized build configs (adjust if your repo uses different config names)
VEC_CONFIG_AVX2="--config=enable_std_upper_bound_avx2"
VEC_CONFIG_AVX512="--config=enable_std_upper_bound_avx512"

if [[ "$EXTRA_TRAIN_ARGS" == *"ensemble_method=Boosting"* ]]; then
  NUM_TREES=300
else
  NUM_TREES=$(( $(nproc) * 5 )) # 5x cores to prevent skewness
fi

# Vectorization applies only to these methods (Oblique only)
VECTORIZE_METHODS=("Random" "Dynamic Random Histogram")

# CPU E features must stay enabled the whole time (build + run) so accuracy
# numbers are taken under a consistent CPU configuration. No
# enable/disable dance like runtime.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SET_CPU_E_FEATURES="$(cd "$SCRIPT_DIR/../.." && pwd)/benchmarks/utils/set_cpu_e_features.sh"
"$SET_CPU_E_FEATURES" --enable

# .bazelrc pins --repo_env=CC=icx. If icx is not on PATH the build either
# hard-fails or (worse, across bazel/cc_configure cache states) silently
# falls back to gcc. Source oneAPI here and abort loudly if icx is missing.
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
      set +u +e                       # setvars.sh is not -e/-u clean
      # shellcheck disable=SC1090
      source "$setvars" >/dev/null 2>&1
      set -u -e
    fi
  fi
  if ! command -v icx >/dev/null 2>&1 || ! command -v icpx >/dev/null 2>&1; then
    echo "ERROR: icx/icpx not on PATH and could not source oneAPI." >&2
    echo "       .bazelrc pins CC=icx. Run 'source /opt/intel/oneapi/setvars.sh'" >&2
    echo "       (or set ONEAPI_SETVARS=/path/to/setvars.sh) and retry." >&2
    exit 1
  fi
  echo "Compiler: $(command -v icx)"
}

bazel_build() {
  ensure_icx
  bazel build "$@" "${EXTRA_BAZEL_CONFIGS_ARR[@]}"
}

logdir="benchmarks/results"
mkdir -p "$logdir"
logfile="${logdir}/accuracy_${SUFFIX}.log"
csvfile="${logdir}/accuracy_${SUFFIX}.csv"

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

# Run one algorithm over all NUM_FOLDS CV folds of a task at the binary's fixed
# seed. $1 = task dir (trailing slash); $2 = every flag except the fold-varying
# --input_mode/--train_csv/--test_csv (i.e. --label_col + split/method +
# BASE_ARGS + EXTRA_TRAIN_ARGS). One representative command line (fold 0) is
# logged first so parse_log_to_csv can extract dataset+algorithm; each fold then
# runs under its own "----- Run k/NUM_FOLDS (fold=k) -----" marker, and its
# held-out 'test-accuracy:' becomes that fold's sample. A missing fold still
# emits its marker (blank sample) so CSV columns stay fold-aligned.
run_cv() {
  local dir="$1" rest="$2"
  echo "$BINARY --input_mode csv --train_csv \"${dir}repeat0_fold0_sample0_train.csv\" $rest" | tee -a "$logfile"
  local fold train test test_arg cmd out rc
  for (( fold=0; fold<NUM_FOLDS; fold++ )); do
    train="${dir}repeat0_fold${fold}_sample0_train.csv"
    test="${dir}repeat0_fold${fold}_sample0_test.csv"
    echo "----- Run $((fold+1))/$NUM_FOLDS (fold=$fold) -----" | tee -a "$logfile"
    if [[ ! -f "$train" ]]; then
      echo "WARNING: missing $train (skipping fold $fold)" | tee -a "$logfile"
      continue
    fi
    test_arg=""
    [[ -f "$test" ]] && test_arg="--test_csv \"$test\""
    cmd="$BINARY --input_mode csv --train_csv \"$train\" $rest $test_arg"
    rc=0
    out=$(bash -c "$cmd" 2>&1) || rc=$?
    echo "$out" | tee -a "$logfile"
    if (( rc != 0 )); then
      echo "WARNING: command exited with status $rc on fold $fold (continuing)" | tee -a "$logfile"
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

      # CSV datasets: one run_cv per task, sweeping all NUM_FOLDS folds.
      for entry in "${CSV_DATASETS[@]}"; do
        IFS='|' read -r dir label <<<"$entry"
        rest="--label_col \"$label\" $feature_arg --numerical_split_type \"$method\" $BASE_ARGS $extra $thresh_arg $EXTRA_TRAIN_ARGS"
        run_cv "$dir" "$rest"
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

        # CSV datasets: one run_cv per task, sweeping all NUM_FOLDS folds.
        for entry in "${CSV_DATASETS[@]}"; do
          IFS='|' read -r dir label <<<"$entry"
          rest="--label_col \"$label\" --feature_split_type \"Oblique\" --numerical_split_type \"$method\" --use_gpu=true $BASE_ARGS $extra $thresh_arg $EXTRA_TRAIN_ARGS"
          run_cv "$dir" "$rest"
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

    # CSV datasets: one run_cv per task, sweeping all NUM_FOLDS folds.
    for entry in "${CSV_DATASETS[@]}"; do
      IFS='|' read -r dir label <<<"$entry"
      rest="--label_col \"$label\" --numerical_split_type \"$method\" $BASE_ARGS $extra $thresh_arg $EXTRA_TRAIN_ARGS"
      run_cv "$dir" "$rest"
    done
  done
done

finalize_log
