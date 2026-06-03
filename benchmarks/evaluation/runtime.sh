#!/usr/bin/env bash
set -euo pipefail

# Quick (default) vs Full evaluation.
#
# Workflow: use Quick to test whether a code change impacted runtime. When a
# >20% runtime improvement is observed on Quick, run Full to confirm.
#
# Quick: 7 runs/cmd; trunk rows*cols sweep at constant ~2e8 cells.
# Full:  7 runs/cmd; quick shapes plus large CSVs (HIGGS/SUSY/epsilon) and
#        extreme-aspect trunk (50M*4, 3M*4096, 3000*4M).
#
# Usage:  $0 [--full] <suffix>
#   <suffix> becomes part of the result filename, e.g. 'AWS_m7i' ->
#   runtime_quick_aws_m7i.csv (or runtime_full_aws_m7i.csv).

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
  echo "  e.g. '$0 AWS_m7i' -> runtime_quick_aws_m7i.csv" >&2
  exit 2
fi

###### Parameters

# Repetitions per command; median runtime is reported in the CSV.
NUM_RUNS=7
NUM_THREADS=-1
COMPUTE_OOB_PERFORMANCES=false  # set true to compute OOB metrics
# Ariel - ENSURE compute_oob_performances===== - it has an equal sign, not a blank space
# NUM_TREES and BASE_ARGS are set later, AFTER e-cores are disabled, so that
# nproc reflects the runtime CPU topology (P-cores only).

histogram_num_bins=64  # 64 -> AVX2, 256 -> AVX512 in vectorized mode.

RUN_CPU=false         # set false to skip the normal (CPU) experiments section
RUN_VECTORIZED=true   # set true to run AVX2/AVX512 vectorized experiments

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
  # "Exact"
  # "Random"
  "Dynamic Random Histogram"
  # "Equal Width"
  # "Dynamic Equal Width Histogram"
)

# Dynamic split thresholds (only affect Dynamic methods). Populate these from
# benchmarks/evaluation/dynamic_threshold_sweep.sh results.
DYNAMIC_SPLIT_THRESHOLD_DEFAULT=1350             # For Dynamic Random (normal)
DYNAMIC_SPLIT_THRESHOLD_VECTORIZED_DEFAULT=250   # For Dynamic Random (vectorized) — 2026-06-03 sweep on 3M*4096 AVX-2

# CSV datasets: large benchmark CSVs added in Full mode only. Entries are
# "path|label_col".
CSV_DATASETS=()

# Full-mode-only large CSVs.
FULL_EXTRA_CSV_DATASETS=(
  "benchmarks/data/HIGGS_with_header.csv|class"
  "benchmarks/data/SUSY_with_header.csv|class"
  "benchmarks/data/epsilon_normalized_train.csv|label"
)
if [[ "$MODE" == "full" ]]; then
  CSV_DATASETS+=("${FULL_EXTRA_CSV_DATASETS[@]}")
fi

# Synthetic trunk datasets as "rows|cols" pairs.
# Quick: rows*cols ~ 2e8 cells, sweeping the rows/cols ratio.
# Full: same shape sweep plus a larger row sweep at fixed cols=4096.
TRUNK_DATASETS=(
  "12500|16384"
  "25000|8192"
  "50000|4096"
  "100000|2048"
  "200000|1024"
)
if [[ "$MODE" == "full" ]]; then
  TRUNK_DATASETS+=(
    "50000000|4" # OOMs for Symmetric trees - comment out
    "3000000|4096"
    "3000|4000000"
  )
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
BAZEL_FLAGS=(-c opt --cxxopt="-O3" --cxxopt="-march=native" --repo_env=CC=icx --repo_env=CXX=icpx)
# Optional space-separated extra build configs/flags injected into every
# bazel_build (e.g. EXTRA_BAZEL_CONFIGS="--config=symmetric_nodewise_control").
# shellcheck disable=SC2206
EXTRA_BAZEL_CONFIGS_ARR=(${EXTRA_BAZEL_CONFIGS:-})
# Vectorized build configs (adjust if your repo uses different config names)
VEC_CONFIG_AVX2="--config=enable_std_upper_bound_avx2"
VEC_CONFIG_AVX512="--config=enable_std_upper_bound_avx512"

# Methods that have vectorized code paths. Per-split applicability:
#   Oblique:      Random, Dynamic Random Histogram
#   Axis Aligned: Random  (Dynamic Random Histogram is Oblique-only)
VECTORIZE_METHODS=("Random" "Dynamic Random Histogram")

# Always: enable -> build -> disable -> run -> re-enable at end.
# All bazel builds are wrapped by bazel_build() so CPU E features are only
# enabled during the build itself, and disabled for every experiment run.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SET_CPU_E_FEATURES="$(cd "$SCRIPT_DIR/../.." && pwd)/benchmarks/utils/set_cpu_e_features.sh"
trap 'sudo "$SET_CPU_E_FEATURES" --enable' EXIT

# .bazelrc pins --repo_env=CC=icx. If icx is not on PATH the build either
# hard-fails or (worse, across bazel/cc_configure cache states) silently
# falls back to gcc -- a gcc-built binary quietly corrupts every timing
# number. Source oneAPI here and abort loudly if icx still can't be found.
ensure_icx() {
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
    echo "       .bazelrc pins CC=icx; building with gcc would silently" >&2
    echo "       corrupt timings. Run 'source /opt/intel/oneapi/setvars.sh'" >&2
    echo "       (or set ONEAPI_SETVARS=/path/to/setvars.sh) and retry." >&2
    exit 1
  fi
  echo "Compiler: $(command -v icx)"
}

bazel_build() {
  ensure_icx
  sudo "$SET_CPU_E_FEATURES" --enable
  bazel build "$@" "${EXTRA_BAZEL_CONFIGS_ARR[@]}"
  sudo "$SET_CPU_E_FEATURES" --disable
}

logdir="benchmarks/results"
mkdir -p "$logdir"
logfile="${logdir}/runtime_${MODE}_${SUFFIX}.log"
csvfile="${logdir}/runtime_${MODE}_${SUFFIX}.csv"

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
# script aborts mid-run (set -e), since this function is never reached.
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

# Normal build (plain CPU binary). Skipped when only vectorized or GPU
# experiments will run, since those sections do their own config-specific
# builds.
if [[ "$RUN_CPU" == "true" ]]; then
  bazel_build "${BAZEL_FLAGS[@]}" "$BUILD_TARGET"
else
  # Ensure features are disabled for experiments even when no initial build runs.
  sudo "$SET_CPU_E_FEATURES" --disable
fi
BINARY="./bazel-bin/examples/train_oblique_forest"

# E-cores are now disabled (bazel_build did the disable, or the else branch did).
# Compute NUM_TREES from the *runtime* nproc so it reflects P-core count.
NUM_TREES=$(( $(nproc) * 5 )) # 5x cores to prevent skewness
BASE_ARGS="--num_trees=$NUM_TREES --num_threads=$NUM_THREADS --compute_oob_performances=$COMPUTE_OOB_PERFORMANCES"

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
    # Parse "random_forest.cc Training block took: <X> s". A missing line means
    # the binary crashed before training; under set -euo pipefail this kills
    # the script -- intentional, so a broken dataset surfaces immediately
    # instead of silently producing empty CSV rows.
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
    # Sample stddev (n-1). Undefined for n<2; report N/A so eyeballing the log
    # still shows that only one run survived.
    local stddev
    if (( n >= 2 )); then
      stddev=$(printf '%s\n' "${times[@]}" | awk '
        {x[NR]=$1; s+=$1}
        END{m=s/NR; ss=0; for(i=1;i<=NR;i++){d=x[i]-m; ss+=d*d}
            printf "%.6f", sqrt(ss/(NR-1))}')
    else
      stddev="N/A"
    fi
    echo "MEDIAN of ${#times[@]}/${NUM_RUNS} runs: ${median} s  STDDEV: ${stddev} s  (samples: ${times[*]})" | tee -a "$logfile"
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

    if is_dynamic_method "$method"; then
      thresholds=("$DYNAMIC_SPLIT_THRESHOLD_DEFAULT")
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

      # Trunk datasets (rows|cols)
      for entry in "${TRUNK_DATASETS[@]}"; do
        IFS='|' read -r rows cols <<<"$entry"
        cmd="$BINARY --input_mode trunk --rows $rows --cols $cols $feature_arg --numerical_split_type \"$method\" $BASE_ARGS $extra $thresh_arg"
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

      if is_dynamic_method "$method"; then
        thresholds=("$DYNAMIC_SPLIT_THRESHOLD_DEFAULT")
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

        # Trunk datasets (rows|cols)
        for entry in "${TRUNK_DATASETS[@]}"; do
          IFS='|' read -r rows cols <<<"$entry"
          cmd="$BINARY --input_mode trunk --rows $rows --cols $cols --feature_split_type \"Oblique\" --numerical_split_type \"$method\" --use_gpu=true $BASE_ARGS $extra $thresh_arg"
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
# Vectorized experiments. Per-split methods:
#   Oblique:      Random, Dynamic Random Histogram
#   Axis Aligned: Random (Dynamic Random Histogram is Oblique-only)
# -------------------------

if [[ "$RUN_VECTORIZED" != "true" ]]; then
  banner "Skipping vectorized experiments (RUN_VECTORIZED=false)"
  finalize_log
  exit 0
fi

# Return the subset of $METHODS that is vectorizable for the given split.
vec_methods_for_split() {
  local split="$1"
  local -a allowed_for_split
  if [[ "$split" == "Axis Aligned" ]]; then
    allowed_for_split=("Random")
  else
    allowed_for_split=("${VECTORIZE_METHODS[@]}")
  fi
  local m a
  for m in "${METHODS[@]}"; do
    for a in "${allowed_for_split[@]}"; do
      if [[ "$m" == "$a" ]]; then
        echo "$m"
        break
      fi
    done
  done
}

any_vec=false
for split in "${SPLIT_TYPES[@]}"; do
  if [[ -n "$(vec_methods_for_split "$split")" ]]; then
    any_vec=true
    break
  fi
done

if [[ "$any_vec" != "true" ]]; then
  banner "No vectorizable (split, method) combos selected; skipping vectorized experiments"
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

banner "VECTORIZED EXPERIMENTS [${vec_name}] histogram_num_bins=${histogram_num_bins}"
echo "USING INSTRUCTION SET: ${vec_name}" | tee -a "$logfile"

for split in "${SPLIT_TYPES[@]}"; do
  mapfile -t methods_to_run < <(vec_methods_for_split "$split")
  if [[ "${#methods_to_run[@]}" -eq 0 ]]; then
    banner "VECTORIZED [$split]: no vectorizable methods selected, skipping"
    continue
  fi

  if [[ "$split" == "Axis Aligned" ]]; then
    feature_arg='--feature_split_type "Axis Aligned"'
  else
    feature_arg='--feature_split_type "Oblique"'
  fi

  for method in "${methods_to_run[@]}"; do
    extra="${METHOD_EXTRA_ARGS[$method]:-}"

    if is_dynamic_method "$method"; then
      thresholds=("$DYNAMIC_SPLIT_THRESHOLD_VECTORIZED_DEFAULT")
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

      banner "Running $method [$split] [VECTORIZE: ${vec_name}] with $histogram_num_bins bins${thresh_label}"

      # CSV datasets
      for entry in "${CSV_DATASETS[@]}"; do
        IFS='|' read -r path label <<<"$entry"
        cmd="$BINARY --input_mode csv --train_csv \"$path\" --label_col \"$label\" $feature_arg --numerical_split_type \"$method\" $BASE_ARGS $extra $thresh_arg"
        run_cmd "$cmd"
      done

      # Trunk datasets (rows|cols)
      for entry in "${TRUNK_DATASETS[@]}"; do
        IFS='|' read -r rows cols <<<"$entry"
        cmd="$BINARY --input_mode trunk --rows $rows --cols $cols $feature_arg --numerical_split_type \"$method\" $BASE_ARGS $extra $thresh_arg"
        run_cmd "$cmd"
      done
    done
  done
done

# CPU features re-enabled by trap on exit
finalize_log
