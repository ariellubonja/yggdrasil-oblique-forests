#!/usr/bin/env bash
set -euo pipefail

# Runtime evaluation.
#
# Workflow: use a low --runs count to test whether a code change impacted
# runtime. When a >20% runtime improvement is observed, re-run with more runs
# to confirm.
#
# --runs controls only the number of repetitions per command (median is
# reported); the datasets are the same regardless.
#
# Usage:  $0 [--runs=N] <suffix>     (default --runs=3)
#   <suffix> becomes part of the result filename, e.g. 'AWS_m7i' ->
#   aws_m7i.csv (the run count lives in the provenance header).

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

# Repetitions per command; median runtime is reported in the CSV.
# Controlled by --runs (default 3); NUM_RUNS is already set during arg parsing.
NUM_THREADS=-1
# Ariel - ENSURE compute_oob_performances===== - it has an equal sign, not a blank space
# NUM_TREES and BASE_ARGS are set later, AFTER e-cores are disabled, so that
# nproc reflects the runtime CPU topology (P-cores only).

histogram_num_bins=64  # 64 -> AVX2, 256 -> AVX512 in vectorized mode.

RUN_SCALAR=false      # run the scalar-binner experiments (SIMD compiled out)
RUN_VECTORIZED=true   # run the AVX2/AVX512 vectorized experiments

# The two flags are independent sections: RUN_SCALAR runs the SCALAR experiments,
# RUN_VECTORIZED runs the SIMD ones; both true runs both, both false runs
# neither. SIMD binning is default-ON in the code (runtime dispatch from cpuid
# + bin count), so the scalar section must compile it out — its build always
# adds this config. The vectorized section uses the default (SIMD) build.
SCALAR_CONFIG=(--config=disable_std_upper_bound_vectorization)

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

# CSV datasets. Entries are "path|label_col". Same in Quick and Full modes.
CSV_DATASETS=(
  "benchmarks/data/HIGGS_with_header.csv|class"
  # "benchmarks/data/SUSY_with_header.csv|class"
  # "benchmarks/data/epsilon_normalized_train.csv|label"
  )
# Synthetic trunk datasets as "rows|cols" pairs. Same in Quick and Full modes.
TRUNK_DATASETS=(
  # "30000000|4" # OOMs for Symmetric trees - comment out
  # "3000000|4096"
  "1500000|4096"
  # "300000|40000"
  "150000|40000"
  #"30000|400000"
  "15000|400000"
)

# Optional per-invocation dataset overrides. Used to isolate ONE dataset per run
# so that an OOM (common for immature full-depth/row-major paths) costs only that
# dataset instead of aborting the whole experiment under `set -e` before any CSV
# is written. Setting a var to the empty string selects ZERO datasets of that
# kind; use ';' to separate multiple entries. Unset => the defaults above.
if [[ "${CSV_DATASETS_OVERRIDE+x}" == "x" ]]; then
  CSV_DATASETS=()
  [[ -n "$CSV_DATASETS_OVERRIDE" ]] && IFS=';' read -r -a CSV_DATASETS <<<"$CSV_DATASETS_OVERRIDE"
fi
if [[ "${TRUNK_DATASETS_OVERRIDE+x}" == "x" ]]; then
  TRUNK_DATASETS=()
  [[ -n "$TRUNK_DATASETS_OVERRIDE" ]] && IFS=';' read -r -a TRUNK_DATASETS <<<"$TRUNK_DATASETS_OVERRIDE"
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
# EXTRA_BAZEL_CONFIGS / EXTRA_TRAIN_ARGS are read by bench_common.sh below.
# SIMD histogram binning is default-ON with runtime dispatch: the split-finder
# picks AVX2 (64 bins) / AVX-512 (256 bins) / scalar from cpuid + the bin count
# at RUNTIME, so no build config is needed to vectorize. The "vectorized"
# experiments below are just the default build driven at bins=64/256; the ISA is
# a label derived from the bin count, not a compile flag. The scalar baseline is
# the RUN_SCALAR section, built with SCALAR_CONFIG.

# Methods that have vectorized code paths. Per-split applicability:
#   Oblique:      Random, Dynamic Random Histogram
#   Axis Aligned: Random  (Dynamic Random Histogram is Oblique-only)
VECTORIZE_METHODS=("Random" "Dynamic Random Histogram")

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

# Scalar-section build (SIMD binner compiled out). Skipped when only
# vectorized or GPU experiments will run, since those sections do their own
# config-specific builds.
if [[ "$RUN_SCALAR" == "true" ]]; then
  bazel_build "${BAZEL_FLAGS[@]}" "${SCALAR_CONFIG[@]}" "$BUILD_TARGET"
else
  # Ensure features are disabled for experiments even when no initial build runs.
  bench_ecores --disable
fi
BINARY="./bazel-bin/examples/train_oblique_forest"

# E-cores are now disabled (bazel_build did the disable, or the else branch did).
# Compute NUM_TREES from the *runtime* nproc so it reflects P-core count.
# Boosting (GBT/MART) builds trees sequentially, so the "5x cores to prevent
# skewness" heuristic doesn't apply; use a fixed 30 trees instead.
if [[ "$EXTRA_TRAIN_ARGS" == *"ensemble_method=Boosting"* ]]; then
  NUM_TREES=300
else
  NUM_TREES=$(( $(bench_nproc) * 5 )) # 5x cores to prevent skewness
fi
BASE_ARGS="--num_trees=$NUM_TREES --num_threads=$NUM_THREADS"

# Provenance: written to the log AND a temp file; the temp file is prepended to
# the top of the CSV on a successful parse (see finalize_log) and then removed.
metafile="$(mktemp)"
bench_provenance_block \
  "NUM_TREES: $NUM_TREES  NUM_RUNS: $NUM_RUNS  NUM_THREADS: $NUM_THREADS  bins: $histogram_num_bins" \
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

# -------------------------
# Normal experiments (Oblique and/or Axis Aligned per SPLIT_TYPES)
# -------------------------
oblique_selected=false
if [[ "$RUN_SCALAR" != "true" ]]; then
  banner "Skipping scalar experiments (RUN_SCALAR=false)"
  # We still need `oblique_selected` for the GPU/Vectorized gates below, so
  # pre-compute it from SPLIT_TYPES without running any scalar experiments.
  for split in "${SPLIT_TYPES[@]}"; do
    if [[ "$split" != "Axis Aligned" ]]; then
      oblique_selected=true
    fi
  done
fi

if [[ "$RUN_SCALAR" == "true" ]]; then
banner "SCALAR EXPERIMENTS (SIMD binner compiled out) histogram_num_bins=${histogram_num_bins}"

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
        cmd="$BINARY --input_mode csv --train_csv \"$path\" --label_col \"$label\" $feature_arg --numerical_split_type \"$method\" $BASE_ARGS $extra $thresh_arg $EXTRA_TRAIN_ARGS"
        run_cmd "$cmd"
      done

      # Trunk datasets (rows|cols)
      for entry in "${TRUNK_DATASETS[@]}"; do
        IFS='|' read -r rows cols <<<"$entry"
        cmd="$BINARY --input_mode trunk --rows $rows --cols $cols $feature_arg --numerical_split_type \"$method\" $BASE_ARGS $extra $thresh_arg $EXTRA_TRAIN_ARGS"
        run_cmd "$cmd"
      done
    done
  done
done
fi  # RUN_SCALAR

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
          cmd="$BINARY --input_mode csv --train_csv \"$path\" --label_col \"$label\" --feature_split_type \"Oblique\" --numerical_split_type \"$method\" --use_gpu=true $BASE_ARGS $extra $thresh_arg $EXTRA_TRAIN_ARGS"
          run_cmd "$cmd"
        done

        # Trunk datasets (rows|cols)
        for entry in "${TRUNK_DATASETS[@]}"; do
          IFS='|' read -r rows cols <<<"$entry"
          cmd="$BINARY --input_mode trunk --rows $rows --cols $cols --feature_split_type \"Oblique\" --numerical_split_type \"$method\" --use_gpu=true $BASE_ARGS $extra $thresh_arg $EXTRA_TRAIN_ARGS"
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

# ISA is selected at RUNTIME from the bin count; here it is only a label.
vec_name=""
if [[ "$histogram_num_bins" -eq 64 ]]; then
  vec_name="AVX2"
elif [[ "$histogram_num_bins" -eq 256 ]]; then
  vec_name="AVX512"
else
  banner "Vectorized experiments require histogram_num_bins to be 64 (AVX2) or 256 (AVX512). Current: $histogram_num_bins. Skipping vectorized experiments."
  finalize_log
  exit 0
fi

# Default build already compiles the SIMD binners; no vectorization config.
bazel_build "${BAZEL_FLAGS[@]}" "$BUILD_TARGET"

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
        cmd="$BINARY --input_mode csv --train_csv \"$path\" --label_col \"$label\" $feature_arg --numerical_split_type \"$method\" $BASE_ARGS $extra $thresh_arg $EXTRA_TRAIN_ARGS"
        run_cmd "$cmd"
      done

      # Trunk datasets (rows|cols)
      for entry in "${TRUNK_DATASETS[@]}"; do
        IFS='|' read -r rows cols <<<"$entry"
        cmd="$BINARY --input_mode trunk --rows $rows --cols $cols $feature_arg --numerical_split_type \"$method\" $BASE_ARGS $extra $thresh_arg $EXTRA_TRAIN_ARGS"
        run_cmd "$cmd"
      done
    done
  done
done

# CPU features re-enabled by trap on exit
finalize_log
