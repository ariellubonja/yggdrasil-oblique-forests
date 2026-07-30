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

# Every CC18 fold ships a matching _test.csv, so the binary is given --test_csv
# and the parser prefers the held-out 'test-accuracy:' line -- the only number
# comparable across RF and GBT. OOB (RF) / train-accuracy (GBT) stays on as a
# fallback, used only if a test fold is missing / fails to load. The model seed
# is fixed in the binary (--seed default = 1), so folds -- not seeds -- are the
# sample axis: exactly one training run per fold, no seed sweep.
# NOTE: --num_trees is NOT here; it is set per ensemble (see ensemble_num_trees).
BASE_ARGS="--num_threads=-1 --compute_oob_performances=true"

# Ensemble methods to run. Each combines with SPLIT_TYPES below to form the
# 'algorithm' label in the CSV:
#   Oblique  + Bagging  -> SPORF     Oblique  + Boosting -> SPO-GBT
#   non-obl. + Bagging  -> RF        non-obl. + Boosting -> GBT
# Default runs both oblique variants (SPORF and SPO-GBT). Non-oblique RF/GBT are
# only produced when "Axis Aligned" is uncommented in SPLIT_TYPES.
ENSEMBLES=(
  "Bagging"    # Random Forest  (Oblique -> SPORF)
  # "Boosting"   # Gradient Boosted Trees (Oblique -> SPO-GBT)
)

# num_trees per ensemble: GBT is boosted (sequential, deeper effect per tree) so
# a smaller fixed count; RF gets 5x cores to keep per-thread tree counts even.
ensemble_num_trees() {
  if [[ "$1" == "Boosting" ]]; then
    echo 300
  else
    echo $(( $(bench_nproc) * 5 ))
  fi
}

# CSV 'algorithm' family label for banners (must match parse_log_to_csv.py).
family_label() {  # $1 = ensemble (Bagging|Boosting), $2 = split type
  local ens="$1" split="$2"
  if [[ "$split" == "Axis Aligned" ]]; then
    [[ "$ens" == "Boosting" ]] && echo "GBT" || echo "RF"
  else
    [[ "$ens" == "Boosting" ]] && echo "SPO-GBT" || echo "SPORF"
  fi
}

histogram_num_bins=64  # 64 -> AVX2, 256 -> AVX512 in vectorized mode.

RUN_SCALAR=true       # run the scalar-binner experiments (SIMD compiled out)
RUN_VECTORIZED=true   # run the AVX2/AVX512 vectorized experiments (Vectorized Random)

# AVX2/AVX512 are x86-only instruction sets. On any non-x86 host (e.g. Apple
# Silicon) the SIMD binner never engages anyway (the code gates on __x86_64__),
# so the "AVX2"/"AVX512" run labels would be lies. Force vectorization off there.
if [[ "$RUN_VECTORIZED" == "true" && "$(uname -m)" != "x86_64" ]]; then
  echo "CPU ($(uname -m)) does not support AVX instructions; disabling vectorized experiments." >&2
  RUN_VECTORIZED=false
fi

# The two flags are independent sections: RUN_SCALAR runs the SCALAR
# experiments, RUN_VECTORIZED runs the SIMD ones; both true runs both, both
# false runs neither. SIMD binning is default-ON in the code (runtime dispatch
# from cpuid + bin count), so the scalar section must compile it out — its
# build always adds this config. The vectorized section uses the default (SIMD)
# build.
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

# Dynamic split threshold (only affects Dynamic methods)
DYNAMIC_SPLIT_THRESHOLD_DEFAULT=1350             # For Dynamic Random (normal)
DYNAMIC_SPLIT_THRESHOLD_VECTORIZED_DEFAULT=250   # For Dynamic Random (vectorized)

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
# EXTRA_BAZEL_CONFIGS / EXTRA_TRAIN_ARGS are read by bench_common.sh below. Do
# NOT put --ensemble_method or --num_trees in EXTRA_TRAIN_ARGS: the script owns
# both (see ENSEMBLES / ensemble_num_trees).
# SIMD histogram binning is default-ON with runtime dispatch: the split-finder
# picks AVX2 (64 bins) / AVX-512 (256 bins) / scalar from cpuid + the bin count
# at RUNTIME, so no build config is needed to vectorize. The "vectorized"
# experiments below are just the default build driven at bins=64/256; the ISA is
# a label derived from the bin count, not a compile flag. The scalar baseline is
# the RUN_SCALAR section, built with SCALAR_CONFIG.

# Vectorization applies only to these methods (Oblique only)
VECTORIZE_METHODS=("Random" "Dynamic Random Histogram")

# Shared plumbing (ensure_icx, bazel_build + EXTRA_BAZEL_CONFIGS,
# EXTRA_TRAIN_ARGS, provenance, banner). CPU E features must stay enabled the
# whole time (build + run), so bench_common's e-core toggling is switched off.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ECORE_TOGGLE=false
source "$SCRIPT_DIR/../utils/bench_common.sh"
bench_ecores --enable force

logdir="benchmarks/results"
mkdir -p "$logdir"
logfile="${logdir}/accuracy_${SUFFIX}.log"
csvfile="${logdir}/accuracy_${SUFFIX}.csv"
BENCH_LOGFILE="$logfile"

bench_require_absent "$logfile" "$csvfile"

# Parse log -> CSV. On parser success the log is deleted; if the parser fails
# the log is kept for debugging. The log is also kept implicitly when the
# script aborts mid-sweep (set -e), since this function is never reached.
finalize_log() {
  echo "Parsing log -> CSV..."
  if python3 benchmarks/utils/parse_log_to_csv.py "$logfile" "$csvfile"; then
    bench_prepend_provenance "$csvfile" "$metafile"
    rm -f "$logfile"
    echo "CSV: $csvfile  (log deleted on success)"
  else
    echo "ERROR: parser failed; log kept at $logfile" >&2
    return 1
  fi
}

# Provenance: the temp file is prepended to the CSV on a successful parse.
metafile="$(mktemp)"
bench_provenance_block \
  "NUM_FOLDS: $NUM_FOLDS  ENSEMBLES: ${ENSEMBLES[*]}  bins: $histogram_num_bins" \
  | tee -a "$logfile" "$metafile"

# Scalar-section build (SIMD binner compiled out). Only needed when the scalar
# experiments run; the vectorized and GPU sections do their own builds.
if [[ "$RUN_SCALAR" == "true" ]]; then
  bazel_build "${BAZEL_FLAGS[@]}" "${SCALAR_CONFIG[@]}" "$BUILD_TARGET"
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

for ensemble in "${ENSEMBLES[@]}"; do
  nt="$(ensemble_num_trees "$ensemble")"
  ensemble_arg="--ensemble_method $ensemble --num_trees=$nt"

for split in "${SPLIT_TYPES[@]}"; do
  fam="$(family_label "$ensemble" "$split")"
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
      banner "$fam: No compatible methods selected (need Exact, Random, or Equal Width). Skipping."
      continue
    fi
    banner "$fam EXPERIMENTS feature_split_type=Axis Aligned histogram_num_bins=${histogram_num_bins}"
    feature_arg='--feature_split_type "Axis Aligned"'
  else
    # Oblique: use METHODS as-is
    methods_to_run=("${METHODS[@]}")
    oblique_selected=true
    banner "$fam EXPERIMENTS histogram_num_bins=${histogram_num_bins}"
    feature_arg='--feature_split_type "Oblique"'
  fi

  for method in "${methods_to_run[@]}"; do
    extra="${METHOD_EXTRA_ARGS[$method]:-}"

    # Build list of threshold values to iterate over
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
        banner "Running $fam $method with $histogram_num_bins bins${thresh_label}"
      else
        banner "Running $fam $method${thresh_label}"
      fi

      # CSV datasets: one run_cv per task, sweeping all NUM_FOLDS folds.
      for entry in "${CSV_DATASETS[@]}"; do
        IFS='|' read -r dir label <<<"$entry"
        rest="--label_col \"$label\" $feature_arg --numerical_split_type \"$method\" $ensemble_arg $BASE_ARGS $extra $thresh_arg $EXTRA_TRAIN_ARGS"
        run_cv "$dir" "$rest"
      done
    done
  done
done
done  # ENSEMBLES
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

    for ensemble in "${ENSEMBLES[@]}"; do
      nt="$(ensemble_num_trees "$ensemble")"
      ensemble_arg="--ensemble_method $ensemble --num_trees=$nt"
      fam="$(family_label "$ensemble" "Oblique")"

    for method in "${METHODS[@]}"; do
      extra="${METHOD_EXTRA_ARGS[$method]:-}"

      # Build list of threshold values to iterate over
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

        banner "Running $fam $method [GPU: $gpu_mode] with $histogram_num_bins bins${thresh_label}"

        # CSV datasets: one run_cv per task, sweeping all NUM_FOLDS folds.
        for entry in "${CSV_DATASETS[@]}"; do
          IFS='|' read -r dir label <<<"$entry"
          rest="--label_col \"$label\" --feature_split_type \"Oblique\" --numerical_split_type \"$method\" --use_gpu=true $ensemble_arg $BASE_ARGS $extra $thresh_arg $EXTRA_TRAIN_ARGS"
          run_cv "$dir" "$rest"
        done
      done
    done
    done  # ENSEMBLES
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

banner "VECTORIZED EXPERIMENTS [${vec_name}] (Oblique only) histogram_num_bins=${histogram_num_bins}"
echo "USING INSTRUCTION SET: ${vec_name}" | tee -a "$logfile"

# The "USING INSTRUCTION SET" line above latches the parser into naming every
# subsequent run's method "Vectorized_<method>" (e.g. Vectorized_Random), so no
# per-command tag is needed here. Vectorized runs are Oblique-only, so ensembles
# map to SPORF (Bagging) / SPO-GBT (Boosting).
for ensemble in "${ENSEMBLES[@]}"; do
  nt="$(ensemble_num_trees "$ensemble")"
  ensemble_arg="--ensemble_method $ensemble --num_trees=$nt"
  fam="$(family_label "$ensemble" "Oblique")"

for method in "${selected_vec_methods[@]}"; do
  extra="${METHOD_EXTRA_ARGS[$method]:-}"

  # Build list of threshold values to iterate over
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

    banner "Running $fam Vectorized $method [${vec_name}] with $histogram_num_bins bins${thresh_label}"

    # CSV datasets: one run_cv per task, sweeping all NUM_FOLDS folds.
    for entry in "${CSV_DATASETS[@]}"; do
      IFS='|' read -r dir label <<<"$entry"
      rest="--label_col \"$label\" --numerical_split_type \"$method\" $ensemble_arg $BASE_ARGS $extra $thresh_arg $EXTRA_TRAIN_ARGS"
      run_cv "$dir" "$rest"
    done
  done
done
done  # ENSEMBLES

finalize_log
