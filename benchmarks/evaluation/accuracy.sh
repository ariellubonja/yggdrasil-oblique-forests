#!/usr/bin/env bash
set -euo pipefail

# Accuracy evaluation: 10-fold cross-validation over all CC18 binary tasks.
# Each task ships a matching train/test split per fold
# (repeat0_fold{0..9}_sample0_{train,test}.csv); every fold is trained at the
# binary's fixed --seed and evaluated on its held-out _test.csv. The per-fold
# 'test-accuracy:' values are the samples; mean +/- std over folds is the CV
# estimate.
#
# Usage:  $0 <suffix>
#   <suffix> becomes part of the result filename, e.g. 'AWS_m7i' ->
#   accuracy_aws_m7i.csv.
#
# The script is agnostic about WHAT is evaluated: it builds once and runs the
# CV sweep. Everything else comes from the environment:
#   EXTRA_BAZEL_CONFIGS  build configs, e.g. "--config=symmetric_optimized"
#   EXTRA_TRAIN_ARGS     binary flags, e.g. '--numerical_split_type "Dynamic
#                        Random Histogram" --histogram_num_bins=64
#                        --dynamic_split_threshold=1350 --ensemble_method Boosting'
# Both are recorded in the CSV provenance header. The CSV's `algorithm` column
# is derived by the parser from the command line, so pass the flags that
# identify the experiment (feature_split_type / numerical_split_type /
# ensemble_method / dynamic_split_threshold) in EXTRA_TRAIN_ARGS. --num_trees is
# the one flag the script owns (see NUM_TREES below).
#
# Ariel - ENSURE compute_oob_performances===== - it has an equal sign, not a blank space

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
BASE_ARGS="--compute_oob_performances=true"

# CSV datasets are built from the CC18 binary tasks. Entries are
# "task_dir|label_col".
# ACCURACY_DATA_DIR overrides the task dir (e.g. benchmarks/data/tabular_folds
# from make_folds.py); any task_*/ tree in the CC18 fold layout works.
CC18_DIR="${ACCURACY_DATA_DIR:-benchmarks/data/cc18_binary_csv}"
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

BUILD_TARGET="//examples:train_oblique_forest"
BAZEL_FLAGS=(-c opt --cxxopt="-O3" --cxxopt="-march=native")

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

confirm_overwrite "$logfile" "$csvfile"

# Parse log -> CSV. On parser success the log is deleted; if the parser fails
# the log is kept for debugging. The log is also kept implicitly when the
# script aborts mid-sweep (set -e), since this function is never reached.
finalize_log() {
  echo "Parsing log -> CSV..."
  if python3 benchmarks/utils/parse_log_to_csv.py "$logfile" "$csvfile" accuracy; then
    # The parser may emit side CSVs (_auc/_logloss) next to the main one;
    # prepend the same provenance block to every file it wrote.
    local f tmp
    for f in "${csvfile%.csv}_auc.csv" "${csvfile%.csv}_logloss.csv"; do
      [[ -f "$f" ]] || continue
      tmp=$(mktemp)
      cat "$metafile" "$f" >"$tmp" && mv "$tmp" "$f"
    done
    bench_prepend_provenance "$csvfile" "$metafile"
    rm -f "$logfile"
    echo "CSV: $csvfile  (log deleted on success)"
  else
    echo "ERROR: parser failed; log kept at $logfile" >&2
    return 1
  fi
}

# bazel_build appends EXTRA_BAZEL_CONFIGS.
bazel_build "${BAZEL_FLAGS[@]}" "$BUILD_TARGET"
BINARY="./bazel-bin/examples/train_oblique_forest"

# 5x cores to prevent skewness, or a fixed count under Boosting.
NUM_TREES="$(bench_num_trees)"
BASE_ARGS="--num_trees=$NUM_TREES $BASE_ARGS"

# Provenance: the temp file is prepended to the CSV on a successful parse.
metafile="$(mktemp)"
bench_provenance_block \
  "NUM_FOLDS: $NUM_FOLDS  NUM_TREES: $NUM_TREES" \
  | tee -a "$logfile" "$metafile"

# Run one algorithm over all NUM_FOLDS CV folds of a task at the binary's fixed
# seed. $1 = task dir (trailing slash); $2 = every flag except the fold-varying
# --input_mode/--train_csv/--test_csv (i.e. --label_col + BASE_ARGS +
# EXTRA_TRAIN_ARGS). One representative command line (fold 0) is logged first so
# parse_log_to_csv can extract dataset+algorithm; each fold then runs under its
# own "----- Run k/NUM_FOLDS (fold=k) -----" marker, and its held-out
# 'test-accuracy:' becomes that fold's sample. A missing fold still emits its
# marker (blank sample) so CSV columns stay fold-aligned.
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

banner "ACCURACY CV  configs: ${EXTRA_BAZEL_CONFIGS:-<none>}  args: ${EXTRA_TRAIN_ARGS:-<none>}"

# One run_cv per CC18 task, sweeping all NUM_FOLDS folds.
for entry in "${CSV_DATASETS[@]}"; do
  IFS='|' read -r dir label <<<"$entry"
  run_cv "$dir" "--label_col \"$label\" $BASE_ARGS $EXTRA_TRAIN_ARGS"
done

finalize_log
