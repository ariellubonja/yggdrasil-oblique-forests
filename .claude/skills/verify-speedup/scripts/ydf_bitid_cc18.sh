#!/usr/bin/env bash
# Bit-identity check for two harness binaries on 10 fixed CC18 tasks (fold 0):
#   ydf_bitid_cc18.sh <bin_A> <bin_B> <out_dir>
# Protocol flags come from EXTRA_TRAIN_ARGS (same as accuracy.sh); tree count
# from bench_num_trees. Prints a markdown table; exit 0 = all identical.
set -euo pipefail
BIN_A="$1"; BIN_B="$2"; OUT="$3"
REPO="$(git rev-parse --show-toplevel)"
source "$REPO/benchmarks/utils/bench_common.sh"
CC18="${ACCURACY_DATA_DIR:-$REPO/benchmarks/data/cc18_binary_csv}"
# Fixed, numeric, NaN-free (dataset policy 2026-09-04), spanning 4..1776 features.
TASKS=(task_10093_banknote-authentication task_37_diabetes task_9946_wdbc task_3917_kc1
       task_3902_pc4 task_9957_qsar-biodeg task_43_spambase task_9976_madelon
       task_9910_Bioresponse task_9952_phoneme)
NUM_TREES="$(bench_num_trees)"
mkdir -p "$OUT"
echo "| task | trees | result |"; echo "|---|---:|---|"
fail=0
for t in "${TASKS[@]}"; do
  train="$CC18/$t/repeat0_fold0_sample0_train.csv"
  [[ -f "$train" ]] || { echo "| $t | - | MISSING $train |"; fail=1; continue; }
  label=$(head -n 1 "$train" | awk -F',' '{print $NF}' | tr -d '\r\n ')
  for arm in A B; do
    bin="$BIN_A"; [[ $arm == B ]] && bin="$BIN_B"
    rm -rf "$OUT/$t.$arm"
    eval "$bin --input_mode csv --train_csv \"$train\" --label_col \"$label\" --num_trees=$NUM_TREES \
      --compute_oob_performances=true $EXTRA_TRAIN_ARGS --model_out_dir \"$OUT/$t.$arm\"" \
      > "$OUT/$t.$arm.log" 2>&1 || { echo "| $t | $NUM_TREES | arm $arm FAILED (see $OUT/$t.$arm.log) |"; fail=1; continue 2; }
  done
  if "$REPO/benchmarks/evaluation/compare_models.sh" "$OUT/$t.A" "$OUT/$t.B" > "$OUT/$t.compare.txt" 2>&1; then
    echo "| $t | $NUM_TREES | BIT-IDENTICAL |"
  else
    echo "| $t | $NUM_TREES | **DIFFER** ($(grep -m1 RESULT "$OUT/$t.compare.txt")) |"; fail=1
  fi
done
exit $fail
