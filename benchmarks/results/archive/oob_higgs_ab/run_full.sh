#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
OUT=benchmarks/results/oob_higgs_ab
TRAIN=benchmarks/data/HIGGS_with_header.csv
NUM_TREES=$(( $(nproc) * 5 ))
for arm in after before; do
  log="$OUT/higgs_full_${arm}.log"
  echo "ARM=$arm trees=$NUM_TREES train=$TRAIN (no test_csv)" | tee "$log"
  "$OUT/bin/train_$arm" --input_mode csv --train_csv "$TRAIN" --label_col class \
    --num_trees=$NUM_TREES --compute_oob_performances=true 2>&1 | tee -a "$log" | grep -E "Training block took|Final OOB|wall-time"
done
echo ALL_DONE
