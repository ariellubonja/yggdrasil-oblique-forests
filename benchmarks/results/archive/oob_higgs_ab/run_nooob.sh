#!/usr/bin/env bash
set -euo pipefail
cd /home/ariel/prog/ydf/yggdrasil-oblique-forests
OUT=benchmarks/results/oob_higgs_ab
until grep -q ALL_DONE "$OUT/run_full.out"; do sleep 5; done
TRAIN=benchmarks/data/HIGGS_with_header.csv
NUM_TREES=$(( $(nproc) * 5 ))
log="$OUT/higgs_full_after_nooob.log"
echo "ARM=after(no-OOB) trees=$NUM_TREES train=$TRAIN (no test_csv)" | tee "$log"
"$OUT/bin/train_after" --input_mode csv --train_csv "$TRAIN" --label_col class \
  --num_trees=$NUM_TREES --compute_oob_performances=false 2>&1 | tee -a "$log" | grep -E "Training block took|Final OOB|wall-time"
echo NOOOB_DONE
