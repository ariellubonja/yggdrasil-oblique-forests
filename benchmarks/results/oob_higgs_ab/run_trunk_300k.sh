#!/usr/bin/env bash
set -euo pipefail
cd /home/ariel/prog/ydf/yggdrasil-oblique-forests
OUT=benchmarks/results/oob_higgs_ab
NUM_TREES=$(( $(nproc) * 5 ))
run() {  # $1=arm $2=oob $3=tag
  log="$OUT/trunk300k_4096_$3.log"
  echo "ARM=$1 oob=$2 trees=$NUM_TREES trunk 300000x4096" | tee "$log"
  "$OUT/bin/train_$1" --input_mode trunk --rows 300000 --cols 4096 --num_trees=$NUM_TREES \
    --compute_oob_performances=$2 2>&1 | tee -a "$log" | grep -E "Training block took|Final OOB|wall-time"
}
run after  true  after_oob
run before true  before_oob
run after  false after_nooob
echo ALL_DONE
