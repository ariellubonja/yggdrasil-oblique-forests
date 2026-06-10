#!/bin/bash
# dw1 column-centric follow-ups: block-size ablation + shape generalization.
# Fully local; safe offline. Sequential; failures don't stop the queue.
cd /home/ariel/prog/ydf/yggdrasil-oblique-forests
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
set -u

LOG=benchmarks/results/dw1_queue_2026-06-10.log
echo "=== dw1 queue start $(date) ===" >> "$LOG"

run() {
  local name="$1"; shift
  echo "--- $name start $(date) ---" >> "$LOG"
  "$@" >> "$LOG" 2>&1
  echo "--- $name exit=$? $(date) ---" >> "$LOG"
}

CH3M="python3 benchmarks/profiling/parallel_chrono.py --rows=3000000 --cols=4096 --num_trees=5 --num_threads=1 --feature_split_type=Oblique --depthwise_1_pass"

# Block-size ablation at 3M x 4096 (default 4M floats = 16 MiB already run)
YDF_DW1_BLOCK_FLOATS=1048576 run dw1_blk4mib $CH3M \
  --numerical_split_type="Dynamic Random Histogram" \
  --experiment_name=dw1_colcentric_blk4mib_5trees_pcore_2026-06-10

YDF_DW1_BLOCK_FLOATS=16777216 run dw1_blk64mib $CH3M \
  --numerical_split_type="Dynamic Random Histogram" \
  --experiment_name=dw1_colcentric_blk64mib_5trees_pcore_2026-06-10

# Shape: wide-short 30k x 400k (compare vs control_cm_5trees_30kx400k AP 5.09)
run dw1_30kx400k python3 benchmarks/profiling/parallel_chrono.py \
  --rows=30000 --cols=400000 --num_trees=5 --num_threads=1 \
  --feature_split_type=Oblique --depthwise_1_pass \
  --numerical_split_type="Dynamic Random Histogram" \
  --experiment_name=dw1_colcentric_5trees_30kx400k_pcore_2026-06-10

# Shape: tall-narrow HIGGS-like 11M x 28 — dw1 new vs old-equivalent nodewise.
run dw1_11mx28 python3 benchmarks/profiling/parallel_chrono.py \
  --rows=11000000 --cols=28 --num_trees=5 --num_threads=1 \
  --feature_split_type=Oblique --depthwise_1_pass \
  --numerical_split_type="Dynamic Random Histogram" \
  --experiment_name=dw1_colcentric_5trees_11mx28_pcore_2026-06-10

run control_11mx28 python3 benchmarks/profiling/parallel_chrono.py \
  --rows=11000000 --cols=28 --num_trees=5 --num_threads=1 \
  --feature_split_type=Oblique \
  --numerical_split_type="Dynamic Random Histogram" \
  --experiment_name=control_cm_5trees_11mx28_pcore_2026-06-10

echo "=== dw1 queue done $(date) ===" >> "$LOG"
