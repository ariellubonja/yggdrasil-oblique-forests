#!/bin/bash
# Offline experiment queue 2026-06-10. Fully local: trunk synthetic data,
# warm bazel cache, NOPASSWD CPU staging via parallel_chrono.py. Safe to run
# with no internet. Each step is independent; failures don't stop the queue.
#
# Focus: dual_fp32 hybrid (precision-neutral CM/RM dispatch). Dual fp32 needs
# 2x4-byte stores, so shape is 1.5M x 4096 (24+24 GiB) on this 62 GiB box,
# with a matched fp32 control at the same shape and a dual_bf16 cross-point.

set -u
cd /home/ariel/prog/ydf/yggdrasil-oblique-forests
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1

LOG=benchmarks/results/offline_queue_2026-06-10.log
echo "=== queue start $(date) ===" >> "$LOG"

run() {
  local name="$1"; shift
  echo "--- $name start $(date) ---" >> "$LOG"
  "$@" >> "$LOG" 2>&1
  echo "--- $name exit=$? $(date) ---" >> "$LOG"
}

CHRONO="python3 benchmarks/profiling/parallel_chrono.py --rows=1500000 --cols=4096 --num_trees=5 --num_threads=1 --feature_split_type=Oblique"

# A. fp32 column-major control at 1.5M x 4096 (matched baseline)
run control_1500k $CHRONO --numerical_split_type="Dynamic Random Histogram" \
  --experiment_name=control_cm_5trees_1500k_pcore_2026-06-10

# B. dual_fp32 hybrid, threshold 20000 (the 3M optimum, untuned for 1.5M)
YDF_RM_MAX_ROWS=20000 run dualfp32_rm20k_1500k $CHRONO \
  --numerical_split_type="Dynamic Random Histogram" \
  --dataset_layout=dual_fp32 --projection_matrix_control \
  --experiment_name=dualfp32_rm20k_5trees_1500k_pcore_2026-06-10

# C. dual_bf16 at the same shape (cross-comparison: fp32-hybrid vs bf16-hybrid)
YDF_RM_MAX_ROWS=20000 run dualbf16_rm20k_1500k $CHRONO \
  --numerical_split_type="Dynamic Random Histogram" \
  --dataset_layout=dual_bf16 --projection_matrix_control \
  --experiment_name=dualbf16_rm20k_5trees_1500k_pcore_2026-06-10

# D. dual_fp32 threshold ablation
YDF_RM_MAX_ROWS=10000 run dualfp32_rm10k_1500k $CHRONO \
  --numerical_split_type="Dynamic Random Histogram" \
  --dataset_layout=dual_fp32 --projection_matrix_control \
  --experiment_name=dualfp32_rm10k_5trees_1500k_pcore_2026-06-10

YDF_RM_MAX_ROWS=40000 run dualfp32_rm40k_1500k $CHRONO \
  --numerical_split_type="Dynamic Random Histogram" \
  --dataset_layout=dual_fp32 --projection_matrix_control \
  --experiment_name=dualfp32_rm40k_5trees_1500k_pcore_2026-06-10

# E. shape generalization: wide-short trunk 30k x 400000 (12e9 values,
# dual stores 48 GiB bf16 / fp32 control ~48 GiB VD) — dual_bf16 only.
CHRONO_WIDE="python3 benchmarks/profiling/parallel_chrono.py --rows=30000 --cols=400000 --num_trees=5 --num_threads=1 --feature_split_type=Oblique"

run control_30kx400k $CHRONO_WIDE \
  --numerical_split_type="Dynamic Random Histogram" \
  --experiment_name=control_cm_5trees_30kx400k_pcore_2026-06-10

YDF_RM_MAX_ROWS=20000 run dualbf16_30kx400k $CHRONO_WIDE \
  --numerical_split_type="Dynamic Random Histogram" \
  --dataset_layout=dual_bf16 --projection_matrix_control \
  --experiment_name=dualbf16_rm20k_5trees_30kx400k_pcore_2026-06-10

echo "=== queue done $(date) ===" >> "$LOG"
