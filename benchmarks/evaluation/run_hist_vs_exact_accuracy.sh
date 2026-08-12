#!/usr/bin/env bash
set -euo pipefail

# Sweep driver for the paper's split-finder accuracy study:
# Random Histogramming vs Exact, for SPO-RF (Bagging) and SPO-GBT (Boosting).
#
# Arms per learner:
#   exact  Exact split finding (baseline)
#   rand   Random histogram, 64 bins, at EVERY node (worst case for accuracy;
#          upper-bounds the penalty of any exact/histogram hybrid)
#   dyn    Dynamic Random Histogram, 64 bins, threshold 250 (the shipped
#          default configuration used in the runtime evaluation)
#
# Seeds: outer loop, so all seed-1 arms (the primary analysis) finish first;
# seeds 2..N exist to measure the seed-noise yardstick. Each arm x seed is one
# accuracy.sh invocation (10-fold CV over all CC18 binary tasks) and lands in
# benchmarks/results/accuracy_hve_<learner>_<arm>_s<seed>{,_auc,_logloss}.csv.
# Finished sweeps are skipped, so the driver is resumable.
#
# Analysis: benchmarks/utils/accuracy_stats.py over the produced CSVs.

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

SEEDS=(${SEEDS_OVERRIDE:-1 2 3})
LEARNERS=(rf gbt)
ARMS=(exact rand dyn)

for seed in "${SEEDS[@]}"; do
  for learner in "${LEARNERS[@]}"; do
    ens_args=""
    [[ "$learner" == gbt ]] && ens_args="--ensemble_method Boosting"
    for arm in "${ARMS[@]}"; do
      case "$arm" in
        exact) split_args='--numerical_split_type "Exact"' ;;
        rand)  split_args='--numerical_split_type "Random" --histogram_num_bins=64' ;;
        dyn)   split_args='--numerical_split_type "Dynamic Random Histogram" --histogram_num_bins=64 --dynamic_split_threshold=250' ;;
      esac
      suffix="hve_${learner}_${arm}_s${seed}"
      if [[ -f "benchmarks/results/accuracy_${suffix}.csv" ]]; then
        echo "=== SKIP $suffix (CSV exists)"
        continue
      fi
      echo "=== RUN $suffix"
      EXTRA_TRAIN_ARGS="$split_args $ens_args --seed=$seed" \
        bash benchmarks/evaluation/accuracy.sh "$suffix"
    done
  done
done
echo "=== Sweep complete."
