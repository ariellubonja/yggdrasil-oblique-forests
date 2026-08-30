#!/usr/bin/env bash
# HIGGS-only A/B of accuracy.sh's exact build + flags, before/after the 4-commit OOB cherry-pick.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
OUT=benchmarks/results/oob_higgs_ab
BEFORE=1e5d48ef; AFTER=1a168b16
source benchmarks/utils/bench_common.sh
ensure_icx
bench_ecores --enable force   # accuracy.sh keeps e-cores on for build + run
BAZEL_FLAGS=(-c opt --cxxopt="-O3" --cxxopt="-march=native")
NUM_TREES=$(( $(nproc) * 5 ))
TRAIN=benchmarks/data/HIGGS_train_1000k.csv
TEST=benchmarks/data/HIGGS_test_20k.csv

build_arm() {  # $1=sha $2=name
  git checkout -q "$1"
  bazel build "${BAZEL_FLAGS[@]}" //examples:train_oblique_forest
  cp -L bazel-bin/examples/train_oblique_forest "$OUT/bin/train_$2"
}
build_arm "$AFTER" after
build_arm "$BEFORE" before
git checkout -q rebased-main
[[ "$(git rev-parse --short HEAD)" == "$AFTER" ]] || { echo "not back on rebased-main" >&2; exit 1; }

for rep in 1 2 3; do
  for arm in after before; do
    log="$OUT/higgs_${arm}_r${rep}.log"
    echo "ARM=$arm rep=$rep trees=$NUM_TREES train=$TRAIN test=$TEST" | tee "$log"
    "$OUT/bin/train_$arm" --input_mode csv --train_csv "$TRAIN" --test_csv "$TEST" \
      --label_col class --num_trees=$NUM_TREES --compute_oob_performances=true 2>&1 | tee -a "$log" | grep -E "Training block took|test-accuracy|Final OOB|wall-time"
  done
done
echo ALL_DONE
