#!/usr/bin/env bash
# driver.sh <accuracy|runtime> <A|B>  -- one arm, one runner pass, in the upstream-bench worktree.
# A = upstream-bench (no configs); B = upstream-bench-gbt-hd + --config=skip_dead_axis_jobs.
set -uo pipefail
KIND="$1"; ARM="$2"
WT=/private/tmp/claude-501/-Users-ariellubonja-prog-randals-lab-oblique-forests-yggdrasil-decision-forests/2c2bce7b-145d-4629-8ef6-fa6546706cb1/scratchpad/upstream-bench
OUT=/private/tmp/claude-501/-Users-ariellubonja-prog-randals-lab-oblique-forests-yggdrasil-decision-forests/2c2bce7b-145d-4629-8ef6-fa6546706cb1/scratchpad/verify-speedup-workspace/iteration-2/eval-4-gbt-high-dim-vs-upstream/with_skill/outputs
cd "$WT" || exit 2
if [[ "$ARM" == A ]]; then BRANCH=upstream-bench; unset EXTRA_BAZEL_CONFIGS; SUF=upb_gbt_hd_a_exact_t30
else BRANCH=upstream-bench-gbt-hd; export EXTRA_BAZEL_CONFIGS="--config=skip_dead_axis_jobs"; SUF=upb_gbt_hd_b_exact_t30; fi
export NUM_TREES_DIVISOR=10
export EXTRA_TRAIN_ARGS='--ensemble_method Boosting --numerical_split_type "Exact"'
if pgrep -f 'bazel-bin/examples/train_oblique_forest' >/dev/null || pgrep -f 'evaluation/(runtime|accuracy).sh' >/dev/null; then
  echo "REFUSING TO START: another benchmark process is running"; exit 3; fi
git checkout -q "$BRANCH" || exit 2
echo "START $(date -u +%FT%TZ) kind=$KIND arm=$ARM branch=$(git branch --show-current) HEAD=$(git rev-parse --short HEAD) configs=${EXTRA_BAZEL_CONFIGS:-<none>} args=$EXTRA_TRAIN_ARGS divisor=$NUM_TREES_DIVISOR"
t0=$(date +%s)
if [[ "$KIND" == accuracy ]]; then
  bash benchmarks/evaluation/accuracy.sh "$SUF" </dev/null; rc=$?
  cp -f bazel-bin/examples/train_oblique_forest "$OUT/bin_$ARM" && shasum -a 256 "$OUT/bin_$ARM"
  for f in benchmarks/results/accuracy_${SUF}*.csv benchmarks/results/accuracy_${SUF}.log; do [[ -f "$f" ]] && cp -f "$f" "$OUT/"; done
else
  bash benchmarks/evaluation/runtime.sh --runs=1 "${SUF}_r1" </dev/null; rc=$?
  shasum -a 256 bazel-bin/examples/train_oblique_forest "$OUT/bin_$ARM"
  for f in benchmarks/results/${SUF}_r1.csv benchmarks/results/${SUF}_r1.log; do [[ -f "$f" ]] && cp -f "$f" "$OUT/"; done
fi
echo "$KIND.sh exit=$rc END $(date -u +%FT%TZ) wall=$(( $(date +%s) - t0 )) s"
