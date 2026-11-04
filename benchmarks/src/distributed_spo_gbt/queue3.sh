#!/usr/bin/env bash
# not a paper source, parameters not governed by params/
# Pass 3 (after queue2): matched budget at 4 workers via the legacy per-worker LOCAL count: each worker samples
# ceil(F_local^e) from its ~F/4 columns, e chosen so the 4 together test ~ceil(sqrt F) per node (SUSY 4 vs 5,
# Epsilon 48 vs 45, HIGGS 8 vs 6, YouTube-8M 36 vs 34). Split budget (library default since 2026-11-02) pinned off.
set -uo pipefail
B=/home/ubuntu/yggdrasil-oblique-forests/benchmarks/src/distributed_spo_gbt
O=/home/ubuntu/yggdrasil-oblique-forests/benchmarks/results/runtime/distributed_spo_gbt/logs
LOCK=/home/ubuntu/spo_vs_gbt/run_all.lock
until grep -q "QUEUE2_DONE" "$O/queue2.out" 2>/dev/null; do sleep 120; done
for spec in "SUSY 0.0" "EPSILON 0.39" "HIGGS 0.3" "YOUTUBE8M 0.38"; do
  set -- $spec
  echo "### $(date -u +%FT%TZ) START $1 W=4 matched-budget exponent=$2"
  flock "$LOCK" "$B/run_dgbt.sh" "$1" 4 SPARSE_OBLIQUE 300 \
    --sparse_oblique_local_projection_count=true --sparse_oblique_split_budget=false \
    --sparse_oblique_num_projections_exponent="$2"
  echo "### $(date -u +%FT%TZ) END   $1 rc=$?"
done
echo "### QUEUE3_DONE $(date -u +%FT%TZ)"
