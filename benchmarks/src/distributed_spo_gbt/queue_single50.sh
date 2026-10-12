#!/usr/bin/env bash
# After the 50-tree distributed grid: the non-distributed references at 50 trees (paper binary,
# harness defaults: all cores, binary weights, density 1.5, Exact finder) — SPO and axis-aligned.
set -uo pipefail
B=/home/ubuntu/yggdrasil-oblique-forests/benchmarks/src/distributed_spo_gbt
O=/home/ubuntu/yggdrasil-oblique-forests/benchmarks/results/runtime/distributed_spo_gbt/logs
until grep -q "GRID_DONE" "$O/t50_grid.out" 2>/dev/null; do sleep 60; done
for ds in HIGGS SUSY EPSILON YOUTUBE8M; do for sp in Oblique AxisAligned; do
  echo "### $(date -u +%FT%TZ) START single $ds $sp T=50"
  flock /home/ubuntu/spo_vs_gbt/run_all.lock "$B/run_single_anchor.sh" "$ds" 50 "$sp"
  echo "### $(date -u +%FT%TZ) END   single $ds $sp rc=$?"
done; done
echo "### SINGLE50_DONE $(date -u +%FT%TZ)"
