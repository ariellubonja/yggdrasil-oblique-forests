#!/usr/bin/env bash
# Resume the Table-9 suite (run_dgbt_suite.sh skips finished cells) under the training lock.
set -uo pipefail
REPO=/home/ubuntu/yggdrasil-oblique-forests
exec 9>/home/ubuntu/spo_vs_gbt/run_all.lock; flock 9 || exit 1
echo "### $(date -u +%FT%TZ) SUITE RESUME (threads = max(6, 48/W))"
"$REPO/benchmarks/src/distributed_spo_gbt/run_dgbt_suite.sh"
echo "### QUEUE_SUITE_DONE $(date -u +%FT%TZ)"
