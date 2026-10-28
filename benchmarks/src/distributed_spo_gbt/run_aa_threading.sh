#!/usr/bin/env bash
# AA-vs-SPO cause study (2026-10-26): HIGGS on n2-highmem-16, 10 trees, P=F; three launchers share the
# 96-vCPU quota via the submit lock. Hypotheses + cell table: ariel_notes/results/2026-10-26_aa_threading/HYPOTHESES.md
set -uo pipefail
cd "$(dirname "$0")"
export PATH=$HOME/google-cloud-sdk/bin:$PATH
PY=$HOME/vertex_venv/bin/python
IMAGE=us-east1-docker.pkg.dev/ydf-distributed-spo-gbt/ydf-repo/train-ydf:47fe64fa-33362ed4
LOG=$HOME/yggdrasil-oblique-forests/ariel_notes/results/2026-10-26_aa_threading
mkdir -p "$LOG"
until gcloud artifacts docker tags list "${IMAGE%:*}" --format="value(tag)" 2>/dev/null | grep -qE "(^|/)${IMAGE##*:}$"; do
  echo "$(date -u +%FT%TZ) waiting for $IMAGE"; sleep 30
done
echo "$(date -u +%FT%TZ) image ready, launching"
setsid nohup $PY -u run_vertex_barriers.py --dataset higgs --workers 1,4 --arms aa,spo --trees 10 --p_total 28 \
  --image $IMAGE --run_id cpu20261026 > $LOG/cpu20261026.log 2>&1 < /dev/null &
sleep 20
setsid nohup $PY -u run_vertex_barriers.py --dataset higgs --workers 1,2,4 --arms aa --trees 10 --p_total 28 \
  --image $IMAGE --run_id aapresort20261026 "--extra_args=--max_unique_values_for_discretized_numerical=1" \
  > $LOG/aapresort20261026.log 2>&1 < /dev/null &
sleep 20
setsid nohup $PY -u run_vertex_barriers.py --dataset higgs --workers 1,4 --arms aa --trees 10 --p_total 28 \
  --image $IMAGE --run_id forcedisc20261026 "--extra_args=--force_numerical_discretization" \
  > $LOG/forcedisc20261026.log 2>&1 < /dev/null &
echo "$(date -u +%FT%TZ) 3 launchers started (9 cells)"
wait
echo "$(date -u +%FT%TZ) all launchers finished"
