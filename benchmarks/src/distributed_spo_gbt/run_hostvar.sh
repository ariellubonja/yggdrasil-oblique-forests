#!/usr/bin/env bash
# VM-to-VM variance check (2026-10-26): HIGGS W=4 on n2-highmem-16, AA + SPO, 3 fresh clusters each
# (run_ids hostvar{1,2,3}20261026). train.py logs the host CPU model into cpu_<role>.json.
set -uo pipefail
cd "$(dirname "$0")"
export PATH=$HOME/google-cloud-sdk/bin:$PATH
PY=$HOME/vertex_venv/bin/python
IMAGE=us-east1-docker.pkg.dev/ydf-distributed-spo-gbt/ydf-repo/train-ydf:47fe64fa-345cdf22
LOG=$HOME/yggdrasil-oblique-forests/ariel_notes/results/2026-10-26_aa_threading
until gcloud artifacts docker tags list "${IMAGE%:*}" --format="value(tag)" 2>/dev/null | grep -qE "(^|/)${IMAGE##*:}$"; do
  echo "$(date -u +%FT%TZ) waiting for $IMAGE"; sleep 30
done
echo "$(date -u +%FT%TZ) image ready, launching"
for i in 1 2 3; do
  setsid nohup $PY -u run_vertex_barriers.py --dataset higgs --workers 4 --arms aa,spo --trees 10 --p_total 28 \
    --image $IMAGE --run_id hostvar${i}20261026 > $LOG/hostvar${i}20261026.log 2>&1 < /dev/null &
  sleep 20
done
echo "$(date -u +%FT%TZ) 3 launchers started (6 cells)"
wait
echo "$(date -u +%FT%TZ) all launchers finished"
