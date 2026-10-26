#!/usr/bin/env bash
# After pfgrid20261025 (hm16, untimed) finishes with all 24 cells: n2-highmem-4 W=1..16 grid (untimed image)
# + W=16 barrier-timed twins (barrier image) for barrier breakdown and recorder overhead. One launcher per dataset/run.
set -uo pipefail
cd "$(dirname "$0")"
export PATH=$HOME/google-cloud-sdk/bin:$PATH
PY=$HOME/vertex_venv/bin/python
IMG_UNTIMED=us-east1-docker.pkg.dev/ydf-distributed-spo-gbt/ydf-repo/train-ydf:47fe64fa-bdad5080
IMG_BARRIER=us-east1-docker.pkg.dev/ydf-distributed-spo-gbt/ydf-repo/train-ydf:47fe64fa-1e756548
RES=../../results/runtime/distributed_spo_gbt/barriers/vertex
LOG=$HOME/dgbt_work
while pgrep -f "run_id pfgrid20261025" > /dev/null; do sleep 60; done
n=$(ls $RES/pfgrid20261025_*/*/eval.json 2>/dev/null | wc -l)
echo "$(date -u +%FT%TZ) pfgrid20261025 done: $n/24 cells"
if [ "$n" -ne 24 ]; then echo "NOT SMOOTH: $n/24 cells have eval.json; follow-ups not launched"; exit 1; fi
for d in "susy 18" "higgs 28" "epsilon 2000" "youtube8m 1152"; do set -- $d
  setsid nohup $PY -u run_vertex_barriers.py --dataset $1 --workers 1,2,4,8,16 --arms aa,spo --trees 10 --p_total $2 \
    --machine n2-highmem-4 --image $IMG_UNTIMED --run_id pfhm4_20261025 > $LOG/pfhm4_20261025_$1.log 2>&1 < /dev/null &
  setsid nohup $PY -u run_vertex_barriers.py --dataset $1 --workers 16 --arms aa,spo --trees 10 --p_total $2 \
    --machine n2-highmem-4 --image $IMG_BARRIER --run_id pfhm4bar20261025 > $LOG/pfhm4bar20261025_$1.log 2>&1 < /dev/null &
  sleep 20
done
echo "$(date -u +%FT%TZ) launched 8 follow-up launchers"
wait
echo "$(date -u +%FT%TZ) follow-ups finished"
