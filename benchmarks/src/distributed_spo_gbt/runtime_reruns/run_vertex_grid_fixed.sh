#!/usr/bin/env bash
# Paper runtime grid rerun on the bug-fixed image (main + bin/sort/fma fixes, 10509539): AA + SPO, P = F, 10 trees,
# n2-highmem-4 W=1..16 and n2-highmem-16 W=1..4. Barrier recorder is always on in this train.py (feeds the YT8M figures).
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/google-cloud-sdk/bin:$PATH
PY=$HOME/vertex_venv/bin/python
IMG=${IMG:-us-east1-docker.pkg.dev/ydf-distributed-spo-gbt/ydf-repo/train-ydf:fullfix-10509539}
HM4=${HM4:-pfhm4fix20261101}; HM16=${HM16:-pfgridfix20261101}
LOG=$HOME/dgbt_work/runtimefix; mkdir -p $LOG
for d in "susy 18" "higgs 28" "epsilon 2000" "youtube8m 1152"; do set -- $d
  setsid nohup $PY -u run_vertex_barriers.py --dataset $1 --workers 1,2,4 --arms aa,spo --trees 10 --p_total $2 \
    --machine n2-highmem-16 --image $IMG --run_id $HM16 > $LOG/${HM16}_$1.log 2>&1 < /dev/null &
  sleep 20
  setsid nohup $PY -u run_vertex_barriers.py --dataset $1 --workers 1,2,4,8,16 --arms aa,spo --trees 10 --p_total $2 \
    --machine n2-highmem-4 --image $IMG --run_id $HM4 > $LOG/${HM4}_$1.log 2>&1 < /dev/null &
  sleep 20
done
echo "$(date -u +%FT%TZ) launched 8 launchers"
wait
echo "$(date -u +%FT%TZ) grid finished"
