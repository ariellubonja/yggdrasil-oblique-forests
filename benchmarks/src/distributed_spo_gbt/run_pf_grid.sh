#!/usr/bin/env bash
# Paper Table 3 rerun: P=F, 10 trees, n2-highmem-16, AA + SPO (main@47fe64fa = chunk jobs + VQSort), barriers on.
# W<=4 only: 96-vCPU N2 quota (W=4 cell = 80 vCPUs). Builds the train image first if the tag is missing.
set -euo pipefail
cd "$(dirname "$0")"
export PATH=$HOME/google-cloud-sdk/bin:$PATH
RUN_ID=${RUN_ID:-pfgrid20261025}
IMAGE=us-east1-docker.pkg.dev/ydf-distributed-spo-gbt/ydf-repo/train-ydf:47fe64fa-bdad5080
if ! gcloud artifacts docker tags list "${IMAGE%:*}" --format="value(tag)" 2>/dev/null | grep -q "/${IMAGE##*:}$"; then
  echo "$(date -u +%FT%TZ) building $IMAGE"
  gcloud builds submit --tag "$IMAGE" vertex_build/ > vertex_build/train_build_47fe64fa.log 2>&1
  echo "$(date -u +%FT%TZ) build done"
fi
for d in "susy 18" "higgs 28" "epsilon 2000" "youtube8m 1152"; do set -- $d
  "$HOME/vertex_venv/bin/python" -u run_vertex_barriers.py --dataset "$1" --workers "${WORKERS:-1,2,4}" --arms aa,spo \
    --trees 10 --p_total "$2" --image "$IMAGE" --run_id "$RUN_ID"
done
echo "$(date -u +%FT%TZ) grid finished"
