#!/bin/bash
# KDD28 Tables 2-3 with every fix, on Vertex AI (project ydf-distributed-spo-gbt, us-east1, n2-highmem-4, 1 manager + 4
# workers). Builds the two training images, then runs sequentially (the 96-vCPU N2 quota fits one sweep at a time):
#   table2/fullfix              column replication r = 0..1, 5 seeds, P = F per worker (the paper's setting)
#   table2/fullfix_splitbudget  same + --split_budget (cluster total P = F, = single-machine budget)
#   table3/fullfix              in-worker vs cross-network (branch fix-disc-colcomms), seed 1, 300 trees
# Needs gcloud (authenticated, project set) and a Python with google-cloud-aiplatform (PY).
# Usage: run_vertex_fixed.sh <out_root> [steps...]   steps: build table2 table2_split table3 (default: all)
set -u
OUT=${1:?out_root}; shift; STEPS=${*:-build table2 table2_split table3}
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd $HERE/../../../.. && pwd)
# Values: params/kdd28_params.json table2 (dist_per_worker, dist_split; trees/depth/W/seeds = the archived launcher's defaults)
# and table3.seed1. The launchers run from the recorded commits, so their CSVs carry no stamp columns.
eval "$(python3 $HERE/../params/params.py export table2.dist_split T2S_)"
eval "$(python3 $HERE/../params/params.py export table3.seed1 T3_)"
PY=${PY:-python3}
R=us-east1-docker.pkg.dev/ydf-distributed-spo-gbt/ydf-repo
# Recorded commits (images train-ydf:fullfix-10509539, colship-fullfix-1ad2062f already exist: skip `build`).
T2_COMMIT=${T2_COMMIT:-105095395eff07222842ab6f93ca7bf72741077e}
T3_COMMIT=${T3_COMMIT:-1ad2062f1e985d449b6574438e0e5c8113c08e68}
CACHE=${CACHE:-gs://ariel_ydf_bucket/work/cache_cc18_fixed_$(date -u +%Y%m%d)}
EXTRA="--cache_root $CACHE --no_checkpoints --reuse_reader"
mkdir -p $OUT/src
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $OUT/run_vertex_fixed.log; }

# Sources of the two harness versions (vertex_build/ + column_replication/), taken from their commits.
for c in t2:$T2_COMMIT t3:$T3_COMMIT; do
  n=${c%%:*}; sha=${c#*:}; d=$OUT/src/$n
  [ -d $d ] || { mkdir -p $d; git -C $REPO archive $sha benchmarks/src/distributed_spo_gbt/vertex_build \
      benchmarks/src/distributed_spo_gbt/column_replication | tar -x -C $d --strip-components 3; }
done

build() { # src_dir commit image_tag
  local d=$1 sha=$2 tag=$3
  sed -i.bak "s/^ARG YDF_COMMIT=.*/ARG YDF_COMMIT=$sha/" $d/vertex_build/wheel/Dockerfile
  (cd $d/vertex_build &&
   gcloud builds submit wheel --tag $R/ydf-wheel:${sha:0:8} --machine-type=e2-highcpu-32 --timeout=3h > ../wheel_build.log 2>&1 &&
   gcloud builds submit . --config cloudbuild.yaml --substitutions _WHEEL_TAG=${sha:0:8},_IMAGE=$R/train-ydf:$tag \
     > ../train_build.log 2>&1)
}
sweep() { # src_dir image commit run_id prefix out extra
  mkdir -p $6
  (cd $1 && $PY column_replication/run_vertex_colrep.py --image $2 --git ${3:0:8} --run_id $4 --job_prefix $5 \
     --out_root $6 --extra_args "$7" > $6/launcher.log 2>&1)
}
RUN=$(date -u +%Y%m%dT%H%MZ)
for s in $STEPS; do case $s in
  build)
    log "build images"; build $OUT/src/t2 $T2_COMMIT fullfix-${T2_COMMIT:0:8} & build $OUT/src/t3 $T3_COMMIT colship-fullfix-${T3_COMMIT:0:8} & wait ;;
  table2)
    log "table2"; sweep $OUT/src/t2 $R/train-ydf:fullfix-${T2_COMMIT:0:8} $T2_COMMIT $RUN colrep-full $OUT/table2/fullfix "$EXTRA" ;;
  table2_split)
    log "table2_split"; sweep $OUT/src/t2 $R/train-ydf:fullfix-${T2_COMMIT:0:8} $T2_COMMIT $RUN colrep-fullsplit \
      $OUT/table2/fullfix_splitbudget "$EXTRA $([ "$T2S_SPLIT_BUDGET" = true ] && echo --split_budget)" ;;
  table3)
    # Datasets snake-split by size over 4 parallel jobs (the paper run's grouping), levels 0 (in-worker) and ship.
    log "table3"
    DS=(task_9977_nomao task_167120_numerai28.6 task_9910_Bioresponse task_167125_Internet-Advertisements task_9976_madelon task_219_electricity task_7592_adult task_14965_bank-marketing task_9978_ozone-level-8hr task_3904_jm1 task_43_spambase task_14952_PhishingWebsites task_167141_churn task_9952_phoneme task_3903_pc3 task_146820_wilt task_3902_pc4 task_3_kr-vs-kp task_9957_qsar-biodeg task_3917_kc1 task_9946_wdbc task_3918_pc1 task_146819_climate-model-simulation-crashes task_14954_cylinder-bands task_10093_banknote-authentication task_3913_kc2 task_31_credit-g task_9971_ilpd task_29_credit-approval task_37_diabetes task_125920_dresses-sales task_49_tic-tac-toe task_15_breast-w task_10101_blood-transfusion-service-center)
    G=("" "" "" "")
    for i in "${!DS[@]}"; do r=$((i/4)); c=$((i%4)); [ $((r%2)) = 1 ] && c=$((3-c)); G[$c]="${G[$c]:+${G[$c]},}${DS[$i]}"; done
    for g in 0 1 2 3; do
      mkdir -p $OUT/table3/fullfix/g$g
      (cd $OUT/src/t3 && $PY column_replication/run_vertex_colrep.py --image $R/train-ydf:colship-fullfix-${T3_COMMIT:0:8} \
        --git ${T3_COMMIT:0:8} --run_id $RUN --job_prefix colship-g$g --seeds $T3_SEEDS --levels 0,ship --datasets "${G[$g]}" \
        --machine $T3_MACHINE --workers $T3_WORKERS --num_trees $T3_TREES --max_depth $T3_MAX_DEPTH --out_root $OUT/table3/fullfix/g$g \
        --extra_args "$EXTRA --barrier_csv$([ "$T3_LOCAL_PROJECTION_COUNT" = true ] && echo " --local_projection_count")" > $OUT/table3/fullfix/g$g/launcher.log 2>&1) &
      sleep 25
    done
    wait ;;
  *) echo "unknown step $s"; exit 1 ;;
esac; done
log "done"
