#!/usr/bin/env bash
# CC18 disjoint vs naive column shipping on Vertex: 4 parallel jobs (datasets snake-split by size).
set -u
cd /Users/ariellubonja/prog/randals_lab/oblique_forests/wt-colship-vertex/benchmarks/src/distributed_spo_gbt
OUT=/Users/ariellubonja/prog/randals_lab/oblique_forests/yggdrasil-decision-forests/ariel_notes/results/2026-10-29_colship_vertex
RUN=${RUN:-20261029T$(date -u +%H%M)Z}
DS=(task_9977_nomao task_167120_numerai28.6 task_9910_Bioresponse task_167125_Internet-Advertisements task_9976_madelon task_219_electricity task_7592_adult task_14965_bank-marketing task_9978_ozone-level-8hr task_3904_jm1 task_43_spambase task_14952_PhishingWebsites task_167141_churn task_9952_phoneme task_3903_pc3 task_146820_wilt task_3902_pc4 task_3_kr-vs-kp task_9957_qsar-biodeg task_3917_kc1 task_9946_wdbc task_3918_pc1 task_146819_climate-model-simulation-crashes task_14954_cylinder-bands task_10093_banknote-authentication task_3913_kc2 task_31_credit-g task_9971_ilpd task_29_credit-approval task_37_diabetes task_125920_dresses-sales task_49_tic-tac-toe task_15_breast-w task_10101_blood-transfusion-service-center)
G=("" "" "" "")
for i in "${!DS[@]}"; do r=$((i/4)); c=$((i%4)); [ $((r%2)) = 1 ] && c=$((3-c)); G[$c]="${G[$c]:+${G[$c]},}${DS[$i]}"; done
echo "RUN_ID $RUN"
for g in 0 1 2 3; do
  echo "g$g: ${G[$g]}"
  nohup ~/.venvs/distributed_spo_gbt/bin/python column_replication/run_vertex_colrep.py \
    --image us-east1-docker.pkg.dev/ydf-distributed-spo-gbt/ydf-repo/train-ydf:colship-04c24f88 --git 04c24f88 \
    --run_id $RUN --job_prefix colship-g$g --seeds 1 --levels 0,ship --datasets "${G[$g]}" \
    --machine n2-highmem-4 --workers 4 --num_trees 300 --max_depth 6 --out_root $OUT/g$g \
    --extra_args "--cache_root gs://ariel_ydf_bucket/work/cache_cc18 --no_checkpoints --reuse_reader --barrier_csv --local_projection_count" \
    > $OUT/launcher_g$g.log 2>&1 &
  sleep 25
done
