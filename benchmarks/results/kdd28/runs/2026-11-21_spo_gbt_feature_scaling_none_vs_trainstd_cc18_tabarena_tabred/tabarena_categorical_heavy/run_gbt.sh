#!/bin/bash
# 2026-11-21 SPO normalization probe (off-paper): single-machine GBT, profile kdd28_ydf, seed 1, 4 threads.
# Arm A = original folds (normalization NONE), arm B = train-std-scaled folds (emulated STANDARD_DEVIATION). AA = sanity.
R=/home/ubuntu/results/2026-11-21_spo_gbt_normalization_none_vs_trainstd_categorical_heavy_tabarena
cd $R/code && mkdir -p $R/logs || exit 1
DS=NATICUSdroid,Amazon_employee_access,in_vehicle_coupon_recommendation,HR_Analytics_Job_Change_of_Data_Scientists,Diabetes130US
python3 $R/scale_folds.py /home/ubuntu/spo_vs_gbt/folds $R/data \
  benchmarks/data/dataset_properties/kdd28_dataset_categoricals.csv $DS > $R/logs/scale.log 2>&1 || exit 1
D=benchmarks/src/distributed_spo_gbt/accuracy_multi_thread/run_kdd28_ydf_ec2.py
B=/home/ubuntu/results/2026-11-06_hao_seeds/bin
C="--profile kdd28_ydf --tables distacc --dist_axes= --single_axes SPO,AA --single_threads 4 --seed 1
   --bin_dir $B --git $(cat $B/GIT) --machine c8i.96xlarge --cpus 160"
for arm in orig scaled; do
  python3 $D --out $R/gbt_$arm --tab_folds $R/data/$arm $C \
    --off_paper "SPO normalization probe: NONE vs train-std scaling" > $R/logs/gbt_$arm.log 2>&1 &
done; wait
echo "ALL_EXIT $(date -u +%FT%TZ)" >> $R/logs/queue.log
