#!/bin/bash
# 2026-11-21 SPO normalization probe on mixed categorical+numeric sets (off-paper). A = original folds, B = train-std
# scaled. SPO-GBT kdd28_ydf 4 threads, SPO-RF harness defaults 8 threads, seed 1. RF cells time out after 3 h (wall).
R=/home/ubuntu/results/2026-11-21_spo_normalization_none_vs_trainstd_mixed_categorical_numeric_tabarena
cd $R/code && mkdir -p $R/logs $R/rf/logs || exit 1
TA=bank-marketing,Bank_Customer_Churn,churn,coil2000_insurance_policies,credit-g,credit_card_clients_default,customer_satisfaction_in_airline,E-CommereShippingData,Fitness_Club,Is-this-a-good-customer,kddcup09_appetency,Marketing_Campaign,online_shoppers_intention,qsar-biodeg,seismic-bumps
TR=homesite-insurance,homecredit-default
F=/home/ubuntu/spo_vs_gbt/folds
python3 $R/mismatch.py $F $R/column_types.json $R/mismatch.csv \
  $TA,$TR,Diabetes130US,HR_Analytics_Job_Change_of_Data_Scientists,in_vehicle_coupon_recommendation > $R/logs/mismatch.log 2>&1 || exit 1
python3 $R/scale_folds.py $F $R/data benchmarks/data/dataset_properties/kdd28_dataset_categoricals.csv $TA,$TR \
  > $R/logs/scale.log 2>&1 || exit 1
D=benchmarks/src/distributed_spo_gbt/accuracy_multi_thread/run_kdd28_ydf_ec2.py
B=/home/ubuntu/results/2026-11-06_hao_seeds/bin
C="--profile kdd28_ydf --tables distacc --dist_axes= --single_axes SPO --single_threads 4 --seed 1
   --bin_dir $B --git $(cat $B/GIT) --machine c8i.96xlarge --cpus 160"
for arm in orig scaled; do
  python3 $D --out $R/gbt_$arm --tab_folds $R/data/$arm $C \
    --off_paper "SPO normalization probe: NONE vs train-std scaling" > $R/logs/gbt_$arm.log 2>&1 &
done
for arm in orig scaled; do
  $B/train_oblique_forest --input_mode csv --train_csv x --test_csv y --label_col label --feature_split_type=Oblique \
    --num_threads 8 --seed 1 --print_params_only > $R/rf/logs/preflight_$arm.log 2>&1
done
diff <(grep -o 'effective-params.*' $R/rf/logs/preflight_orig.log) <(grep -o 'effective-params.*' $R/rf/logs/preflight_scaled.log) \
  || { echo RF_PREFLIGHT_DIFF >> $R/logs/queue.log; exit 2; }
for d in ${TA//,/ } ${TR//,/ }; do  # TabArena first, then TabReD
  for k in $(python3 -c "import json;print(*range(len(json.load(open('$F/$d/folds.json'))['indices'])))"); do
    for arm in orig scaled; do echo "$d $k $arm"; done
  done
done > $R/rf/cells.txt
xargs -P 24 -L 1 bash -c 'f='$R'/data/$2/$0; timeout 10800 '$B'/train_oblique_forest --input_mode csv \
  --train_csv $f/fold$1_train.csv --test_csv $f/fold$1_test.csv --label_col label --feature_split_type=Oblique \
  --num_threads 8 --seed 1 > '$R'/rf/logs/$0__f$1__$2.log 2>&1; echo "$0 $1 $2 rc=$? $(date -u +%T)" >> '$R'/rf/logs/done.txt' \
  < $R/rf/cells.txt &
wait
echo "ALL_EXIT $(date -u +%FT%TZ)" >> $R/logs/queue.log
