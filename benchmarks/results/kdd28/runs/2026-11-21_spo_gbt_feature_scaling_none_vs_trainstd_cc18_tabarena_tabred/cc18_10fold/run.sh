#!/bin/bash
# 2026-11-21 SPO normalization probe on CC18 (kdd28_cc18_10fold, off-paper). A = original folds, B = train-std scaled.
# SPO-GBT kdd28_ydf (cc18_spo arm) 4 threads, then SPO-RF harness defaults 8 threads, seed 1, 10 folds. RF timeout 3 h.
R=/home/ubuntu/results/2026-11-21_spo_normalization_none_vs_trainstd_cc18_10fold
F=/home/ubuntu/results/2026-11-03_ydf_defaults_rerun/data/cc18_10fold
B=/home/ubuntu/results/2026-11-06_hao_seeds/bin
cd $R/code && mkdir -p $R/logs $R/rf/logs || exit 1
q() { echo "$1 $(date -u +%FT%TZ)" >> $R/logs/queue.log; }
q START
(cd $F && sha256sum -c --quiet $R/code/benchmarks/src/distributed_spo_gbt/params/eval_suites/kdd28_cc18_10fold.sha256) \
  > $R/logs/sha_check.log 2>&1 || { q SHA_FAIL; exit 1; }
q SHA_OK_680
python3 $R/mismatch_cc18.py $F $R/mismatch.csv > $R/logs/mismatch.log 2>&1 || { q MISMATCH_FAIL; exit 1; }
python3 $R/scale_folds_cc18.py $F $R/data > $R/logs/scale.log 2>&1 || { q SCALE_FAIL; exit 1; }
q SCALED
D=$R/code/benchmarks/src/distributed_spo_gbt/accuracy_multi_thread/run_kdd28_ydf_ec2.py
C="--profile kdd28_ydf --tables cc18 --single_threads 4 --seed 1 --bin_dir $B --git $(cat $B/GIT)
   --machine c8i.metal-96xl --cpus 180"
for arm in orig scaled; do
  python3 $R/run_cc18_spo_only.py $D --out $R/gbt_$arm --cc18_10fold $R/data/$arm $C \
    --off_paper "SPO normalization probe: NONE vs train-std scaling" > $R/logs/gbt_$arm.log 2>&1 &
done
wait
q GBT_DONE
for arm in orig scaled; do
  $B/train_oblique_forest --input_mode csv --train_csv x --test_csv y --label_col label --feature_split_type=Oblique \
    --num_threads 8 --seed 1 --print_params_only > $R/rf/logs/preflight_$arm.log 2>&1
done
diff <(grep -o 'effective-params.*' $R/rf/logs/preflight_orig.log) <(grep -o 'effective-params.*' $R/rf/logs/preflight_scaled.log) \
  || { q RF_PREFLIGHT_DIFF; exit 2; }
for d in $(ls $R/data/orig); do
  l=$(python3 -c "import json;print(json.load(open('$F/$d/meta.json'))['label_col'])")
  for k in 0 1 2 3 4 5 6 7 8 9; do for arm in orig scaled; do echo "$d $k $arm $l"; done; done
done > $R/rf/cells.txt
xargs -P 45 -L 1 bash -c 'f='$R'/data/$2/$0; timeout 10800 '$B'/train_oblique_forest --input_mode csv \
  --train_csv $f/fold$1_train.csv --test_csv $f/fold$1_test.csv --label_col $3 --feature_split_type=Oblique \
  --num_threads 8 --seed 1 > '$R'/rf/logs/$0__f$1__$2.log 2>&1; echo "$0 $1 $2 rc=$? $(date -u +%T)" >> '$R'/rf/logs/done.txt' \
  < $R/rf/cells.txt
q ALL_EXIT
