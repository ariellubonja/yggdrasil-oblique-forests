#!/bin/bash
# Off-paper diagnostic (not a screen arm): AA on the k=1 trunks with only the first C columns kept (f0 = all signal), seed 1.
# Tests whether SPO beating AA on k=1 comes from AA overfitting noise columns. AA argv identical to the screen cells.
B=/home/ubuntu/results/2026-11-23_trunk_shape_and_generator_screen_spo_vs_aa_hao; O=$B/offpaper_k1_aa_fewer_candidate_columns
echo dataset,fold,cols_kept,test_auc > $O/results.csv
for d in trunk_shift_n4000_f1024_k1_auc75_rho0 trunk_shift_n4000_f1024_k1_auc90_rho0; do for c in 32 128; do for k in 0 1 2 3 4; do
  ( D=$O/data/${d}_c$c; mkdir -p $D
    for s in train test; do cut -d, -f1-$c,1025 $B/data/folds/$d/fold${k}_$s.csv > $D/fold${k}_$s.csv; done
    L=$O/${d}__${k}__aa_cols$c.log
    $B/bin/train_oblique_forest --input_mode csv --train_csv $D/fold${k}_train.csv --test_csv $D/fold${k}_test.csv --label_col label --ensemble_method Boosting --ydf_defaults "--feature_split_type=Axis Aligned" --validation_set_ratio 0 --early_stopping NONE --num_threads 4 --seed 1 > $L 2>&1
    echo "$d,$k,$c,$(grep -m1 "auc:" $L | awk "{print \$2}")" >> $O/results.csv ) &
done; done; done; wait; echo OFFPAPER_EXIT
