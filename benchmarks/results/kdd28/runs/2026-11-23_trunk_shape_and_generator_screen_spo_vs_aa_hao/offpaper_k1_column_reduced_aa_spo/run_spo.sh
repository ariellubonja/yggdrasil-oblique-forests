#!/bin/bash
# Off-paper: SPO (kdd28_hao argv) on the same column-reduced k=1 copies as run.sh, seed 1, all folds.
B=/home/ubuntu/results/2026-11-23_trunk_shape_and_generator_screen_spo_vs_aa_hao; O=$B/offpaper_k1_aa_fewer_candidate_columns
for d in trunk_shift_n4000_f1024_k1_auc75_rho0 trunk_shift_n4000_f1024_k1_auc90_rho0; do for c in 32 128; do for k in 0 1 2 3 4; do
  D=$O/data/${d}_c$c
  $B/bin/train_oblique_forest --input_mode csv --train_csv $D/fold${k}_train.csv --test_csv $D/fold${k}_test.csv --label_col label --ensemble_method Boosting --feature_split_type=Oblique --num_trees 300 --numerical_split_type=Exact --num_projections_exponent 1 --max_num_projections 1000 --projection_density_factor 1.5 --validation_set_ratio 0 --early_stopping NONE --num_threads 4 --seed 1 > $O/${d}__${k}__spo_cols$c.log 2>&1 &
done; done; done; wait; echo SPO_EXIT
