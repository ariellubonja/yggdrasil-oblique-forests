#!/bin/bash
# Off-paper: ct_slices with GroupKFold(5) by patientId, seed 1, SPO (kdd28_hao) vs AA (kdd28_ydf). Tests whether the edge needs patient leakage.
B=/home/ubuntu/results/2026-11-22_wide_real_dataset_candidates_spo_vs_aa_screen_hao; cd $B/code
D=benchmarks/src/distributed_spo_gbt/accuracy_multi_thread/run_kdd28_ydf_ec2.py
C="--tables reg --tab_folds $B/data/folds_reg_groupkfold --eval_suite kdd28_wide_real_candidates --off_paper ct_slices_GroupKFold_by_patientId_sensitivity --bin_dir /home/ubuntu/results/2026-11-06_hao_seeds/bin --single_threads 4 --git a6917d8a-dirty --machine c8i.metal-96xl-smt-turbo --seed 1 --cpus 20"
python3 $D --out $B/cells_offpaper_ct_groupkfold/kdd28_hao/seed1 --profile kdd28_hao --reg_arms SPO $C &
python3 $D --out $B/cells_offpaper_ct_groupkfold/kdd28_ydf/seed1 --profile kdd28_ydf --reg_arms AA $C &
wait; echo CTG_EXIT
