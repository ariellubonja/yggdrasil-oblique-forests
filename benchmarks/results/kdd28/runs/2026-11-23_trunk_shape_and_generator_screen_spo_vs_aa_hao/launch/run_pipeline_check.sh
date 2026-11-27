#!/bin/bash
# New binaries (origin/main d43e0809) vs recorded 2026-11-22 cells: madeline + jasmine, all folds, seed 1, SPO and AA.
B=/home/ubuntu/results/2026-11-23_trunk_shape_and_generator_screen_spo_vs_aa_hao; cd ~/wt/trunk-screen
O=/home/ubuntu/results/2026-11-22_wide_real_dataset_candidates_spo_vs_aa_screen_hao
D=benchmarks/src/distributed_spo_gbt/accuracy_multi_thread/run_kdd28_ydf_ec2.py
C="--eval_suite kdd28_wide_real_candidates --bin_dir $B/bin --tab_folds $O/data/folds_cls --tables distacc_stamped --single_threads 4 --worker_threads 2 --dist_axes= --git d43e0809-dirty --machine c8i.metal-96xl-smt-turbo --seed 1 --only madeline,jasmine"
python3 $D --out $B/pipeline_check/kdd28_hao --profile kdd28_hao --single_axes SPO --cpus 40 $C &
python3 $D --out $B/pipeline_check/kdd28_ydf --profile kdd28_ydf --single_axes AA --cpus 40 $C &
wait; echo PIPELINE_CHECK_EXIT
