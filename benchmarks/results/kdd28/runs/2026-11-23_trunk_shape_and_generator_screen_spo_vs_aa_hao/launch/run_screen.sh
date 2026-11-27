#!/bin/bash
# Trunk shape x generator screen: single SPO (kdd28_hao profile) vs single AA (kdd28_ydf). Resumable.
# Usage: run_screen.sh <seed> [driver args, e.g. --only a,b]. Seed 1 on all, seeds 2-5 where the seed-1 effect > 0.
B=/home/ubuntu/results/2026-11-23_trunk_shape_and_generator_screen_spo_vs_aa_hao; cd ~/wt/trunk-screen; s=$1; shift
D=benchmarks/src/distributed_spo_gbt/accuracy_multi_thread/run_kdd28_ydf_ec2.py
C="--eval_suite kdd28_trunk_generator_screen --bin_dir $B/bin --tab_folds ${FOLDS:-$B/data/folds} --tables distacc_stamped --single_threads 4 --worker_threads 2 --dist_axes= --git d43e0809-dirty --machine c8i.metal-96xl-smt-turbo --seed $s"
python3 $D --out $B/cells/kdd28_hao/seed$s --profile kdd28_hao --single_axes SPO --cpus ${CPU:-180} $C "$@" &
python3 $D --out $B/cells/kdd28_ydf/seed$s --profile kdd28_ydf --single_axes AA --cpus ${CPU:-180} $C "$@" &
wait; echo "SCREEN_EXIT seed$s $(date -u +%FT%TZ)"
