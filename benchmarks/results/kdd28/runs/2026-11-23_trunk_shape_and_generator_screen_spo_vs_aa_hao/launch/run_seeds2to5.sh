#!/bin/bash
# Seeds 2-5 on trunks with seed-1 effect > 0 (list = seed-1 rows minus seed1_nonpositive.txt), plus seed 1 on newly ready trunks.
B=/home/ubuntu/results/2026-11-23_trunk_shape_and_generator_screen_spo_vs_aa_hao
POS=$(python3 -c "import csv;neg=set(open(\"$B/seed1_nonpositive.txt\").read().split());print(\",\".join(sorted({r[\"dataset\"] for r in csv.DictReader(open(\"$B/cells/kdd28_hao/seed1/distacc_stamped/results.csv\"))}-neg)))")
FOLDS=$B/data/folds CPU=60 $B/run_screen.sh 1 > $B/logs/screen_seed1_ready2.log 2>&1 &
for s in 2 3 4 5; do FOLDS=$B/data/folds CPU=${CPU25:-30} $B/run_screen.sh $s --only $POS > $B/logs/screen_seed$s.log 2>&1 & done
wait; echo SEEDS_EXIT $(date -u +%FT%TZ)
