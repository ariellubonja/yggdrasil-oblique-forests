#!/bin/bash
# After seed 1 (all 78) and the first seeds 2-5 queue end: recompute non-positive trunks, rerun seeds 2-5 (resumes done cells).
B=/home/ubuntu/results/2026-11-23_trunk_shape_and_generator_screen_spo_vs_aa_hao
while tmux has -t seeds 2>/dev/null || tmux has -t seed1big 2>/dev/null; do sleep 60; done
for f in $B/logs/screen_seed[1-5]*.log; do mv $f ${f%.log}_pass1.log; done
python3 $B/nonpositive.py $B
sed -i "s#FOLDS=\$B/data/folds_ready#FOLDS=\$B/data/folds#g" $B/run_seeds2to5.sh
$B/run_seeds2to5.sh
echo FOLLOWUP_EXIT $(date -u +%FT%TZ)
