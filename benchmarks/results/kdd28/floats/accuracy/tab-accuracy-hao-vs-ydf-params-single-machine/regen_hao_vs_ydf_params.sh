#!/bin/bash
# kdd28_hao vs kdd28_ydf comparison bodies (sec:accuracy-hao-vs-ydf-params). Run from the repo root.
set -e
R=benchmarks/results/kdd28/runs; G=benchmarks/src/distributed_spo_gbt
python3 $R/2026-11-06_hao_seeds/assemble_hao_vs_ydf_params.py
# tab:accuracy-hao-vs-ydf-params-{single-machine,distributed}: SPAA block YDF seeds 1-5 vs Hao 1-5, wide block YDF seed 1 vs Hao 1-5
# single-machine continuations (2026-11-20): CC18, multiclass (majority vs rest), regression (RMSE), both seeds 1-5. AA: YDF column only
# SPO clearly beats AA check marks (2026-11-21): floor + BH q=0.05 + positive in every seed, per profile
python3 benchmarks/results/kdd28/floats/accuracy/tab-accuracy-hao-vs-ydf-params-single-machine/select_spo_clearly_beats_aa.py
python3 $G/paper/tables/make_hao_vs_ydf_compare_tables.py --eval_suite kdd28_tabarena33 --wide_eval_suite kdd28_wide_datasets --wide_ydf_seeds 1 \
  --cc18_eval_suite kdd28_cc18_10fold --mc_eval_suite kdd28_tabarena_multiclass --reg_eval_suite kdd28_regression \
  --single_arraystretch 0.75  # rows at 0.75 height: the single-machine block fits its page (ht 608.5pt of 626pt)
