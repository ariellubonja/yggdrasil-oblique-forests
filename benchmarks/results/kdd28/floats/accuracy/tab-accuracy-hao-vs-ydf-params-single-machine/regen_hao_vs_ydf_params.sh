#!/bin/bash
# kdd28_hao vs kdd28_ydf comparison bodies (sec:accuracy-hao-vs-ydf-params). Run from the repo root.
set -e
R=benchmarks/results/kdd28/runs; G=benchmarks/src/distributed_spo_gbt
python3 $R/2026-11-06_hao_seeds/assemble_hao_vs_ydf_params.py
# tab:accuracy-hao-vs-ydf-params-{single-machine,distributed}: SPAA block YDF seeds 1-5 vs Hao 1-5, wide block YDF seed 1 vs Hao 1-5
python3 $G/paper/tables/make_hao_vs_ydf_compare_tables.py --eval_suite kdd28_tabarena33 --wide_eval_suite kdd28_wide_datasets --wide_ydf_seeds 1 \
  --single_arraystretch 0.75  # rows at 0.75 height: the single-machine block fits its page (ht 608.5pt of 626pt)
