#!/bin/bash
# kdd28_hao vs kdd28_ydf 5-seed comparison bodies (sec:accuracy-hao-vs-ydf-params). Run from the repo root.
set -e
R=benchmarks/results/kdd28/runs; G=benchmarks/src/distributed_spo_gbt
python3 $R/2026-11-06_hao_seeds/assemble_hao_vs_ydf_params.py
# tab:accuracy-hao-vs-ydf-params-{single-machine,distributed}: kdd28_ydf seeds 1-5 vs kdd28_hao seeds 1-5
python3 $G/paper/tables/make_hao_vs_ydf_compare_tables.py --eval_suite kdd28_tabarena33
