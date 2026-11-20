#!/bin/bash
# kdd28_hao 5-seed accuracy bodies (sec:accuracy-hao-vs-ydf-params). Run from the repo root. Figures first, tables last (same file names).
set -e
A=benchmarks/results/kdd28/accuracy; G=benchmarks/src/distributed_spo_gbt
D=$A/tab-accuracy-hao-vs-ydf-params; S=kdd28_tabarena33
python3 $D/assemble_hao_vs_ydf_params.py
COMMON=(--accuracy_csv $D/distacc/results.csv --arms distacc_single_spo,distacc_single_aa,distacc_spo,distacc_aa
  --profile kdd28_hao --suffix _hao_vs_ydf_params --purple --extra aaspo=$D/distacc_aaspo/results.csv --extra_first --seed_mean
  --eval_suite $S --accuracy_note "IN-PROCESS reruns, kdd28_hao, mean over seeds 1-5, SPAA paper eval")
python3 $G/paper/tables/make_paper_tables.py "${COMMON[@]}"
# 33 rows only fit the page with one-line rows (std as subscript, shape inline)
python3 $G/paper/tables/make_paper_tables.py "${COMMON[@]}" --group_header --time_row --no_figure
python3 $G/paper/figures/make_column_replication_fig.py --csv $D/colrep_distacc/results.csv --suffix _hao_vs_ydf_params --metric test_auc \
  --dist_arm colrep_distacc_spo --single_csv $D/distacc/results.csv --single_filter arm=single:SPO --single_arms distacc_single_spo \
  --r0_csv $D/distacc/results.csv --r0_filter arm=4:SPO --r0_arm distacc_spo --profile kdd28_hao --eval_suite $S
