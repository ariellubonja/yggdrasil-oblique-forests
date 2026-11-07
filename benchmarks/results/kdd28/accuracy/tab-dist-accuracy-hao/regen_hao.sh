#!/bin/bash
# kdd28_hao 5-seed accuracy bodies (sec:accuracy-hao). Run from the repo root. Figures first, tables last (same file names).
set -e
A=benchmarks/results/kdd28/accuracy; G=benchmarks/src/distributed_spo_gbt
python3 $A/tab-dist-accuracy-hao/assemble_hao.py
for v in "" _wide; do
  D=$A/tab-dist-accuracy-hao${v/_/-}
  if [ -z "$v" ]; then S=kdd28_tabarena33; SH=(); NOTE="SPAA paper eval"; else S=kdd28_wide_union; SH=(--dataset_shapes $A/tab-dist-accuracy-ydf-wide/wide_shapes.csv); NOTE="wide union (F>=100)"; fi
  COMMON=(--accuracy_csv $D/distacc/results.csv --arms distacc_single_spo,distacc_single_aa,distacc_spo,distacc_aa
    --profile kdd28_hao --suffix _hao$v --purple --extra aaspo=$D/distacc_aaspo/results.csv --extra_first --seed_mean
    --eval_suite $S "${SH[@]}" --accuracy_note "IN-PROCESS reruns, kdd28_hao, mean over seeds 1-5, $NOTE")
  python3 $G/make_paper_tables.py "${COMMON[@]}"
  # SPAA: 33 rows only fit the page with one-line rows (std as subscript, shape inline)
  if [ -z "$v" ]; then TF=(--group_header); else TF=(--std_below --shape_below --group_header); fi
  python3 $G/make_paper_tables.py "${COMMON[@]}" "${TF[@]}" --time_row --no_figure
  python3 $G/make_column_replication_fig.py --csv $D/colrep_distacc/results.csv --suffix _hao$v --metric test_auc \
    --dist_arm colrep_distacc_spo --single_csv $D/distacc/results.csv --single_filter arm=single:SPO --single_arms distacc_single_spo \
    --r0_csv $D/distacc/results.csv --r0_filter arm=4:SPO --r0_arm distacc_spo --profile kdd28_hao --eval_suite $S
done
