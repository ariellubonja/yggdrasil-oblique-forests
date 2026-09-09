#!/usr/bin/env bash
# Copy analyze.py outputs into the Overleaf snapshot under the names/labels that
# spo_vs_gbt_results.tex and main.tex reference. Usage:
#   bash export_to_paper.sh <analysis_dir> [<overleaf_snapshot_dir>]
set -euo pipefail
A="${1:?analysis dir (output of analyze.py --out-dir)}"
P="${2:-/home/ubuntu/yggdrasil-oblique-forests/benchmarks/results/overleaf-spaa27}"
F="$P/figures/results"
mkdir -p "$F"

# tables: rename file + label/ref (never hand-edit the generated bodies). The
# tables cross-reference each other, so rewrite \ref{} as well as \label{}.
RELABEL='s/{tab:main-summary-all}/{tab:spo-vs-gbt-summary}/g;
         s/{tab:huge-datasets}/{tab:spo-vs-gbt-huge}/g;
         s/{tab:timing}/{tab:spo-vs-gbt-timing}/g;
         s/{tab:per-dataset-appendix/{tab:spo-vs-gbt-per-dataset/g'
sed "$RELABEL" "$A/tables/table_main_summary_all_datasets_both.tex" \
    > "$P/table_spo_vs_gbt_summary_both.tex"
sed "$RELABEL" "$A/tables/table_huge_datasets_both.tex" \
    > "$P/table_spo_vs_gbt_huge_both.tex"
sed "$RELABEL" "$A/tables/table_timing_both.tex" \
    > "$P/table_spo_vs_gbt_timing_both.tex"
sed "$RELABEL" "$A/tables/table_per_dataset_appendix_both.tex" \
    > "$P/table_spo_vs_gbt_per_dataset_both.tex"

# figures
declare -A MAP=(
  [fig_huge_pareto_rf]=spo_vs_gbt_pareto_rf
  [fig_huge_pareto]=spo_vs_gbt_pareto_all
  [fig_cd_rf_auc]=spo_vs_gbt_cd_rf_auc
  [fig_cd_gbt_auc]=spo_vs_gbt_cd_gbt_auc
  [fig_speedup_vs_depth]=spo_vs_gbt_speedup_vs_depth
  [fig_trunk_width]=spo_vs_gbt_trunk_width
  [fig_gbt_depth]=spo_vs_gbt_gbt_depth
  [fig_rowcol_map]=spo_vs_gbt_rowcol_map
  [fig_rowcol_map_stdsort]=spo_vs_gbt_rowcol_map_stdsort
)
for src in "${!MAP[@]}"; do
  cp -f "$A/figures/$src.pdf" "$F/${MAP[$src]}.pdf"
done
echo "exported to $P:"
ls -1 "$P"/table_spo_vs_gbt_*.tex "$F"/spo_vs_gbt_*.pdf
