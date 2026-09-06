#!/usr/bin/env bash
# Copy analyze.py outputs into the Overleaf snapshot under the names/labels that
# spo_vs_gbt_results.tex and main.tex reference. Usage:
#   bash export_to_paper.sh <analysis_dir> [<overleaf_snapshot_dir>]
set -euo pipefail
A="${1:?analysis dir (output of analyze.py --out-dir)}"
P="${2:-/home/ubuntu/yggdrasil-oblique-forests/benchmarks/results/overleaf-sept-3-26}"
F="$P/figures/results"
mkdir -p "$F"

# tables: rename file + label (never hand-edit the generated bodies)
sed 's/\\label{tab:main-summary-all}/\\label{tab:spo-vs-gbt-summary}/' \
    "$A/tables/table_main_summary_all_datasets.tex" > "$P/table_spo_vs_gbt_summary.tex"
sed 's/\\label{tab:huge-datasets}/\\label{tab:spo-vs-gbt-huge}/' \
    "$A/tables/table_huge_datasets.tex" > "$P/table_spo_vs_gbt_huge.tex"
sed 's/\\label{tab:per-dataset-appendix}/\\label{tab:spo-vs-gbt-per-dataset}/' \
    "$A/tables/table_per_dataset_appendix.tex" > "$P/table_spo_vs_gbt_per_dataset.tex"

# figures
declare -A MAP=(
  [fig_huge_pareto_rf]=spo_vs_gbt_pareto_rf
  [fig_huge_pareto]=spo_vs_gbt_pareto_all
  [fig_cd_rf_auc]=spo_vs_gbt_cd_rf_auc
  [fig_cd_gbt_auc]=spo_vs_gbt_cd_gbt_auc
  [fig_speedup_vs_depth]=spo_vs_gbt_speedup_vs_depth
  [fig_trunk_width]=spo_vs_gbt_trunk_width
  [fig_speedup_vs_min_examples]=spo_vs_gbt_speedup_vs_min_examples
  [fig_gbt_depth]=spo_vs_gbt_gbt_depth
)
for src in "${!MAP[@]}"; do
  cp -f "$A/figures/$src.pdf" "$F/${MAP[$src]}.pdf"
done
echo "exported to $P:"
ls -1 "$P"/table_spo_vs_gbt_*.tex "$F"/spo_vs_gbt_*.pdf
