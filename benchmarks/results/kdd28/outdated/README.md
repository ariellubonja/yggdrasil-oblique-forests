# outdated/: KDD28 results that no paper float reads (moved here 2026-11-04)

Kept as history and as the "before" of the fix comparisons. One line each: what it is, why it is outdated.
`check_layout.py` requires every entry of this folder to be named here.

| Entry | Why outdated |
|---|---|
| `tab-cc18-aa-spo-ydf/` | kdd28_ydf CC18 arms (a), (d), (e), 1020 rows (2026-11-03 m7i): float dropped from main.tex (Overleaf 39e5431). Valid results, body `table_cc18_aa_spo_ydf_body.tex` still regenerates (with `--recorded` = `appendix-earlier-settings/tab-cc18-aa-spo/results.csv`) |
| `tab-column-replication-ydf-cc18/` | kdd28_ydf CC18 fold-0 column replication (1870 + 340 rows, 2026-11-03 m7i): superseded on 2026-11-04 by the TabArena `colrep_distacc` run in `accuracy/tab-column-replication-ydf/` |
| `fixed_reruns_2026-11-01_stages/` | intermediate fix stages of the 2026-11-01 reruns (bin fix only, per-worker budget, numeric typing, a repeat): superseded by the all-fixes runs |
| `fixed_reruns_2026-11-01_duplicates/` | `paper_inputs/table2/` (= table2_single10 with the 5-seed single row) and the seed-1 `table3_results.csv`: superseded copies |
| `paper_refs_pre_7c0eebd3/` | the published pre-fix Table 2/3 cells (bugs 7c0eebd3 discretization, dc1a8494 ties): the "before" of `fixed_reruns/compare_fixed_reruns.py` |
| `vertex_p_F_prefix_20261025/` | Vertex P = F grids on the pre-fix image (pfhm4_, pfgrid, pfhm4bar 20261025): replaced by `cells/vertex_p_F/pf*fix20261101_*` |
| `higgs_hm16_probes_20261026/` | HIGGS n2-highmem-16 probes (presort, CPU series, forced discretization, host variance), journal 2026-10-26: no float |
| `inprocess_m7i_grid_20261012/` | in-process m7i 50-tree grid, anchors and their driver logs: smoke tests only, runtime table dropped 2026-11-03 |
| `vertex_scaling_20261015/` | family B weak/strong scaling on n2-highmem-2: out of the paper since 2026-10-26 (`make_scaling_tables.py` still reads it) |
| `naive_column_shipping_20261029/` | the published pre-fix Table 3 Vertex run (merged from cross-worker-col-comms): replaced by `runtime/tab-cross-network/` |
| `runtime_distributed_spo_gbt_README_20261026.md` | the old `runtime/distributed_spo_gbt/README.md` file table: replaced by `benchmarks/results/kdd28/README.md` |
