# KDD28 results, laid out like the paper (2026-11-04)

Paper: `paper/distributed_spo_gbt_kdd28/main.tex` (live copy: the Overleaf clone). One directory per float, named after its
LaTeX label (`tab:x` -> `tab-x`). A table and the figure drawn from it share the table's directory. Every float dir has a
`README.md` (labels, caption, status, profile and effective settings, eval suite, machine, date, code, the generator command
that rebuilds the body, raw-cell links) and an `inputs.txt` (the files the generator reads, one repo-relative path per line).
How to recreate a run: `benchmarks/src/distributed_spo_gbt/REPLICABILITY.md`.

| Paper section | Dir | Labels | Profile |
|---|---|---|---|
| Accuracy | `accuracy/tab-dist-accuracy-ydf/` | tab:dist-accuracy-ydf, fig:dist-accuracy-mean-ydf | kdd28_ydf.distacc_* |
| Accuracy | `accuracy/tab-column-replication-ydf/` | tab:column-replication-ydf, fig:column-replication-ydf | kdd28_ydf.colrep_distacc_spo |
| Runtime | `runtime/tab-vertex-hm4-pf-runtime/` | tab:vertex-hm4-pf-runtime, fig:vertex-hm4-hm16-pf-runtime | vertex_grid_pf |
| Runtime | `runtime/fig-yt8m-barriers/` | fig:yt8m-barriers | vertex_grid_pf |
| Runtime (cross-worker communication) | `runtime/tab-cross-network/` | tab:cross-network, fig:cross-network | table3 (old settings) |
| Appendix, earlier settings | `appendix-earlier-settings/tab-dist-accuracy/` | tab:dist-accuracy, fig:dist-accuracy-mean | table8 |
| Appendix, earlier settings | `appendix-earlier-settings/tab-cc18-aa-spo/` | tab:cc18-aa-spo | cc18_aa_spo |
| Appendix, earlier settings | `appendix-earlier-settings/tab-column-replication/` | tab:column-replication, fig:column-replication | table2 |
| Appendix, in-node parallelism | `appendix/tab-in-node-dist-vqsort-ablation/` | tab:in-node-dist-vqsort-ablation | vertex_grid_pf |
| Appendix, in-node parallelism | `appendix/tab-in-node-dist-vqsort-vertex/` | tab:in-node-dist-vqsort-vertex | in_node_threads |
| Appendix | `appendix/tab-setup-cost/` | tab:setup-cost | vertex_grid_pf |
| Appendix | `appendix/fig-cc18-tree-feature-hist/` | fig:cc18-tree-feature-hist | cc18_feature_study |

Other top-level dirs: `cells/` (raw Vertex cells shared by several floats), `analysis/` (evidence with no float),
`outdated/` (superseded or stale-bug results, one-line reason each in its README).

Rules:
1. A generator reads only the files its float's `inputs.txt` names (`make_pf_ablation_tables.py` enforces it, the others
   default to these paths). New results land in the float dir they feed, or in `cells/` for Vertex cells.
2. When a float is added, renamed or dropped in main.tex, move its dir (dropped -> `outdated/` with a reason line).
3. Check: `python3 benchmarks/src/distributed_spo_gbt/params/check_layout.py [main.tex]` (default: the repo snapshot).
   It is also run by `params.py selftest` and `check_argv_equivalence.sh`.

Diagram floats without data (`fig:ydf-dist-gbt-training-loop`, `fig:ydf-dist-gbt-load-balancing`) have no dir.
