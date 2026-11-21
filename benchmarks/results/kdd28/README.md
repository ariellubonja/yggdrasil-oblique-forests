# KDD28 results: measurements in runs/, paper floats in floats/ (2026-11-20)

Paper: `paper/distributed_spo_gbt_kdd28/main.tex` (live copy: the Overleaf clone). How to recreate a run:
`benchmarks/src/distributed_spo_gbt/REPLICABILITY.md`.

- `runs/<YYYY-MM-DD>_<topic>/`: MEASUREMENT. Each experiment stored once, as the harness or run script wrote it, plus a
  `README.md` (profile, effective settings, eval suite, machine, date, code, raw-cell locations, which floats read it).
  `runs/vertex_p_F/` holds the raw Vertex AI cells of several grids (its README lists them).
- `floats/<section>/<float-label>/`: PRESENTATION. One dir per paper float, named after its LaTeX label (`tab:x` -> `tab-x`).
  A table and the figure drawn from it share the table's dir. Each has a `README.md` (labels, caption, status, the runs it
  reads, profile and settings, the generator command), an `inputs.txt` (the files the generator reads, one repo-relative
  path per line), the selection scripts that build its row subsets, and `derived/` (their output).
- `analysis/`: evidence with no float. `outdated/`: superseded or stale-bug results, one-line reason each in its README.

| Paper section | Float dir | Labels | Runs read (under `runs/`) |
|---|---|---|---|
| Accuracy, SPO-advantage datasets | `floats/accuracy/tab-accuracy-where-spo-outperforms-aa/` | tab:accuracy-where-spo-outperforms-aa, fig:accuracy-where-spo-outperforms-aa, fig:accuracy-where-spo-outperforms-aa-regression, fig:column-replication-where-spo-outperforms-aa, fig:column-replication-where-spo-outperforms-aa-regression | `2026-11-03_distacc`, `2026-11-04_distacc_aaspo`, `2026-11-04_colrep_distacc`, `2026-11-04_wide_eval`, table also `2026-11-06_hao_seeds`, `2026-11-20_ydf_seeds`, `2026-11-20_cc18_seeds`, `2026-11-20_regression_multiclass_tabarena_tabred`, `2026-11-21_where_spo_beats_aa_cc18_reg_distributed`, `2026-11-21_wide_eval_ydf_aa_seeds2to5` |
| Accuracy, Hao's settings, 5 seeds (appendix) | `floats/accuracy/tab-accuracy-hao-vs-ydf-params-single-machine/` | tab:accuracy-hao-vs-ydf-params-single-machine, tab:accuracy-hao-vs-ydf-params-distributed | `2026-11-06_hao_seeds`, `2026-11-03_distacc`, `2026-11-04_distacc_aaspo`, `2026-11-20_ydf_seeds`, `2026-11-04_wide_eval`, `2026-11-20_cc18_seeds`, `2026-11-20_regression_multiclass_tabarena_tabred` |
| Evaluation setup (appendix) | `floats/accuracy/tab-ordinal-vs-native/` | tab:ordinal-vs-native | `2026-11-19_ordinal_vs_native` |
| Runtime | `floats/runtime/tab-vertex-hm4-pf-runtime/` | tab:vertex-hm4-pf-runtime, fig:vertex-hm4-hm16-pf-runtime | `vertex_p_F` (pfhm4fix20261101, pfgridfix20261101) |
| Runtime | `floats/runtime/fig-yt8m-barriers/` | fig:yt8m-barriers | `vertex_p_F` (pfhm4fix20261101_youtube8m) |
| Runtime (cross-worker communication) | `floats/runtime/tab-cross-network/` | tab:cross-network, fig:cross-network | `2026-11-01_cross_network` |
| Appendix, in-node parallelism | `floats/appendix/tab-in-node-dist-vqsort-ablation/` | tab:in-node-dist-vqsort-ablation | `vertex_p_F` (pfw1*20261025) |
| Appendix, in-node parallelism | `floats/appendix/tab-in-node-dist-vqsort-vertex/` | tab:in-node-dist-vqsort-vertex | `2026-10-22_in_node_parallelism` |
| Appendix | `floats/appendix/tab-setup-cost/` | tab:setup-cost | `vertex_p_F` (pfhm4fix20261101, pfgridfix20261101, pfw1*20261025) |
| Appendix | `floats/appendix/fig-cc18-tree-feature-hist/` | fig:cc18-tree-feature-hist | `2026-10-26_cc18_gbt_feature_study` |

Diagram
floats without data (`fig:ydf-dist-gbt-training-loop`, `fig:ydf-dist-gbt-load-balancing`) have no dir.

Rules:
1. New measurements land in a new `runs/<YYYY-MM-DD>_<topic>/` with a README, never in a float dir. The topic never names a
   table or section number. Vertex P = F cells go to `runs/vertex_p_F/` (`run_vertex_barriers.py`).
2. Files in a run dir are never edited or renamed after the run. A float reads them in place. A row subset or union lives in
   the float's `derived/`, written by a script in the float dir, and must rebuild byte-identically from `runs/`.
3. `inputs.txt` paths point into `runs/`, into the float's own `derived/`, or outside the results tree. Reading another
   float's `derived/` needs a `# cross-float` comment on that line.
4. When a float is added, renamed or dropped in main.tex, move its dir (dropped -> `outdated/` with a reason line).
5. Check: `python3 benchmarks/src/distributed_spo_gbt/params/check_layout.py [main.tex]` (default: the Overleaf clone, else
   the repo snapshot). It is also run by `params.py selftest` and `check_argv_equivalence.sh`.
