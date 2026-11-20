# tab:column-replication

Results behind the KDD28 float(s) below. Rebuild the body with the generator command. Index and rules:
`benchmarks/results/kdd28/README.md`.

- **Labels:** `tab:column-replication`, `fig:column-replication`
  - `tab:column-replication`: "Test accuracy of distributed SPO-GBT \TODO{($W=4$) is this why acc doesn't change?} vs.\ column replication $r$."
  - `fig:column-replication`: "Table~\ref{tab:column-replication} as charts: mean (top) and per-dataset (bottom) $\Delta$ test accuracy vs.\ \texttt{single}."
- **Status:** outdated: float dropped from main.tex (Overleaf 549e754, 2026-11-04), dir moved from floats/appendix-earlier-settings/ 2026-11-20. Was: appendix, frozen
- **Runs:** `runs/2026-11-01_colrep_vertex_split_budget/` (distributed rows), `runs/2026-11-01_colrep_single_spo_10seeds/`
  (single row, 10 seeds). Run facts: their READMEs
- **Profile:** `table2.dist_split`, `table2.single_spo`, `table2.single_spo_audit` (`benchmarks/src/distributed_spo_gbt/params/kdd28_params.json`)
- **Settings:** effective values from the profile (`params.py show <arm>`), checked against the CSV stamp columns
  by the generator where the CSV has them
  - `table2.dist_split`: trees=10, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=1, spo_max_projections=2147483647, spo_density=1.5, spo_min_projections=1, split_budget=true, local_projection_count=false, column_replication=0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1, max_unique_discretized=1, workers=4, machine=n2-highmem-4, folds=CC18 fold 0, seeds=1 2 3 4 5, profile=table2.dist_split
  - `table2.single_spo`: trees=10, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=1, spo_max_projections=2147483647, spo_density=1.5, spo_min_projections=1, folds=CC18 fold 0, seeds=1 2 3 4 5, profile=table2.single_spo
  - `table2.single_spo_audit`: trees=10, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=1, spo_max_projections=2147483647, spo_density=1.5, spo_min_projections=1, folds=CC18 fold 0, seeds=6 7 8 9 10, profile=table2.single_spo_audit
- **Eval suite:** `kdd28_cc18_fold0` (34 CC18, fold 0, seeds 1-5, single 1-10)
- **Machine:** Vertex AI n2-highmem-4, W = 4 (distributed). Single rows: m7i.metal-24xl. SMT/turbo: unknown, not recorded
- **Date:** 2026-11-01 (single seeds 1-5: 2026-10-29, seeds 6-10: 2026-11-01 parity audit)
- **Code:** dgbt-spo-full-fixes 10509539 + `--split_budget` (Vertex run colrep-fullparity_fullparity20261101). Single: 2c0990f0 (seeds 1-5), 7c0eebd3 (6-10)
- **Generator:** run from the repo root, writes into the paper dir (`--out` to redirect)

```
python3 benchmarks/src/distributed_spo_gbt/paper/tables/make_column_replication_table.py   # defaults = the two run dirs below
python3 benchmarks/src/distributed_spo_gbt/paper/figures/make_column_replication_fig.py
```
- **Inputs:** `inputs.txt` (one repo-relative path per line)
  - `benchmarks/results/kdd28/runs/2026-11-01_colrep_vertex_split_budget/results.csv`
  - `benchmarks/results/kdd28/runs/2026-11-01_colrep_vertex_split_budget/summary_pooled.csv`
  - `benchmarks/results/kdd28/runs/2026-11-01_colrep_single_spo_10seeds/single_machine_spo_gbt.csv`
- **Raw cells:** `benchmarks/results/kdd28/runs/2026-11-01_colrep_vertex_split_budget/`. GCS URIs not recorded. Single row: `benchmarks/results/kdd28/runs/2026-11-01_colrep_single_spo_10seeds/README.md`

Regenerated 2026-11-04: table, figure and `column_replication/*.dat` equal the Overleaf copies in every non-`%` line. Until
2026-11-20 this float held its own `results.csv` and `summary_pooled.csv`, byte-identical copies (sha256) of the files in
`runs/2026-11-01_colrep_vertex_split_budget/`, now removed. The figure command fails on 2026-11-20 (`KeyError: 'fold'` in `seeds_balanced`), the same before
and after the move.
