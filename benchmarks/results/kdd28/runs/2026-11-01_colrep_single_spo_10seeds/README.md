# runs/2026-11-01_colrep_single_spo_10seeds: single-machine SPO-GBT row of tab:column-replication, 10 seeds

- Distributed rows: `runs/2026-11-01_colrep_vertex_split_budget/` (until 2026-11-20 this dir also held byte-identical copies of its `results.csv` and `summary_pooled.csv`, removed as duplicates).
- `single_machine_spo_gbt.csv` seeds 1-5 (git 2c0990f0): the 170 rows of `outdated/fixed_reruns_2026-11-01_duplicates/paper_inputs_table2/single_machine_spo_gbt.csv`, unchanged.
- Seeds 6-10 (git 7c0eebd3): parity audit config S1, `ariel_notes/results/2026-11-01_dist_vs_local_parity/` (train_oblique_forest built in wt-parity = wt-disc-fix 7c0eebd3 + distributed-only prototype.diff). Metrics from `logs/cell_logs.tgz` (`test-accuracy/auc/logloss` line), cross-checked against `cells/S1__*.csv`. rc 0, wall_s 0 (not recorded).
- Check (2026-11-02): audit S1 seeds 1-5 match the recorded rows exactly (acc, AUC, log loss) for all 34 datasets.

Measurement, stored once. Floats read it through their `inputs.txt`. Index and rules: `benchmarks/results/kdd28/README.md`.
Line 5 above is cited by `kdd28_params.json` (`table2` `_source`), keep it there.

- **Profile:** `table2.single_spo` (seeds 1-5), `table2.single_spo_audit` (seeds 6-10)
- **Settings:** effective values (`params.py show <arm>`)
  - `table2.single_spo`: trees=10, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=1, spo_max_projections=2147483647, spo_density=1.5, spo_min_projections=1, folds=CC18 fold 0, seeds=1 2 3 4 5, profile=table2.single_spo
  - `table2.single_spo_audit`: trees=10, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=1, spo_max_projections=2147483647, spo_density=1.5, spo_min_projections=1, folds=CC18 fold 0, seeds=6 7 8 9 10, profile=table2.single_spo_audit
- **Eval suite:** `kdd28_cc18_fold0` (34 CC18, fold 0, seeds 1-10)
- **Machine:** EC2 m7i.metal-24xl. SMT/turbo: unknown, not recorded
- **Date:** seeds 1-5: 2026-10-29 (`ariel_notes/results/2026-10-29_local_spo_gbt_baseline`, per
  `outdated/paper_refs_pre_7c0eebd3/table2_single_machine_run_local.sh`), seeds 6-10: 2026-11-01. One file, kept whole
- **Code:** seeds 1-5 2c0990f0, seeds 6-10 7c0eebd3 (the `git` column)
- **Raw cells:** seeds 6-10 `ariel_notes/results/2026-11-01_dist_vs_local_parity/` (gitignored)
- **Read by:** tab:column-replication, fig:column-replication
