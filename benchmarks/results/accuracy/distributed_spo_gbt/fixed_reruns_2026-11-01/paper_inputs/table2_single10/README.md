# table2_single10: Table 2 / column-replication figure inputs, single row with 10 seeds

- `results.csv`, `summary_pooled.csv`: unchanged copies of `../table2/` (Vertex run colrep-fullparity_fullparity20261101, --split_budget, dgbt-spo-full-fixes 10509539).
- `single_machine_spo_gbt.csv` seeds 1-5 (git 2c0990f0): the 170 rows of `../table2/single_machine_spo_gbt.csv`, unchanged.
- Seeds 6-10 (git 7c0eebd3): parity audit config S1, `ariel_notes/results/2026-11-01_dist_vs_local_parity/` (train_oblique_forest built in wt-parity = wt-disc-fix 7c0eebd3 + distributed-only prototype.diff). Metrics from `logs/cell_logs.tgz` (`test-accuracy/auc/logloss` line), cross-checked against `cells/S1__*.csv`. rc 0, wall_s 0 (not recorded).
- Check (2026-11-02): audit S1 seeds 1-5 match the recorded rows exactly (acc, AUC, log loss) for all 34 datasets.
