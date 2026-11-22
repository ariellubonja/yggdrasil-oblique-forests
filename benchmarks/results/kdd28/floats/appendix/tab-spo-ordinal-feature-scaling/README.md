# tab:spo-ordinal-feature-scaling

Results behind the KDD28 float below. Rebuild the body with the generator command. Index and rules:
`benchmarks/results/kdd28/README.md`.

- **Labels:** `tab:spo-ordinal-feature-scaling`
  - `tab:spo-ordinal-feature-scaling`: "Test AUC of single-machine SPO-GBT with features scaled by their standard deviation vs.\ unscaled."
- **Status:** current (appendix, after tab:ordinal-vs-native). Added 2026-11-22
- **Runs:** `runs/2026-11-21_spo_gbt_feature_scaling_none_vs_trainstd_cc18_tabarena_tabred/`. Run facts: its README
- **Profile:** `kdd28_ydf` (off-paper probe, `--off_paper`)
- **Settings:** SPO-GBT, YDF defaults, 300 trees, no validation, early stopping off, seed 1, 4 threads. Arms `A_none`
  (original fold files) and `B_trainstd` (every column divided by its train-fold std). Per dataset: fold mean of
  B minus A test AUC. Mixed = 0 < `n_ordinal_cols` < `cols`. p = Wilcoxon signed-rank over the nonzero dataset deltas
- **Eval suite:** `kdd28_cc18_10fold` (row CC18), `kdd28_tabarena33` + TabReD fold files (row TabArena + TabReD, the
  union of the two TabArena sub-runs)
- **Machine:** ariel-ydf-7 (CC18) and ariel-ydf-1 (TabArena, TabReD). Details: the run README
- **Date:** 2026-11-21
- **Code:** binary 0416519f-dirty. The generator asserts one binary per sub-run, seed 1, 4 threads, both arms and every
  fold (CC18 10, TabArena 5, TabReD 1)
- **Generator:** run from the repo root, writes into the paper dir (no `--out` flag)

```
python3 benchmarks/src/distributed_spo_gbt/paper/tables/make_spo_ordinal_feature_scaling_table.py
```
- **Inputs:** `inputs.txt`
- **Raw cells:** the run README (mirrors under `ariel_notes/results/2026-11-21_*normalization*`, gitignored)
