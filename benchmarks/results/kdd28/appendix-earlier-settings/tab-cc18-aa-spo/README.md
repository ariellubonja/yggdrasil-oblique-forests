# tab:cc18-aa-spo

Results behind the KDD28 float(s) below. Rebuild the body with the generator command. Index and rules:
`benchmarks/results/kdd28/README.md`.

- **Labels:** `tab:cc18-aa-spo`
  - `tab:cc18-aa-spo`: "Test accuracy of single-machine AA and SPO-GBT in CC18."
- **Status:** appendix, frozen
- **Profile:** `cc18_aa_spo.aa_ydf`, `cc18_aa_spo.spo_harness`, `cc18_aa_spo.spo_harness_exact300`, `cc18_aa_spo.spo_ydf`, `cc18_aa_spo.aa_spo_ydf` (`benchmarks/src/distributed_spo_gbt/params/kdd28_params.json`)
- **Settings:** effective values from the profile (`params.py show <arm>`), checked against the CSV stamp columns
  by the generator where the CSV has them
  - `cc18_aa_spo.aa_ydf`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, num_candidate_attributes=-1, validation_set_ratio=0.1, early_stopping=VALIDATION_LOSS_INCREASE, folds=0 1 2 3 4 5 6 7 8 9, seeds=1, profile=cc18_aa_spo.aa_ydf
  - `cc18_aa_spo.spo_harness`: trees=30, max_depth=6, min_examples=5, split_type=DYNAMIC_RANDOM_HISTOGRAM, spo_exponent=1, spo_max_projections=1000, spo_density=1.5, spo_min_projections=1, num_candidate_attributes=-1, validation_set_ratio=0, early_stopping=NONE, include_axis_aligned=false, folds=0 1 2 3 4 5 6 7 8 9, seeds=1, profile=cc18_aa_spo.spo_harness
  - `cc18_aa_spo.spo_harness_exact300`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=1, spo_max_projections=1000, spo_density=1.5, spo_min_projections=1, num_candidate_attributes=-1, validation_set_ratio=0, early_stopping=NONE, include_axis_aligned=false, folds=0 1 2 3 4 5 6 7 8 9, seeds=1, profile=cc18_aa_spo.spo_harness_exact300
  - `cc18_aa_spo.spo_ydf`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=2, spo_max_projections=6000, spo_density=2, spo_min_projections=1, num_candidate_attributes=-1, validation_set_ratio=0.1, early_stopping=VALIDATION_LOSS_INCREASE, include_axis_aligned=false, folds=0 1 2 3 4 5 6 7 8 9, seeds=1, profile=cc18_aa_spo.spo_ydf
  - `cc18_aa_spo.aa_spo_ydf`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=2, spo_max_projections=6000, spo_density=2, spo_min_projections=1, num_candidate_attributes=-1, validation_set_ratio=0.1, early_stopping=VALIDATION_LOSS_INCREASE, include_axis_aligned=true, folds=0 1 2 3 4 5 6 7 8 9, seeds=1, profile=cc18_aa_spo.aa_spo_ydf
- **Eval suite:** `kdd28_cc18_10fold` (34 CC18 binary, 10 CV folds, seed 1)
- **Machine:** EC2 m7i.metal-24xl, single machine. SMT/turbo: unknown, not recorded
- **Date:** 2026-11-03
- **Code:** binary 9a94b070 (`bin_sha` 96c9a72d28ba, before 9dc54c93: harness SPO flags honoured). PROVENANCE.txt is one commit line, no stamp columns: settings are the profile's reconstruction of the run script
- **Generator:** run from the repo root, writes into the paper dir (`--out` to redirect)

```
python3 benchmarks/src/cc18_aa_spo_gbt/make_cc18_aa_spo_table.py benchmarks/results/kdd28/appendix-earlier-settings/tab-cc18-aa-spo/results.csv \
  paper/distributed_spo_gbt_kdd28/tables/table_cc18_aa_spo_body.tex --time
```
- **Inputs:** `inputs.txt` (one repo-relative path per line)
  - `benchmarks/results/kdd28/appendix-earlier-settings/tab-cc18-aa-spo/results.csv`
- **Raw cells:** gitignored `ariel_notes/results/2026-11-03_cc18_harness_exact300/` (logs, PROVENANCE.txt, run_exact300.log)

Regenerated 2026-11-04: equal to the Overleaf copy in every non-`%` line.
