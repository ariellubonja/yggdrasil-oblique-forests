# tab:setup-cost

Results behind the KDD28 float(s) below. Rebuild the body with the generator command. Index and rules:
`benchmarks/results/kdd28/README.md`.

- **Labels:** `tab:setup-cost`
  - `tab:setup-cost`: "Setup time in seconds (min--max over all runs)."
- **Status:** appendix. Body regenerated from the cells below differs from the Overleaf copy (made from the superseded 2026-10-25 cells), not pushed: the user decides
- **Profile:** `vertex_grid_pf.spo`, `vertex_grid_pf.aa` (`benchmarks/src/distributed_spo_gbt/params/kdd28_params.json`)
- **Settings:** effective values from the profile (`params.py show <arm>`), checked against the CSV stamp columns
  by the generator where the CSV has them
  - `vertex_grid_pf.spo`: trees=10, max_depth=6, min_examples=5, split_type=EXACT, spo_p_total=F, spo_exponent=log(max(1, F // W) - 0.5) / log(F) if F // W > 1 else 0.0, spo_max_projections=6000, spo_density=1.5, spo_min_projections=1, split_budget=false, local_projection_count=false, budget=per_worker, workers=1 2 4 8 16, machine=n2-highmem-4 n2-highmem-16, folds=shipped test sets, seeds=123456, profile=vertex_grid_pf.spo
  - `vertex_grid_pf.aa`: trees=10, max_depth=6, min_examples=5, split_type=EXACT, workers=1 2 4 8 16, machine=n2-highmem-4 n2-highmem-16, folds=shipped test sets, seeds=123456, profile=vertex_grid_pf.aa
- **Eval suite:** none (runtime float)
- **Machine:** Vertex AI n2-highmem-4 and n2-highmem-16. SMT/turbo: unknown, not recorded
- **Date:** 2026-10-25 (pfw1*), 2026-11-01 (pf*fix20261101)
- **Code:** images as in the two grids: fullfix-10509539 (pf*fix20261101) and the four pfw1* images
- **Generator:** run from the repo root, writes into the paper dir (`--out` to redirect)

```
python3 benchmarks/src/distributed_spo_gbt/make_pf_ablation_tables.py --setup_inputs benchmarks/results/kdd28/appendix/tab-setup-cost/inputs.txt
```
- **Inputs:** `inputs.txt` (one repo-relative path per line)
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfgridfix20261101_epsilon/aa_w1/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfgridfix20261101_epsilon/aa_w2/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfgridfix20261101_epsilon/aa_w4/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfgridfix20261101_epsilon/spo_w1/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfgridfix20261101_epsilon/spo_w2/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfgridfix20261101_epsilon/spo_w4/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfgridfix20261101_higgs/aa_w1/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfgridfix20261101_higgs/aa_w2/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfgridfix20261101_higgs/aa_w4/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfgridfix20261101_higgs/spo_w1/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfgridfix20261101_higgs/spo_w2/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfgridfix20261101_higgs/spo_w4/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfgridfix20261101_susy/aa_w1/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfgridfix20261101_susy/aa_w2/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfgridfix20261101_susy/aa_w4/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfgridfix20261101_susy/spo_w1/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfgridfix20261101_susy/spo_w2/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfgridfix20261101_susy/spo_w4/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfgridfix20261101_youtube8m/aa_w1/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfgridfix20261101_youtube8m/aa_w2/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfgridfix20261101_youtube8m/aa_w4/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfgridfix20261101_youtube8m/spo_w1/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfgridfix20261101_youtube8m/spo_w2/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfgridfix20261101_youtube8m/spo_w4/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_epsilon/aa_w1/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_epsilon/aa_w16/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_epsilon/aa_w2/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_epsilon/aa_w4/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_epsilon/aa_w8/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_epsilon/spo_w1/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_epsilon/spo_w16/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_epsilon/spo_w2/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_epsilon/spo_w4/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_epsilon/spo_w8/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_higgs/aa_w1/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_higgs/aa_w16/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_higgs/aa_w2/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_higgs/aa_w4/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_higgs/aa_w8/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_higgs/spo_w1/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_higgs/spo_w16/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_higgs/spo_w2/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_higgs/spo_w4/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_higgs/spo_w8/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_susy/aa_w1/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_susy/aa_w16/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_susy/aa_w2/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_susy/aa_w4/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_susy/aa_w8/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_susy/spo_w1/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_susy/spo_w16/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_susy/spo_w2/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_susy/spo_w4/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_susy/spo_w8/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_youtube8m/aa_w16/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_youtube8m/aa_w2/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_youtube8m/aa_w4/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_youtube8m/aa_w8/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_youtube8m/spo_w16/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_youtube8m/spo_w4/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_youtube8m/spo_w8/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfw1base20261025_epsilon/aa_w1/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfw1base20261025_epsilon/spo_w1/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfw1base20261025_higgs/aa_w1/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfw1base20261025_higgs/spo_w1/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfw1base20261025_susy/aa_w1/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfw1base20261025_susy/spo_w1/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfw1innode20261025_epsilon/spo_w1/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfw1innode20261025_higgs/spo_w1/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfw1innode20261025_susy/spo_w1/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfw1parstd20261025_epsilon/spo_w1/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfw1parstd20261025_higgs/spo_w1/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfw1parstd20261025_susy/spo_w1/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfw1vqsort20261025_epsilon/spo_w1/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfw1vqsort20261025_higgs/spo_w1/eval.json`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfw1vqsort20261025_susy/spo_w1/eval.json`
- **Raw cells:** the cells of tab:vertex-hm4-pf-runtime and tab:in-node-dist-vqsort-ablation (GCS URIs in their READMEs)

Cell list (decision 2026-11-04): the paper's own cells only, pfhm4fix20261101 + pfgridfix20261101 + pfw1*20261025. Before, the generator read every eval.json under p_F/, incl. the superseded 2026-10-25 grids and the HIGGS probes.
