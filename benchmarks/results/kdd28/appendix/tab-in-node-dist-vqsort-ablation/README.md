# tab:in-node-dist-vqsort-ablation

Results behind the KDD28 float(s) below. Rebuild the body with the generator command. Index and rules:
`benchmarks/results/kdd28/README.md`.

- **Labels:** `tab:in-node-dist-vqsort-ablation`
  - `tab:in-node-dist-vqsort-ablation`: "Benefit of VQSort and distributed in-node parallelism: tree-training time in seconds ($P{=}F$)."
- **Status:** appendix (Enabling Parallelism for Distributed SPO)
- **Profile:** `vertex_grid_pf.spo`, `vertex_grid_pf.aa` (`benchmarks/src/distributed_spo_gbt/params/kdd28_params.json`)
- **Settings:** effective values from the profile (`params.py show <arm>`), checked against the CSV stamp columns
  by the generator where the CSV has them
  - `vertex_grid_pf.spo`: trees=10, max_depth=6, min_examples=5, split_type=EXACT, spo_p_total=F, spo_exponent=log(max(1, F // W) - 0.5) / log(F) if F // W > 1 else 0.0, spo_max_projections=6000, spo_density=1.5, spo_min_projections=1, split_budget=false, local_projection_count=false, budget=per_worker, workers=1 2 4 8 16, machine=n2-highmem-4 n2-highmem-16, folds=shipped test sets, seeds=123456, profile=vertex_grid_pf.spo
  - `vertex_grid_pf.aa`: trees=10, max_depth=6, min_examples=5, split_type=EXACT, workers=1 2 4 8 16, machine=n2-highmem-4 n2-highmem-16, folds=shipped test sets, seeds=123456, profile=vertex_grid_pf.aa
- **Eval suite:** none (runtime float, shipped test sets)
- **Machine:** Vertex AI n2-highmem-16, W = 1. SMT/turbo: unknown, not recorded
- **Date:** 2026-10-25
- **Code:** four pre-fix images: pfw1base `0e37d40d-aaf05986`, pfw1vqsort `1a1934fa-f6fdd172`, pfw1parstd `4df36453-stdsort-ad2efe13`, pfw1innode `4df36453-3aa5c882` (`eval.json["image"]`)
- **Generator:** run from the repo root, writes into the paper dir (`--out` to redirect)

```
python3 benchmarks/src/distributed_spo_gbt/make_pf_ablation_tables.py   # writes this body and tab:setup-cost's, each from its inputs.txt
```
- **Inputs:** `inputs.txt` (one repo-relative path per line)
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
- **Raw cells:** `gs://ariel_ydf_bucket/runs/barriers_pfw1*20261025_<dataset>/`, launcher logs `benchmarks/results/kdd28/cells/vertex_p_F/logs/pfw1*.log`

Caveat (hypothesis, untested): dc1a8494 widened the VQSort key after these runs. Test: rerun pfw1vqsort and pfw1innode on image fullfix-10509539. Regenerated 2026-11-04: equal to the Overleaf copy in every non-`%` line.
