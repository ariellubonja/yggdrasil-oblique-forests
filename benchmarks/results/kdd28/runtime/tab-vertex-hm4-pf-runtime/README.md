# tab:vertex-hm4-pf-runtime

Results behind the KDD28 float(s) below. Rebuild the body with the generator command. Index and rules:
`benchmarks/results/kdd28/README.md`.

- **Labels:** `tab:vertex-hm4-pf-runtime`, `fig:vertex-hm4-hm16-pf-runtime`
  - `tab:vertex-hm4-pf-runtime`: "Distributed tree-training time (s) on Vertex AI"
  - `fig:vertex-hm4-hm16-pf-runtime`: "Table~\ref{tab:vertex-hm4-pf-runtime} as bars: distributed tree-training time of SPO-GBT and AA-GBT vs. workers $W$, log scale."
- **Status:** current (main body, Runtime section). Not rerun under kdd28_ydf (10 trees, P = F)
- **Profile:** `vertex_grid_pf.spo`, `vertex_grid_pf.aa` (`benchmarks/src/distributed_spo_gbt/params/kdd28_params.json`)
- **Settings:** effective values from the profile (`params.py show <arm>`), checked against the CSV stamp columns
  by the generator where the CSV has them
  - `vertex_grid_pf.spo`: trees=10, max_depth=6, min_examples=5, split_type=EXACT, spo_p_total=F, spo_exponent=log(max(1, F // W) - 0.5) / log(F) if F // W > 1 else 0.0, spo_max_projections=6000, spo_density=1.5, spo_min_projections=1, split_budget=false, local_projection_count=false, budget=per_worker, workers=1 2 4 8 16, machine=n2-highmem-4 n2-highmem-16, folds=shipped test sets, seeds=123456, profile=vertex_grid_pf.spo
  - `vertex_grid_pf.aa`: trees=10, max_depth=6, min_examples=5, split_type=EXACT, workers=1 2 4 8 16, machine=n2-highmem-4 n2-highmem-16, folds=shipped test sets, seeds=123456, profile=vertex_grid_pf.aa
- **Eval suite:** none (runtime float). Shipped test sets of Epsilon, HIGGS, SUSY, YouTube-8M, accuracy column only
- **Machine:** Vertex AI CustomJobs, GRPC, one VM per manager and per worker: n2-highmem-4 (W = 1..16) and n2-highmem-16 (W = 1..4). SMT/turbo: unknown, `eval.json` does not record them (GCE default vCPU = hyperthread)
- **Date:** 2026-11-01
- **Code:** image `train-ydf:fullfix-10509539` (`eval.json["image"]`), branch dgbt-spo-full-fixes 10509539
- **Generator:** run from the repo root, writes into the paper dir (`--out` to redirect)

```
python3 benchmarks/src/distributed_spo_gbt/make_pf_grid_table.py --out paper/distributed_spo_gbt_kdd28/tables/table_vertex_hm4_hm16_pf_body.tex
python3 benchmarks/src/distributed_spo_gbt/make_pf_grid_fig.py
# defaults = pfhm4fix20261101 (n2-highmem-4, OOM marks from its youtube8m launcher.log) + pfgridfix20261101
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
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_youtube8m/launcher.log`
- **Raw cells:** `gs://ariel_ydf_bucket/runs/barriers_<run>_<dataset>/<arm>_w<W>/` (the `model` field of each `eval.json`), local copies `benchmarks/results/kdd28/cells/vertex_p_F/<run>_<dataset>/`. REPLICABILITY §A

Regenerated 2026-11-04: table, both panels and `.dat` files equal the Overleaf copies in every non-`%` line.
