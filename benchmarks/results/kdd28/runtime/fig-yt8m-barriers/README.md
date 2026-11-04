# fig:yt8m-barriers

Results behind the KDD28 float(s) below. Rebuild the body with the generator command. Index and rules:
`benchmarks/results/kdd28/README.md`.

- **Labels:** `fig:yt8m-barriers`
  - `fig:yt8m-barriers`: "Barrier timing of distributed AA-GBT and SPO-GBT on YouTube-8M with $W{=}16$ workers on Vertex AI \texttt{n2-highmem-4}. (b) Stacked bars sum the slowest worker's time per stage."
- **Status:** current (main body, Runtime section)
- **Profile:** `vertex_grid_pf.spo`, `vertex_grid_pf.aa` (`benchmarks/src/distributed_spo_gbt/params/kdd28_params.json`)
- **Settings:** effective values from the profile (`params.py show <arm>`), checked against the CSV stamp columns
  by the generator where the CSV has them
  - `vertex_grid_pf.spo`: trees=10, max_depth=6, min_examples=5, split_type=EXACT, spo_p_total=F, spo_exponent=log(max(1, F // W) - 0.5) / log(F) if F // W > 1 else 0.0, spo_max_projections=6000, spo_density=1.5, spo_min_projections=1, split_budget=false, local_projection_count=false, budget=per_worker, workers=1 2 4 8 16, machine=n2-highmem-4 n2-highmem-16, folds=shipped test sets, seeds=123456, profile=vertex_grid_pf.spo
  - `vertex_grid_pf.aa`: trees=10, max_depth=6, min_examples=5, split_type=EXACT, workers=1 2 4 8 16, machine=n2-highmem-4 n2-highmem-16, folds=shipped test sets, seeds=123456, profile=vertex_grid_pf.aa
- **Eval suite:** none (runtime float)
- **Machine:** Vertex AI n2-highmem-4, W = 16 (manager = worker type). SMT/turbo: unknown, not recorded
- **Date:** 2026-11-01
- **Code:** image `train-ydf:fullfix-10509539`
- **Generator:** run from the repo root, writes into the paper dir (`--out` to redirect)

```
python3 benchmarks/src/distributed_spo_gbt/make_barrier_figs.py   # --run_dir benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_youtube8m (default)
```
- **Inputs:** `inputs.txt` (one repo-relative path per line)
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_youtube8m/aa_w16/barriers.csv`
  - `benchmarks/results/kdd28/cells/vertex_p_F/pfhm4fix20261101_youtube8m/spo_w16/barriers.csv`
- **Raw cells:** `gs://ariel_ydf_bucket/runs/barriers_pfhm4fix20261101_youtube8m/{aa,spo}_w16/`

Regenerated 2026-11-04: both bodies and `yt8m_barriers/*.dat` equal the Overleaf copies in every non-`%` line.
