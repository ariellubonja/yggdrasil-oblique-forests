# runs/vertex_p_F: raw Vertex AI cells, stored once

Measurement, stored once (moved from `cells/` 2026-11-20, internal structure unchanged). Index and rules:
`benchmarks/results/kdd28/README.md`.

- **Files:** `<run>_<dataset>/<arm>_w<W>/` (eval.json, barriers.csv, manager.log, cpu_*.json) and the run's `launcher.log`,
  written by `run_vertex_barriers.py`. P = F, 10 trees. `logs/` holds the pfw1* launcher logs. Runs: `pfhm4fix20261101_*`
  (n2-highmem-4, W = 1..16), `pfgridfix20261101_*` (n2-highmem-16, W = 1..4), `pfw1{base,vqsort,parstd,innode}20261025_*`
  (n2-highmem-16, W = 1, the in-node ablation). Two dates in one dir, as before the move
- **Profile:** `vertex_grid_pf.spo`, `vertex_grid_pf.aa` (`benchmarks/src/distributed_spo_gbt/params/kdd28_params.json`)
- **Settings:** effective values (`params.py show <arm>`)
  - `vertex_grid_pf.spo`: trees=10, max_depth=6, min_examples=5, split_type=EXACT, spo_p_total=F, spo_exponent=log(max(1, F // W) - 0.5) / log(F) if F // W > 1 else 0.0, spo_max_projections=6000, spo_density=1.5, spo_min_projections=1, split_budget=false, local_projection_count=false, budget=per_worker, workers=1 2 4 8 16, machine=n2-highmem-4 n2-highmem-16, folds=shipped test sets, seeds=123456, profile=vertex_grid_pf.spo
  - `vertex_grid_pf.aa`: trees=10, max_depth=6, min_examples=5, split_type=EXACT, workers=1 2 4 8 16, machine=n2-highmem-4 n2-highmem-16, folds=shipped test sets, seeds=123456, profile=vertex_grid_pf.aa
- **Eval suite:** none (runtime, shipped test sets of Epsilon, HIGGS, SUSY, YouTube-8M)
- **Machine:** Vertex AI CustomJobs, GRPC, one VM per manager and per worker: n2-highmem-4 and n2-highmem-16. SMT/turbo:
  unknown, `eval.json` does not record them (GCE default vCPU = hyperthread)
- **Date:** 2026-11-01 (pf*fix20261101), 2026-10-25 (pfw1*20261025)
- **Code:** image `train-ydf:fullfix-10509539` for pf*fix20261101 (branch dgbt-spo-full-fixes 10509539), four pre-fix images for
  pfw1*: pfw1base `0e37d40d-aaf05986`, pfw1vqsort `1a1934fa-f6fdd172`, pfw1parstd `4df36453-stdsort-ad2efe13`, pfw1innode
  `4df36453-3aa5c882` (`eval.json["image"]`)
- **Raw cells:** `gs://ariel_ydf_bucket/runs/barriers_<run>_<dataset>/<arm>_w<W>/` (the `model` field of each `eval.json`).
  REPLICABILITY §A
- **Read by:** each float lists the files it uses in its `inputs.txt`: `floats/runtime/tab-vertex-hm4-pf-runtime/`,
  `floats/runtime/fig-yt8m-barriers/`, `floats/appendix/tab-in-node-dist-vqsort-ablation/`, `floats/appendix/tab-setup-cost/`
