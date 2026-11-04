# cells/: raw Vertex AI cells, stored once

`vertex_p_F/<run>_<dataset>/<arm>_w<W>/` (eval.json, barriers.csv, manager.log, cpu_*.json) and the run's
`launcher.log`, written by `run_vertex_barriers.py`. P = F, 10 trees. `logs/` holds the pfw1* launcher logs.
Several floats read the same cells, so each float dir lists the files it uses in its `inputs.txt`:
`runtime/tab-vertex-hm4-pf-runtime/`, `runtime/fig-yt8m-barriers/`, `appendix/tab-in-node-dist-vqsort-ablation/`,
`appendix/tab-setup-cost/`. GCS: `gs://ariel_ydf_bucket/runs/barriers_<run>_<dataset>/`.
