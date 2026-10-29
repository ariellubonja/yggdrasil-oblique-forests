# Column replication study: results (branch `column-replication`)

The code has been on `main` since 8ddb03f7: the `in_worker_oblique.column_replication` knob, the Vertex harness
`vertex_build/train_cells.py`, the launcher `column_replication/run_vertex_colrep.py`, and the GCS/cache speedups.
This branch also holds the C++ random_seed fix (713c70f7, not on main yet), the results below, and the paper
generators. The paper bodies the generators write are on `main`.

| Folder | What | Reportable? |
|---|---|---|
| `column_replication_vertex/` | **The result**: Vertex run `full20261029b`, 1870 cells (34 CC18 sets, fold 0 × seeds 1-5 × r = 0, 0.1 … 1). Includes `summary_*`, `launcher.log`, `prep_report.csv` (dataset massage) and `single_machine_spo_gbt.csv` (copy of `single_machine_spo_gbt_baseline/results.csv`). | yes (KDD27 Table 2 + column-replication figure) |
| `single_machine_spo_gbt_baseline/` | Single-machine SPO-GBT, same hyperparameters (`train_oblique_forest`, Exact, 10 trees, depth 6, P = F, density 1.5, seeds 1-5) on the same fold-0 CSVs; run locally on the Mac (`run_local.sh`, git 2c0990f0). | accuracy only |
| `column_replication_vertex_seedbug/` | First Vertex run `full20261029`. The DGBT learner ignored `random_seed` passed through Python, so all 5 "seeds" ran as 123456. Kept only as bug evidence. | no |
| `column_replication_ec2_inprocess_smoke/` | In-process smoke sweep on EC2 (workers = threads, 20 trees, icx build), 2026-10-28. The 32 MB of raw logs stay in `ariel_notes/`. | no (smoke) |
| `column_replication_vertex_smokes/` | Smoke runs that gated the harness changes: cache reuse (v2), local CSV + dataspec (v3), and the 12-cell seed-123456 reproduction on the speed image. | no |
| `../../runtime/distributed_spo_gbt/vertex_speed_2026-10-29/` | Vertex wall-time work: `REPORT.md` (levers, before/after, bit-identity), ablation cells, `BUG_dgbt_random_seed.md`, and the checkpoints-off smoke. | runtime of the harness, not of the method |

**Setup of `column_replication_vertex/`:**
- Cluster: n2-highmem-4, 1 manager + 4 workers (GRPC, one VM per node).
- Model: 10 trees, depth 6, P = F projections per worker per node, density 1.5, binary weights,
  `max_unique_values_for_discretized_numerical=1`.
- Checkpoints off (`--no_checkpoints`).
- Image `train-ydf:speed-1c8152c1` on wheel `ydf-wheel:speed-76c81faa`. The wheel's C++ is identical to main's.
- Data: `gs://ariel_ydf_bucket/cc18_fold0/`, made by `column_replication/prep_cc18_fold0.py`.
- Raw cells: `gs://ariel_ydf_bucket/runs/colrep_full20261029b/s<seed>/<dataset>/r<RR>/{eval.json,manager.log}`.

**Regenerate the paper artifacts.** Run these from this branch with `main`'s paper dir checked out. Both outputs are
byte-identical to Overleaf (checked 2026-10-29).
- `python benchmarks/src/distributed_spo_gbt/make_column_replication_table.py` → `tables/table_column_replication_body.tex`
- `python benchmarks/src/distributed_spo_gbt/make_column_replication_fig.py` → `figures/evaluation/fig_column_replication_body.tex` + `column_replication/*.dat`

**Rerun the sweep** (about 20 min for 5 seeds, under $5):
`python benchmarks/src/distributed_spo_gbt/column_replication/run_vertex_colrep.py --image <train image> --extra_args "--cache_root gs://ariel_ydf_bucket/work/cache_cc18 --reuse_reader"`.
Checkpoints are off by default since cd61cc0f.
