# Distributed SPO-GBT results: what each file is (2026-10-26)

Scripts, run recipes and the paper-artifact map: `benchmarks/src/distributed_spo_gbt/REPLICABILITY.md`.

| Path | Provenance | Status |
|---|---|---|
| `barriers/vertex/p_F/<run>/<arm>_w<W>/` | Vertex AI CustomJobs, one VM per node, **P = F**, 10 trees (`pfw1*` W=1 ablation on n2-highmem-16; `pfhm4_*` grid on n2-highmem-4; `pfgrid*` grid on n2-highmem-16; `pfhm4bar*` W=16 with the barrier recorder) | **the paper's distributed numbers** |
| `barriers/vertex/p_sqrtF/<run>/<arm>_w<W>/` | Vertex AI, n2-highmem-4, 50 trees, **P = ceil(sqrt F)** (2026-10-24) | superseded design; feeds `table_vertex_hm4_body.tex` only |
| `barriers/*.csv`, `barriers/*.log` | first barrier measurement, 4 in-process workers on one 8-vCPU box, 100k x 256 trunk | method note, `barriers/README.md` |
| `results.csv`, `anchors.csv` | **in-process** on the m7i (`examples/train_distributed_gbt`, workers = threads of one process); `extra` tags every row `in_process_m7i`, the 300-tree rows also `continuous_weights` (ran continuous instead of binary weights, not comparable) | smoke-test family; still behind `table_dist_runtime_body.tex` |
| `suite_results.csv` | **in-process** on the m7i, Table-9 accuracy suite (33 datasets x folds x W in {1,4,8,16} x 2 arms, 300 trees) | smoke-test family; still behind `table_dist_accuracy_body.tex` |
| `suite_table9_compare.md` | `compare_suite.py` output over `suite_results.csv` | derived |
| `in_node_parallelism_2026-10-22/` | m7i single-machine thread sweep + in-process old-vs-new worker comparison | feeds `table_in_node_threads_body.tex` |
| `vertex_scaling/` | Vertex AI, n2-highmem-2, trunk 32 / 512 columns, 50 trees (2026-10-15) | out of the paper since 2026-10-26 |
| `logs/`, `logs/suite/` | driver logs of the in-process runs (one per `results.csv` / `suite_results.csv` row) | reference |
| `MEMORY_BUDGET.md` | per-worker RAM model, written for 16 GB nodes | B/cell numbers stand; the 16 GB budget column is moot on n2-highmem-16 |

Deleted on 2026-10-26 (recoverable from git history, commits 9cb1fc0f / 88b70553): `20261024T003657Z_epsilon`
(duplicate of `162251Z`), `20261024T225254Z_criteo` (2-tree probe), `20261024T224943Z_*` (dropped in-node
branch at P = ceil(sqrt F); the "SPO in-node" rows of `table_vertex_hm4_body.tex` came from it).
