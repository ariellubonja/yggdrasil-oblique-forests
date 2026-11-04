# Distributed SPO-GBT results: what each file is (2026-10-26)

Scripts, run recipes and the paper-artifact map: `benchmarks/src/distributed_spo_gbt/REPLICABILITY.md`.

| Path | Provenance | Status |
|---|---|---|
| `barriers/vertex/p_F/<run>/<arm>_w<W>/` | Vertex AI CustomJobs, one VM per node, **P = F**, 10 trees (`pfw1*` W=1 ablation on n2-highmem-16; `pfhm4_*` grid on n2-highmem-4; `pfgrid*` grid on n2-highmem-16; `pfhm4bar*` W=16 with the barrier recorder) | **the paper's distributed numbers** |
| `barriers/*.csv`, `barriers/*.log` | first barrier measurement, 4 in-process workers on one 8-vCPU box, 100k x 256 trunk | method note, `barriers/README.md` |
| `results.csv`, `anchors.csv` | **in-process** on the m7i (`examples/train_distributed_gbt`, workers = threads of one process); `extra` tags every row `in_process_m7i`, the 300-tree rows also `continuous_weights` (ran continuous instead of binary weights, not comparable) | smoke-test family; its runtime table was dropped from the paper (2026-11-03) |
| `in_node_parallelism_2026-10-22/` | m7i single-machine thread sweep + in-process old-vs-new worker comparison | feeds `table_in_node_threads_body.tex` |
| `vertex_scaling/` | Vertex AI, n2-highmem-2, trunk 32 / 512 columns, 50 trees (2026-10-15) | out of the paper since 2026-10-26 |
| `logs/` | driver logs of the in-process runs (one per `results.csv` row) | reference |
| `MEMORY_BUDGET.md` | per-worker RAM model, written for 16 GB nodes | B/cell numbers stand; the 16 GB budget column is moot on n2-highmem-16 |

Deleted on 2026-11-04: `suite_results.csv`, `logs/suite/` and `suite_table9_compare.md` (in-process Table-9 accuracy suite of
2026-10-12), removed on 2026-11-04 as pre-fix (no discretization fix 7c0eebd3, VQSort tie fix dc1a8494 or fma fix 5e42cd20), recoverable from git history at commit 185b8b53. No paper float read them. `table_dist_accuracy_body.tex` comes from
`accuracy/distributed_spo_gbt/fixed_reruns_2026-11-01/paper_inputs/table8_results.csv`.

Deleted on 2026-10-26 (recoverable from git history, commits 9cb1fc0f / 88b70553): `20261024T003657Z_epsilon`
(duplicate of `162251Z`), `20261024T225254Z_criteo` (2-tree probe), `20261024T224943Z_*` (dropped in-node
branch at P = ceil(sqrt F)). Also deleted on 2026-10-26, with the paper's `tab:vertex-hm4-runtime` table:
`barriers/vertex/p_sqrtF/` (n2-highmem-4, 50 trees, P = ceil(sqrt F) grids of 2026-10-24), `make_vertex_hm4_table.py`
and `table_vertex_hm4_body.tex` (recoverable from commit 54e79671).
