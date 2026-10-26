# Distributed SPO-GBT: how to recreate the results

Three families of runs live under `benchmarks/results/runtime/distributed_spo_gbt/`. They share the C++
learner (this fork's `learner/distributed_gradient_boosted_trees`, in-worker sparse-oblique kernel) but
not the harness. **Only family A yields numbers the paper may report as "distributed"** (user directive
2026-10-25); B is out of the paper and C is in-process on one box (smoke tests, kept as history).

| Family | Where it runs | Driver | Results |
|---|---|---|---|
| A. Vertex AI grids on natural datasets (GRPC, one VM per manager and per worker) | GCP project `ydf-distributed-spo-gbt`, `us-east1` | `run_vertex_barriers.py` + `vertex_build/` | `barriers/vertex/p_F/<RUN_ID>_<dataset>/<arm>_w<W>/` (P = F) and `barriers/vertex/p_sqrtF/...` (2026-10-24, P = ceil(sqrt F)) |
| B. Vertex AI weak/strong scaling on trunk data (2026-10-15) | same project, `n2-highmem-2` | the two notebooks + `vertex_build/` | `vertex_scaling/<RUN_ID>_<cols>c/`; removed from `main.tex` on 2026-10-26 |
| C. Single box, workers as threads in one process | AWS `m7i.metal-24xl` | `examples/train_distributed_gbt` via `run_dgbt*.sh`, `queue*.sh` | `results.csv`, `anchors.csv`, `suite_results.csv`, `in_node_parallelism_2026-10-22/`; smoke tests since 2026-10-25 |

## A. Vertex AI grids on natural datasets (`run_vertex_barriers.py`)

One `CustomJob` per (dataset, arm, W) cell: worker pool 0 = manager, pool 1 = W workers, all one machine
type. The training image is built from `vertex_build/` (below); the manager writes `barriers.csv` (the
barrier recorder, `DGBT_BARRIER_CSV`, commit 0e37d40d; `train.py` uploads it since 44cad8d7) next to `eval.json`.
Finished cells are downloaded to
`barriers/vertex/p_F/<RUN_ID>_<dataset>/<arm>_w<W>/{eval.json,barriers.csv,manager.log,summary.txt}` and
summarized with `summarize_barriers.py`. Flags: `--dataset` (epsilon, higgs, susy, youtube8m, climsim,
numerai, criteo, jane_street, airline), `--workers 1,2,4,8`, `--arms aa,spo`, `--machine` (default
`n2-highmem-16`), `--trees` (default 50), `--p_total` (default ceil(sqrt F)), `--image`, `--run_id`
(reuse: finished cells skipped, running jobs adopted), `--dry_run`.

    python run_vertex_barriers.py --dataset epsilon --workers 1,2,4 --arms aa,spo --trees 10 --p_total 2000

Rules (CLAUDE.md): **`n2-highmem-16` for every node, one machine type per comparison, never mixed**
(2026-10-25; N2 custom-training quota in `us-east1` is 96 vCPUs, so W <= 5 per cell on -16). Threads per
worker = the node's vCPUs (YDF default). SPO arms: binary weights, density 1.5, **P = F in total**
(`--p_total F`; the per-W exponent is recorded in `eval.json["learner_kwargs"]`). The 2026-10-24 grids
predate both rules (n2-highmem-4, P = ceil(sqrt F)).

### Runs so far (`barriers/vertex/`, reorganised 2026-10-26 into `p_sqrtF/` and `p_F/`)

| Folders | Date | Machine | Trees | SPO budget | Cells | Feeds |
|---|---|---|---|---|---|---|
| `p_sqrtF/20261024T162251Z_epsilon`, `..162321Z_youtube8m`, `..162440Z_{higgs,susy}` | 10-24 | n2-highmem-4 | 50 | ceil(sqrt F) | aa, spo; W = 1..8 (16 where run) | AA/SPO rows of `tables/table_vertex_hm4_body.tex` via `make_vertex_hm4_table.py`; hm4 column of the setup-cost table |
| `p_F/pfw1{base,vqsort,parstd,innode}20261025_{epsilon,higgs,susy}` | 10-25 | n2-highmem-16 | 10 | F | W = 1: aa + four SPO workers (`0e37d40d` main before the fix, `1a1934fa` VQSort only, `4df36453-stdsort` chunk jobs only, `4df36453` both) | `tables/table_in_node_ablation_body.tex`, `tables/table_setup_cost_body.tex` |
| `p_F/pfhm4_20261025_*`, `p_F/pfgrid20261025_*` | 10-25 | n2-highmem-4 / n2-highmem-16 | 10 | F | aa, spo (parallel worker on `main`); W = 1..16 / 1..4; Epsilon, HIGGS, SUSY, YouTube-8M | `tables/table_vertex_hm4_hm16_pf_body.tex` (the paper's main runtime table); hm16 column of the setup-cost table |
| `p_F/pfhm4bar20261025_*` | 10-25 | n2-highmem-4 | 10 | F | aa, spo at W = 16 with the barrier recorder on | `make_barrier_figs.py` -> the YouTube-8M barrier figures |

Deleted on 2026-10-26 (recoverable from commits 9cb1fc0f / 88b70553): `20261024T003657Z_epsilon` (duplicate
of `162251Z`), `20261024T225254Z_criteo` (2-tree probe), `20261024T224943Z_*` (dropped in-node branch at
P = ceil(sqrt F); the "SPO in-node" rows of `table_vertex_hm4_body.tex` came from it, so that body can no
longer be regenerated with those rows). The `pf*` cells and the three generators were fetched from the m7i
on 2026-10-26 (box commits ea4dc48e, 44cad8d7 = local branch `m7i-main`; the paired-grid version of
`make_pf_grid_table.py` and `make_barrier_figs.py` were uncommitted there). OOM marks in the hm4 grid come
from `p_F/pfhm4_20261025_youtube8m/launcher.log` (`--oom_logs`), reconstructed from the launcher's stdout.
Barrier taxonomy, the in-process measurement and the GRPC idle/transport numbers:
`benchmarks/results/runtime/distributed_spo_gbt/barriers/README.md`.

### Timing definitions (all on the manager's clock, from `eval.json`)

- `train_seconds`: wall time around `learner.train` on the manager.
- `setup_s`: start to the log line "Start tree training" = dataset-cache creation + workers loading their
  columns. Dominated by GCS / VM noise (43-225 s for identical work); never a headline number.
- `tree_training_s`: "Start tree training" to "Training done" = the per-tree loop. **This is the number
  every table uses.** `phase_marks_s` holds the raw marks.
- Vertex's own `start_time`/`end_time` are unreliable (set late); don't use them.

### Images (`vertex_build/`)

| File | Role |
|---|---|
| `vertex_build/wheel/Dockerfile` | Builds the fork's `ydf` wheel from source (manylinux_2_28, upstream `build_test_linux.sh` + `package_linux.sh` recipe) at a pinned commit `YDF_COMMIT`. Image `ydf-wheel:<commit8>`, ~12 min on Cloud Build `e2-highcpu-32`. Pinned to `main` @ 47fe64fa since 2026-10-25 (commit 124b1765). |
| `vertex_build/Dockerfile` | Training image = `python:3.12` + that wheel + `google-cloud-storage pandas gcsfs` + the two scripts below. Image `train-ydf:<commit8>-<build hash>`; the hash covers `Dockerfile`, `gen_trunk.py`, `train.py`, so an edit to any of them retags and rebuilds (~4 min). `eval.json["image"]` records which image a cell ran. |
| `vertex_build/train.py` | Container entrypoint. Reads the Vertex cluster spec (`ydf.util.get_vertex_ai_cluster_spec`); workers call `ydf.start_worker`, the manager trains `DistributedGradientBoostedTreesLearner` with `split_axis`, `sparse_oblique_weights=BINARY`, optional `--num_trees`, `--sparse_oblique_num_projections_exponent`, `--sparse_oblique_projection_density_factor`; everything else YDF defaults. Captures the C++ log through a pipe with manager-clock timestamps and writes `eval.json` (timing, test accuracy / AUC / log-loss, `learner_kwargs`, the `--meta` JSON), `manager.log` and `barriers.csv` to GCS. |
| `vertex_build/gen_trunk.py` | Trunk data generator (family B), run as a one-replica Vertex job so the CSV is written inside the region. Class 1 at +mu, class 0 at -mu, mu_j = 1/sqrt(j+1) on the first 256 columns, unit Gaussian noise (the recipe of `MakeTrunkDataset` in `examples/train_oblique_forest.cc`; numpy RNG, so not bit-identical to the C++ stream), rows shuffled, every smaller size a prefix of the largest. Writes `<rows>x<cols>/train/shard_NN.csv` (16 shards) and a `test_<rows>x<cols>/` set. |
| `vertex_build/shard_csv.sh` | Splits a headered CSV into N row-balanced shards for GCS upload (the m7i datasets: ClimSim, Numerai, Criteo, Jane Street, Airline). |

Datasets already in `gs://ariel_ydf_bucket/big/` and the per-worker memory model: `MEMORY_BUDGET.md`
(written for 16 GB nodes on 2026-10-22; the budget column is moot on n2-highmem-16, the B/cell numbers stand).

Laptop prerequisites: `gcloud` authenticated with the project owner, the `distributed_spo_gbt` Jupyter
kernel (`~/.venvs/distributed_spo_gbt`: `ydf`, `google-cloud-aiplatform`, `gcsfs`, `pandas`, `matplotlib`,
`nbconvert`), `SSL_CERT_FILE=$(python -c 'import certifi;print(certifi.where())')` exported on macOS. Quota
raise recipe: `gcloud alpha quotas preferences create --service=aiplatform.googleapis.com
--quota-id=CustomModelTrainingN2CPUsPerProjectPerRegion --preferred-value=<n> --dimensions=region=us-east1
--billing-project=<project> --justification=...` (approved in seconds on 2026-10-15; 96 since 2026-10-24).
Peek at a running job with `gcloud ai custom-jobs describe <id> --region=us-east1`; container stdout is not
in Cloud Logging, which is why `train.py` uploads `manager.log`.

## B. Vertex AI trunk scaling (notebooks, 2026-10-15) - out of the paper since 2026-10-26

| File | Role |
|---|---|
| `Distributed_SPO_GBT.ipynb` | Setup + toy run. Enables the GCP services, creates the Artifact Registry repo, builds the two Docker images, shards the Adult CSV into GCS, runs one AA-GBT and one SPO-GBT job (1 manager + 5 workers) and reads back `eval.json`. Run it once per fresh project. |
| `Distributed_SPO_GBT_scaling.ipynb` | The weak/strong scaling grid on synthetic trunk data (`COLS`, `TREES`, `MACHINE_TYPE`, `WEAK`/`STRONG` shapes, `RESUME_RUN_ID`). Headless: `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=distributed_spo_gbt Distributed_SPO_GBT_scaling.ipynb`. Writes `results.csv` + `scaling.png` under `vertex_scaling/<RUN_ID>_<COLS>c/`; per-job artifacts in `gs://<bucket>/runs/scaling_<RUN_ID>_<COLS>c/<arm>_n<rows>k_w<W>/`. |
| `make_scaling_tables.py [--run <dir>]` | `tables/table_vertex_{weak,strong}_body.tex` from a run's `results.csv` (default: the 32c run). |

| RUN_ID | cols | trees | machine | status |
|---|---|---|---|---|
| `20261015T035046Z_32c` | 32 | 50 | n2-highmem-2 | complete (18 cells; AA all defaults, SPO P = F total via the per-W exponent) |
| `20261015T134009Z_512c` | 512 | 50 | n2-highmem-2 | 17 of 18 cells: `spo_n1600k_w1` OOM on 16 GB (never rerun), so 512c strong-scaling speedups are relative to W=2; one row restored by hand after a Vertex re-provision. Details in that folder's `NOTES.md`; plot via `finish_512c.py`. |

The SPO budget lesson from this family still applies everywhere: the proto default `exponent = 2` means
P = F^2 per worker (1.6M x 32 at 16 workers: one tree in 27 min, cancelled); always set P = F in total.
`tables/table_vertex_{weak,strong}_body.tex` and `figures/evaluation/grcp_n2/strong_weak_scaling.pdf` are
kept in the repo but no longer input by `main.tex`.

## C. Single box (`m7i.metal-24xl`, workers = threads in one process) - smoke tests only since 2026-10-25

Driver binary: `bazel-bin/examples/train_distributed_gbt` (flags `--num_workers`, `--num_threads`,
`--split_axis`, `--sparse_oblique_*`, `--sparse_oblique_local_projection_count`). Workers are threads of
one process (MULTI_THREAD distribute), so nothing here measures a network or a cluster; the CSVs are kept
because the paper's runtime and accuracy tables were built from them before the 2026-10-25 directive and
have not been replaced by Vertex runs yet. Paths inside the scripts are the box's
(`/home/ubuntu/yggdrasil-oblique-forests`, `/home/ubuntu/spo_vs_gbt`).

| Script | Role |
|---|---|
| `run_dgbt.sh DATASET W AXIS TREES [extra flags]` | One cell on a natural dataset: parses "Training (incl. cache creation) took", "Dataset cache created in" and the "Start tree training" -> "took" timestamps, appends a row to `results.csv`, keeps the log under `logs/`. |
| `run_grid.sh` | Dataset-outer queue of `run_dgbt.sh` cells (`DATASETS`, `CELLS="W:AXIS ..."`, `TREES`), holds the training lock. |
| `run_single_anchor.sh DATASET [TREES] [Oblique|AxisAligned]` | Same-boot single-machine reference (the paper's `spo_gbt_exact_hwy` / AA arm) -> `anchors.csv`. |
| `run_dgbt_suite.sh` | Table-9 accuracy suite on the paper's CV folds, 1/4/8/16 workers, P = F total via `--sparse_oblique_num_projections_exponent 1 --sparse_oblique_local_projection_count=true`, cap lifted, density 1.5, binary; `--max_unique_values_for_discretized_numerical=1`. Skips finished cells -> `suite_results.csv` (logs under `logs/suite/`, 107 MB). |
| `queue2.sh`, `queue3.sh`, `queue_single50.sh`, `queue_suite.sh`, `queue_suite2.sh` | The 2026-10-11/12 run order on the box; each waits on the previous queue's `*.out` marker file (written on the box, not tracked). |
| `run_barrier_cell.sh AXIS [TREES] [W] [THREADS]` | One in-process cell with the barrier recorder on; needs `TRUNK_DIR` from `gen_trunk.py` (>= 256 columns). Source of the first measurement in `barriers/README.md`. |
| `summarize.py` | `results.csv` (+ `anchors.csv`) -> markdown table: per-tree time and "x slower" vs the single-machine anchor. |
| `compare_suite.py` | `suite_results.csv` vs the paper's single-machine GBT arms: test AUC mean +- std over folds, best per row in bold, `--latex`. |
| `summarize_barriers.py <csv> [tree_training_s]` | Per-stage sums, per-layer FindSplits, identity check, idle share; used for family A cells too. |

What the CSVs hold:
- `results.csv`: the 2026-10-12 50-tree grid (W = 1, 4; AA and SPO; binary weights) plus 7 rows with
  `trees == 300` that ran **continuous** weights by mistake (CLAUDE.md, 2026-10-12). Since 2026-10-26 the
  `extra` column tags every row `in_process_m7i` and those 7 also `continuous_weights`; the generators
  filter `trees == 50`, so they are never used. `anchors.csv`: same-boot single-machine references, 50 trees.
- `suite_results.csv`: the Table-9 suite (33 datasets x 5 folds x W in {1,4,8,16} x 2 arms, 300 trees).
- `in_node_parallelism_2026-10-22/`: the single-machine thread sweep and the in-process "distributed"
  old-vs-new worker comparison (moved here from `runtime/GBT/` on 2026-10-26); source of
  `make_in_node_tables.py`.

## D. Paper artifacts and what regenerates them (`paper/distributed_spo_gbt_kdd27/`, checked 2026-10-26)

Generators emit the tabular body only; `main.tex` owns the environments and captions.

| Input in `main.tex` | Generator | Data | Family |
|---|---|---|---|
| `tables/table_vertex_hm4_hm16_pf_body.tex` | `make_pf_grid_table.py --run_prefix pfhm4_20261025 --machine n2-highmem-4 --w 1,2,4,8,16 --oom_logs 'barriers/vertex/p_F/pfhm4_20261025_youtube8m/launcher.log' --right_run_prefix pfgrid20261025 --right_machine n2-highmem-16 --right_w 1,2,4` | `barriers/vertex/p_F/pfhm4_20261025*`, `pfgrid20261025*` | A |
| `figures/evaluation/grcp_n2/fig_yt8m_barriers_{totals,layers}_body.tex` + `yt8m_barriers/*.dat` | `make_barrier_figs.py` (writes in place) | `barriers/vertex/p_F/pfhm4bar20261025_youtube8m/{aa,spo}_w16/barriers.csv` | A |
| `tables/table_in_node_ablation_body.tex`, `tables/table_setup_cost_body.tex` | `make_pf_ablation_tables.py` (writes in place) | ablation: `p_F/pfw1*`; setup cost: every hm4 50-tree cell (`p_sqrtF`) and every hm16 cell (`p_F`) | A |

Regenerated locally on 2026-10-26 from the fetched cells: every body reproduced byte-for-byte except `table_setup_cost_body.tex`,
whose hm16 ranges now include the `pfgrid` W = 1..4 cells (the Overleaf copy was made from the `pfw1` W = 1 cells alone) and whose
Epsilon hm4 range lost the deleted duplicate grid; the new numbers are in the repo, not yet on Overleaf.
| `tables/table_vertex_hm4_body.tex` | `make_vertex_hm4_table.py` | `barriers/vertex/p_sqrtF/*/*/eval.json` (n2-highmem-4, 50 trees); its "SPO in-node" rows have no source since the 2026-10-26 deletion | A (2026-10-24 grids) |
| `tables/table_dist_runtime_body.tex` | `make_paper_tables.py` (body header says in-process) | `results.csv` (trees == 50; `extra` tags `in_process_m7i`) + `anchors.csv` | C |
| `tables/table_dist_accuracy_body.tex`, `figures/evaluation/multi_thread_m7i_metal_24x/fig_dist_accuracy_{delta,mean}_body.tex` | `make_paper_tables.py` | `suite_results.csv` vs `speedup_map_by_dataset/suite_results.csv` | C |
| `tables/table_in_node_threads_body.tex` | `make_in_node_tables.py` | `in_node_parallelism_2026-10-22/` | C |
| `figures/evaluation/multi_thread_m7i_metal_24x/dist_runtime_bars_chat.pdf` | none (made interactively in a chat session) | `results.csv` + `anchors.csv` | C |
| `figures/method/*.pdf`, `listings/in_worker_oblique` | hand-made (SVG sources in `benchmarks/results/figures/`) | - | - |

Generated files kept in the repo but not input by `main.tex` (2026-10-26): `tables/table_vertex_{weak,strong}_body.tex`
and `figures/evaluation/grcp_n2/strong_weak_scaling.pdf` (family B), `tables/table_vertex_hm4_pf_body.tex`,
`tables/table_vertex_hm16_pf_body.tex` (the two halves of the hm4+hm16 table), `tables/table_in_node_distributed_body.tex`
(in-process; its "distributed" rows are not cluster numbers).

Rerun order for a full refresh: (1) family A cells on Vertex (`run_vertex_barriers.py`, n2-highmem-16,
`--p_total F`), (2) the three m7i-only generators once they are back in the repo, then
`make_vertex_hm4_table.py` and `make_paper_tables.py`, (3) push the `.tex` bodies to Overleaf after
compiling the pulled MCP clone locally (CLAUDE.md).
