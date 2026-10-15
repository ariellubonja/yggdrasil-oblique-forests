# Distributed SPO-GBT: how to recreate the results

Two families of experiments live in this folder. They share the C++ learner (this fork's
`learner/distributed_gradient_boosted_trees`, in-worker sparse-oblique kernel) but not the harness.

| Family | Where it runs | Driver | Results |
|---|---|---|---|
| A. Vertex AI (GRPC cluster of small VMs) | Google Cloud, project `ydf-distributed-spo-gbt` | the two notebooks + `vertex_build/` | `benchmarks/results/runtime/distributed_spo_gbt/vertex_scaling/<RUN_ID>_<cols>c/` and `gs://ariel_ydf_bucket/runs/` |
| B. Single box, workers as local processes | the AWS `m7i.metal-24x` benchmark box | `run_dgbt*.sh`, `queue*.sh` | `benchmarks/results/runtime/distributed_spo_gbt/{results,anchors,suite_results}.csv` |

## A. Vertex AI

### What each file does

| File | Role |
|---|---|
| `Distributed_SPO_GBT.ipynb` | Setup + toy run. Enables the GCP services, creates the Artifact Registry repo, builds the two Docker images, shards the Adult CSV into GCS, runs one AA-GBT and one SPO-GBT job (1 manager + 5 workers) and reads back `eval.json`. Run it once per fresh project. |
| `Distributed_SPO_GBT_scaling.ipynb` | The weak/strong scaling grid on synthetic trunk data. Needs the wheel image from the toy notebook; builds its own training image when a file in `vertex_build/` changed. |
| `vertex_build/wheel/Dockerfile` | Builds the fork's `ydf` wheel from source (manylinux_2_28, upstream `build_test_linux.sh` + `package_linux.sh` recipe) at a pinned commit `YDF_COMMIT`. Image `ydf-wheel:<commit8>`, ~12 min on Cloud Build `e2-highcpu-32`. |
| `vertex_build/Dockerfile` | Training image = `python:3.12` + that wheel + `google-cloud-storage pandas gcsfs` + the two scripts below. Image `train-ydf:<commit8>-<build hash>`; the hash covers `Dockerfile`, `gen_trunk.py`, `train.py`, so an edit to any of them retags and rebuilds (~4 min). |
| `vertex_build/train.py` | Container entrypoint. Reads the Vertex cluster spec (`ydf.util.get_vertex_ai_cluster_spec`); workers call `ydf.start_worker`, the manager trains `DistributedGradientBoostedTreesLearner` with `split_axis`, `sparse_oblique_weights=BINARY`, optional `--num_trees`, `--sparse_oblique_num_projections_exponent`, `--sparse_oblique_projection_density_factor`; everything else YDF defaults. Captures the C++ log through a pipe with manager-clock timestamps and writes `eval.json` (timing, test accuracy / AUC / log-loss, `learner_kwargs`, the `--meta` JSON) and `manager.log` to GCS. |
| `vertex_build/gen_trunk.py` | Trunk data generator, run as a one-replica Vertex job so the CSV is written inside the region (laptop upload ran at 0.15 MB/s). Class 1 at +mu, class 0 at −mu, mu_j = 1/sqrt(j+1) on the first 256 columns, unit Gaussian noise (the recipe of `MakeTrunkDataset` in `examples/train_oblique_forest.cc`; numpy RNG, so not bit-identical to the C++ stream), rows shuffled, every smaller size a prefix of the largest. Writes `<rows>x<cols>/train/shard_NN.csv` (16 shards) and a `test_<rows>x<cols>/` set. |

### Timing definitions (all on the manager's clock, from `eval.json`)

- `train_seconds`: wall time around `learner.train` on the manager.
- `setup_s`: start → log line "Start tree training" = dataset-cache creation + workers loading their columns.
  Dominated by GCS / VM noise (43–225 s for identical work); never a headline number.
- `tree_training_s`: "Start tree training" → "Training done" = the per-tree loop. **This is the number the
  scaling tables and plots use.** `phase_marks_s` holds the raw marks.
- Vertex's own `start_time`/`end_time` are unreliable (set late); don't use them.

### Steps

1. Prerequisites on the laptop: `gcloud` authenticated with the project owner, the `distributed_spo_gbt`
   Jupyter kernel (`~/.venvs/distributed_spo_gbt`: `ydf`, `google-cloud-aiplatform`, `gcsfs`, `pandas`,
   `matplotlib`, `nbconvert`), `SSL_CERT_FILE=$(python -c 'import certifi;print(certifi.where())')` exported
   on macOS. Mask sleep while a run is on: `caffeinate -s -i -w <nbconvert pid>` (AC power, lid open).
2. Run `Distributed_SPO_GBT.ipynb` top to bottom once (services, repo, wheel image, toy jobs).
3. Quota: Vertex custom-training vCPUs are capped per project, region and machine family. N2 was 20 here;
   1 + 16 machines of `n2-highmem-2` need 34. Raise with
   `gcloud alpha quotas preferences create --service=aiplatform.googleapis.com
   --quota-id=CustomModelTrainingN2CPUsPerProjectPerRegion --preferred-value=48 --dimensions=region=us-east1
   --billing-project=<project> --justification=...` (approved in seconds on 2026-10-15). The scaling notebook
   reads the live quota and uses it as its concurrency budget.
4. Set the constants in the scaling notebook's config cell: `COLS`, `TREES`, `MACHINE_TYPE`, `WEAK`/`STRONG`
   shapes, `RESUME_RUN_ID` (`""` = new run; a run id reuses every cell that already has `eval.json`).
5. Execute it headless:
   `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1
   --ExecutePreprocessor.kernel_name=distributed_spo_gbt Distributed_SPO_GBT_scaling.ipynb`
   It generates the data if missing (one job), submits every (rows, workers, arm) cell as a Vertex
   `CustomJob` (worker pool 0 = manager, pool 1 = `w` workers, all `MACHINE_TYPE`) largest cluster first under
   the vCPU budget, polls every 30 s with retries, then writes `results.csv` + `scaling.png` under
   `RESULTS_DIR/<RUN_ID>_<COLS>c/`.
6. Per-job artifacts: `gs://<bucket>/runs/scaling_<RUN_ID>_<COLS>c/<arm>_n<rows>k_w<W>/{eval.json,manager.log,model,work_*}`.
   Peek at a running job with `gcloud ai custom-jobs describe <id> --region=us-east1` (state + error) or
   the checkpoint folder under its `work_*` dir (iteration number = trees done). Container stdout is not in
   Cloud Logging; that is why `train.py` uploads `manager.log`.

### Settings that are NOT library defaults, and why

- **SPO projection budget.** The in-worker kernel samples `ceil(F ** exponent)` projections per worker per
  node, each a gather + exact sort over the node's rows. The proto default `exponent = 2` means P = F² per
  worker: 1.6M × 32 at 16 workers managed one tree in 27 min (run `20261015T035046Z`, first attempt,
  cancelled). The notebooks use the paper's Table-9 budget instead: **P = F in total**, i.e. each worker
  samples F/W from its own columns — exponent `log(F/W − 0.5)/log(F)` per `w` (`spo_exponent` in the
  config cell), density 1.5, binary weights. Recorded per job in `eval.json["learner_kwargs"]`.
- Machine family `n2-highmem-2` for all nodes (uniform silicon; N1 mixes CPU generations).
- Everything else (depth 6, shrinkage, validation, threads = cores) is the YDF default.

### Runs so far

| RUN_ID | cols | trees | machine | status |
|---|---|---|---|---|
| `20261015T035046Z_32c` | 32 | 50 | n2-highmem-2 | complete (18 cells; AA all defaults, SPO P = F total) |
| 512-column repeat | 512 | 50 | n2-highmem-2 | started 2026-10-15 02:01 EDT, laptop lost network during the data-gen wait; re-run with `RESUME_RUN_ID` of the new id |

Memory rule of thumb for choosing `COLS`: each worker holds ~12 B per cell of its F/W columns (values +
pre-sorted indices + sorted uniques). 1.6M × 512 at W=1 ≈ 10 GB (fits 16 GB); 1.6M × 1024 ≈ 20 GB (does not).

## B. Single box (`m7i.metal-24x`, workers = local processes)

Driver binary: `bazel-bin/examples/train_distributed_gbt` (flags `--num_workers`, `--split_axis`,
`--sparse_oblique_*`, `--sparse_oblique_local_projection_count`). Paths inside the scripts are the box's
(`/home/ubuntu/yggdrasil-oblique-forests`, `/home/ubuntu/spo_vs_gbt`).

| Script | Role |
|---|---|
| `run_dgbt.sh DATASET W AXIS TREES [extra flags]` | One cell on a natural dataset (SUSY/EPSILON/HIGGS/YOUTUBE8M...): parses "Training (incl. cache creation) took", "Dataset cache created in" and the "Start tree training" → "took" timestamps, appends a row to `results.csv`, keeps the log under `logs/`. |
| `run_grid.sh` | Dataset-outer queue of `run_dgbt.sh` cells (`DATASETS`, `CELLS="W:AXIS ..."`, `TREES`), holds the training lock. |
| `run_single_anchor.sh DATASET [TREES] [Oblique|AxisAligned]` | Same-boot single-machine reference (the paper's `spo_gbt_exact_hwy` / AA arm) → `anchors.csv`. |
| `run_dgbt_suite.sh` | Table-9 accuracy suite on the paper's CV folds, 1/4/8/16 workers, P = F total via `--sparse_oblique_num_projections_exponent 1 --sparse_oblique_local_projection_count=true`, cap lifted, density 1.5, binary; `--max_unique_values_for_discretized_numerical=1`. Skips finished cells → `suite_results.csv`. |
| `queue2.sh`, `queue3.sh`, `queue_single50.sh`, `queue_suite.sh`, `queue_suite2.sh` | The 2026-10-11/12 run order on the box (anchors, pass 2/3 grids, 50-tree references, suite + resume), each waiting on the shared lock. |
| `summarize.py` | `results.csv` (+ `anchors.csv`) → markdown table: per-tree time and "x slower" vs the single-machine anchor. |
| `compare_suite.py` | `suite_results.csv` vs the paper's single-machine GBT arms: test AUC mean ± std over folds, best per row in bold, `--latex`. |
| `make_paper_tables.py` | Emits the paper's runtime and accuracy LaTeX tables from the three CSVs. |

Timing columns there: `train_took` (incl. cache creation), `tree_training_s` (per-tree loop), `cache_created`.
