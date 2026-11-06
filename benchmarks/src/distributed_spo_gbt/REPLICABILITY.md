# Distributed SPO-GBT: how to recreate the results

**Accuracy numbers made before 2026-11-01 (KDD28 Tables 2, 3, 8 as first published) carry a bug** (distributed SPO
trained on cache-bin representatives; plus nondeterministic tie order). §G has the fixes, the reruns, and how to
recreate them; it supersedes §E/§F for accuracy.

Since 2026-11-04 every KDD28 result on `main` lives in `benchmarks/results/kdd28/`, laid out like the paper (§L): one dir
per float, named after its LaTeX label. Raw Vertex cells of family A are in `kdd28/cells/vertex_p_F/`. Studies E and F keep
their results (and the generators of their paper artifacts) on their own branch; `main` holds the code they ran,
the paper bodies, and the pointers below. They share the C++
learner (this fork's `learner/distributed_gradient_boosted_trees`, in-worker sparse-oblique kernel) but
not the harness. **Only family A yields numbers the paper may report as "distributed"** (user directive
2026-10-25); B is out of the paper and C is in-process on one box (smoke tests, kept as history).

| Family | Where it runs | Driver | Results |
|---|---|---|---|
| A. Vertex AI grids on natural datasets (GRPC, one VM per manager and per worker) | GCP project `ydf-distributed-spo-gbt`, `us-east1` | `run_vertex_barriers.py` + `vertex_build/` | `kdd28/cells/vertex_p_F/<RUN_ID>_<dataset>/<arm>_w<W>/` (P = F), superseded grids in `kdd28/outdated/` |
| B. Vertex AI weak/strong scaling on trunk data (2026-10-15) | same project, `n2-highmem-2` | the two notebooks + `vertex_build/` | `kdd28/outdated/vertex_scaling_20261015/<RUN_ID>_<cols>c/`; removed from `main.tex` on 2026-10-26 |
| C. Single box, workers as threads in one process | AWS `m7i.metal-24xl` | `examples/train_distributed_gbt` via `run_dgbt*.sh`, `queue*.sh` | `kdd28/outdated/inprocess_m7i_grid_20261012/`, `kdd28/appendix/tab-in-node-dist-vqsort-vertex/`; smoke tests since 2026-10-25 |
| E. Column replication on CC18 (Vertex, n2-highmem-4) | same project | `column_replication/run_vertex_colrep.py` + `vertex_build/train_cells.py` | **branch `column-replication`** (b64346ca), see §E |
| F. Naive column shipping / cross-network (Vertex, n2-highmem-4) | same project | see the branch | **branch `cross-worker-col-comms`** (f4548d37), see §F |
| G. Fixed reruns of Tables 2, 3, 8 + distributed-vs-single parity (2026-11-01) | Vertex (2, 3) and the m7i (8) | `fixed_reruns/` | code `dgbt-spo-full-fixes` (63d05422), results on `main` spread over `kdd28/` by `fixed_reruns/layout.py`, see §G |

## A. Vertex AI grids on natural datasets (`run_vertex_barriers.py`)

One `CustomJob` per (dataset, arm, W) cell: worker pool 0 = manager, pool 1 = W workers, all one machine
type. The training image is built from `vertex_build/` (below); the manager writes `barriers.csv` (the
barrier recorder, `DGBT_BARRIER_CSV`, commit 0e37d40d; `train.py` uploads it since 44cad8d7) next to `eval.json`.
Finished cells are downloaded to
`benchmarks/results/kdd28/cells/vertex_p_F/<RUN_ID>_<dataset>/<arm>_w<W>/{eval.json,barriers.csv,manager.log,summary.txt}` and
summarized with `summarize_barriers.py`. Flags: `--dataset` (epsilon, higgs, susy, youtube8m, climsim,
numerai, criteo, jane_street, airline), `--workers 1,2,4,8`, `--arms aa,spo`, `--machine` (default
`n2-highmem-16`), `--trees` (default 50), `--p_total` (default ceil(sqrt F)), `--image`, `--run_id`
(reuse: finished cells skipped, running jobs adopted), `--dry_run`. Dataset cache: by default one shared cache per (image, dataset, arm,
`--extra_args`, machine) under `gs://ariel_ydf_bucket/work/cache_barriers/` (2026-11-02; the cache does not depend on
W), built by the first cell of that key while its siblings wait; `eval.json["dataset_cache_reused"]` marks cells whose
`setup_s` excludes the build. `--fresh_cache` restores one cache per cell, and is required for images whose `train.py`
predates `--dataset_cache_path` (everything before `train-ydf:fullfix-10509539-cache`).

    python run_vertex_barriers.py --dataset epsilon --workers 1,2,4 --arms aa,spo --trees 10 --p_total 2000

Rules (CLAUDE.md): **`n2-highmem-16` for every node, one machine type per comparison, never mixed**
(2026-10-25; N2 custom-training quota in `us-east1` is 96 vCPUs, so W <= 5 per cell on -16). Threads per
worker = the node's vCPUs (YDF default). SPO arms: binary weights, density 1.5, **P = F in total**
(`--p_total F`; the per-W exponent is recorded in `eval.json["learner_kwargs"]`). The 2026-10-24 grids
predate both rules (n2-highmem-4, P = ceil(sqrt F)).

### Runs so far (`kdd28/cells/vertex_p_F/` and `kdd28/outdated/`, all P = F since 2026-10-26)

| Folders | Date | Machine | Trees | SPO budget | Cells | Feeds |
|---|---|---|---|---|---|---|
| `cells/vertex_p_F/pfw1{base,vqsort,parstd,innode}20261025_{epsilon,higgs,susy}` | 10-25 | n2-highmem-16 | 10 | F | W = 1: aa + four SPO workers (`0e37d40d` main before the fix, `1a1934fa` VQSort only, `4df36453-stdsort` chunk jobs only, `4df36453` both) | `tables/table_in_node_ablation_body.tex`, `tables/table_setup_cost_body.tex` |
| `outdated/vertex_p_F_prefix_20261025/pfhm4_20261025_*`, `.../pfgrid20261025_*` | 10-25 | n2-highmem-4 / n2-highmem-16 | 10 | F | aa, spo (parallel worker on `main`); W = 1..16 / 1..4; Epsilon, HIGGS, SUSY, YouTube-8M | superseded by `cells/vertex_p_F/pf*fix20261101_*` (image fullfix-10509539, 2026-11-01), which feed tab:vertex-hm4-pf-runtime, its figure, fig:yt8m-barriers and tab:setup-cost |
| `outdated/vertex_p_F_prefix_20261025/pfhm4bar20261025_*` | 10-25 | n2-highmem-4 | 10 | F | aa, spo at W = 16 with the barrier recorder on | superseded: the barrier figures now come from `cells/vertex_p_F/pfhm4fix20261101_youtube8m` (`make_barrier_figs.py` default) |

Deleted on 2026-10-26 (recoverable from commits 9cb1fc0f / 88b70553): `20261024T003657Z_epsilon` (duplicate
of `162251Z`), `20261024T225254Z_criteo` (2-tree probe), `20261024T224943Z_*` (dropped in-node branch at
P = ceil(sqrt F); the "SPO in-node" rows of `table_vertex_hm4_body.tex` came from it, so that body can no
longer be regenerated with those rows). The rest of the 2026-10-24 sqrt(F) data went the same day with the paper's
`tab:vertex-hm4-runtime` table: `p_sqrtF/` (n2-highmem-4, 50 trees: `20261024T162251Z_epsilon`, `..162321Z_youtube8m`,
`..162440Z_{higgs,susy}`), `make_vertex_hm4_table.py` and `table_vertex_hm4_body.tex` (recoverable from commit 54e79671).
The setup-cost table's n2-highmem-4 column now reads the P = F n2-highmem-4 cells (setup does not depend on P or trees). The `pf*` cells and the three generators were fetched from the m7i
on 2026-10-26 (box commits ea4dc48e, 44cad8d7 = local branch `m7i-main`; the paired-grid version of
`make_pf_grid_table.py` and `make_barrier_figs.py` were uncommitted there). OOM marks in the hm4 grid come
from `cells/vertex_p_F/pfhm4fix20261101_youtube8m/launcher.log` (`--oom_logs`, the default since 2026-11-04).
Barrier taxonomy, the in-process measurement and the GRPC idle/transport numbers:
`benchmarks/results/kdd28/analysis/barrier_smoke_inprocess_20261022/README.md`.

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
| `vertex_build/train.py` | Container entrypoint. Reads the Vertex cluster spec (`ydf.util.get_vertex_ai_cluster_spec`); workers call `ydf.start_worker`, the manager trains `DistributedGradientBoostedTreesLearner` with `split_axis`, `sparse_oblique_weights=BINARY`, optional `--num_trees`, `--sparse_oblique_num_projections_exponent`, `--sparse_oblique_projection_density_factor`; checkpoints off (`create_checkpoints=false`, since 2026-10-29); everything else YDF defaults. Captures the C++ log through a pipe with manager-clock timestamps and writes `eval.json` (timing, test accuracy / AUC / log-loss, `learner_kwargs`, the `--meta` JSON), `manager.log` and `barriers.csv` to GCS. |
| `vertex_build/gen_trunk.py` | Trunk data generator (family B), run as a one-replica Vertex job so the CSV is written inside the region. Class 1 at +mu, class 0 at -mu, mu_j = 1/sqrt(j+1) on the first 256 columns, unit Gaussian noise (the recipe of `MakeTrunkDataset` in `examples/train_oblique_forest.cc`; numpy RNG, so not bit-identical to the C++ stream), rows shuffled, every smaller size a prefix of the largest. Writes `<rows>x<cols>/train/shard_NN.csv` (16 shards) and a `test_<rows>x<cols>/` set. |
| `vertex_build/shard_csv.sh` | Splits a headered CSV into N row-balanced shards for GCS upload (the m7i datasets: ClimSim, Numerai, Criteo, Jane Street, Airline). |

Datasets already in `gs://ariel_ydf_bucket/big/` and the per-worker memory model: `MEMORY_BUDGET.md` (this directory since 2026-11-04)
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
| `run_dgbt_suite.sh` | Table-9 accuracy suite on the paper's CV folds, 1/4/8/16 workers, P = F total via `--sparse_oblique_num_projections_exponent 1 --sparse_oblique_local_projection_count=true`, cap lifted, density 1.5, binary; `--max_unique_values_for_discretized_numerical=1`. Skips finished cells -> `suite_results.csv` (logs under `logs/suite/`). Its 2026-10-12 output was removed on 2026-11-04 as pre-fix (no discretization fix 7c0eebd3, VQSort tie fix dc1a8494 or fma fix 5e42cd20), recoverable from git history at commit 185b8b53. |
| `queue2.sh`, `queue3.sh`, `queue_single50.sh`, `queue_suite.sh`, `queue_suite2.sh` | The 2026-10-11/12 run order on the box; each waits on the previous queue's `*.out` marker file (written on the box, not tracked). |
| `run_barrier_cell.sh AXIS [TREES] [W] [THREADS]` | One in-process cell with the barrier recorder on; needs `TRUNK_DIR` from `gen_trunk.py` (>= 256 columns). Source of the first measurement in `barriers/README.md`. |
| `summarize.py` | `results.csv` (+ `anchors.csv`) -> markdown table: per-tree time and "x slower" vs the single-machine anchor. |
| `summarize_barriers.py <csv> [tree_training_s]` | Per-stage sums, per-layer FindSplits, identity check, idle share; used for family A cells too. |

What the CSVs hold (moved 2026-11-04: `results.csv`, `anchors.csv`, `logs/` to `kdd28/outdated/inprocess_m7i_grid_20261012/`,
`in_node_parallelism_2026-10-22/` to `kdd28/appendix/tab-in-node-dist-vqsort-vertex/`):
- `results.csv`: the 2026-10-12 50-tree grid (W = 1, 4; AA and SPO; binary weights) plus 7 rows with
  `trees == 300` that ran **continuous** weights by mistake (CLAUDE.md, 2026-10-12). Since 2026-10-26 the
  `extra` column tags every row `in_process_m7i` and those 7 also `continuous_weights`; the generators
  filter `trees == 50`, so they are never used. `anchors.csv`: same-boot single-machine references, 50 trees.
- `suite_results.csv`, `logs/suite/`, `suite_table9_compare.md` (the 2026-10-12 Table-9 suite, 33 datasets x 5 folds x
  W in {1,4,8,16} x 2 arms, 300 trees): removed on 2026-11-04 as pre-fix (no discretization fix 7c0eebd3, VQSort tie fix dc1a8494 or fma fix 5e42cd20), recoverable from git history at commit 185b8b53. `compare_suite.py` and `compare_table8_local.py` went with them.
- `in_node_parallelism_2026-10-22/`: the single-machine thread sweep and the in-process "distributed"
  old-vs-new worker comparison (moved here from `runtime/GBT/` on 2026-10-26); source of
  `make_in_node_tables.py`.

## D. Paper floats and what regenerates them (`paper/distributed_spo_gbt_kdd28/`, checked 2026-11-04)

Generators emit the body only. `main.tex` owns the environments and captions. Each float dir under `benchmarks/results/kdd28/`
(`K/` below) has a README with the exact generator command, the profile and its effective settings, machine, code and raw
cells, and an `inputs.txt`. Profile = the `params/kdd28_params.json` setting the CSVs must match.

| Label(s) (main.tex, Overleaf 86e929e + e57222c) | Body | Generator | Data dir | Profile |
|---|---|---|---|---|
| tab:dist-accuracy-ydf, fig:dist-accuracy-mean-ydf | `tables/table_dist_accuracy_ydf_body.tex`, `figures/evaluation/fig_dist_accuracy_mean_ydf_body.tex` | `make_paper_tables.py --profile kdd28_ydf --suffix _ydf --purple --extra ...` | `K/accuracy/tab-dist-accuracy-ydf/` | `kdd28_ydf.distacc_*` |
| tab:dist-accuracy-ydf-spo-advantage, fig:dist-accuracy-mean-ydf-spo-advantage, fig:column-replication-ydf-spo-advantage | `tables/table_dist_accuracy_ydf_spo_advantage_body.tex`, `figures/evaluation/fig_{dist_accuracy_mean,column_replication}_ydf_spo_advantage_body.tex` + `column_replication_ydf_spo_advantage/*.dat` | `select_spo_advantage.py`, then `make_paper_tables.py --suffix _ydf_spo_advantage`, `make_column_replication_fig.py --suffix _ydf_spo_advantage` (README) | `K/accuracy/tab-dist-accuracy-ydf-spo-advantage/` | `kdd28_ydf.distacc_*`, `kdd28_ydf.colrep_distacc_spo` |
| tab:column-replication-ydf, fig:column-replication-ydf | `tables/table_column_replication_ydf_body.tex`, `figures/evaluation/fig_column_replication_ydf_body.tex` + `column_replication_ydf/*.dat` | `make_column_replication_{table,fig}.py --metric test_auc ...` | `K/accuracy/tab-column-replication-ydf/` | `kdd28_ydf.colrep_distacc_spo` |
| tab:vertex-hm4-pf-runtime, fig:vertex-hm4-hm16-pf-runtime | `tables/table_vertex_hm4_hm16_pf_body.tex`, `figures/evaluation/vertex_pf/fig_vertex_{hm4,hm16}_pf_body.tex` + `.dat` | `make_pf_grid_table.py`, `make_pf_grid_fig.py` (defaults = the paper) | `K/runtime/tab-vertex-hm4-pf-runtime/` | `vertex_grid_pf` |
| fig:yt8m-barriers | `figures/evaluation/grcp_n2/fig_yt8m_barriers_{budget,swimlane}_body.tex` + `yt8m_barriers/*.dat` | `make_barrier_figs.py` | `K/runtime/fig-yt8m-barriers/` | `vertex_grid_pf` |
| tab:cross-network, fig:cross-network | `tables/table_cross_network_body.tex`, `figures/evaluation/fig_cross_network_body.tex` + `cross_network/*.dat` | `make_cross_network_{table,fig}.py` | `K/runtime/tab-cross-network/` | `table3` (old settings, kdd28_ydf rerun pending, §J) |
| tab:dist-accuracy, fig:dist-accuracy-mean | `tables/table_dist_accuracy_body.tex`, `figures/evaluation/multi_thread_m7i_metal_24x/fig_dist_accuracy_mean_body.tex` | `make_paper_tables.py --profile table8 --accuracy_csv ...` | `K/appendix-earlier-settings/tab-dist-accuracy/` | `table8` |
| tab:cc18-aa-spo | `tables/table_cc18_aa_spo_body.tex` | `../cc18_aa_spo_gbt/make_cc18_aa_spo_table.py <csv> <out> --time` | `K/appendix-earlier-settings/tab-cc18-aa-spo/` | `cc18_aa_spo` |
| tab:column-replication, fig:column-replication | `tables/table_column_replication_body.tex`, `figures/evaluation/fig_column_replication_body.tex` + `column_replication/*.dat` | `make_column_replication_{table,fig}.py` (defaults) | `K/appendix-earlier-settings/tab-column-replication/` | `table2` |
| tab:in-node-dist-vqsort-ablation, tab:setup-cost | `tables/table_in_node_ablation_body.tex`, `tables/table_setup_cost_body.tex` | `make_pf_ablation_tables.py` (each table reads only its float's `inputs.txt`) | `K/appendix/tab-in-node-dist-vqsort-ablation/`, `K/appendix/tab-setup-cost/` | `vertex_grid_pf` |
| tab:in-node-dist-vqsort-vertex | `tables/table_in_node_threads_body.tex` | `make_in_node_tables.py` | `K/appendix/tab-in-node-dist-vqsort-vertex/` | `in_node_threads` |
| fig:cc18-tree-feature-hist | `figures/evaluation/cc18_tree_feature_hist_body.tex` + `cc18_tree_feature_hist/*.dat` | `../cc18_gbt_feature_study/make_fig_tree_feature_hist.py --out paper/distributed_spo_gbt_kdd28 --trees 1,44,86,129,172,215,257,300 --hsep 0.45cm --no_counts --k 6` | `K/appendix/fig-cc18-tree-feature-hist/` | `cc18_feature_study` |
| `figures/method/*.pdf`, `listings/in_worker_oblique` | hand-made (SVG sources in `benchmarks/results/figures/`) | - | - | - |

Check 2026-11-04 (every generator run from the new paths into a scratch dir, compared with the Overleaf clone ignoring `%`
lines): all bodies and `.dat` files identical except `tables/table_setup_cost_body.tex`. Its Overleaf copy was made from the
superseded 2026-10-25 cells. The generator now reads only the paper's cells (pf*fix20261101 + pfw1*20261025, listed in
`K/appendix/tab-setup-cost/inputs.txt`), which lowers every range. Not pushed: the user decides. Dropped float
tab:cc18-aa-spo-ydf (Overleaf 39e5431): its inputs are in `K/outdated/tab-cc18-aa-spo-ydf/`, the body still regenerates.
`params/check_layout.py` checks this table's structure against main.tex.

## E. Column replication on CC18 (Vertex, 2026-10-28/29) - results on branch `column-replication`

Question: how much accuracy the disjoint F/W column split costs. YDF assigns columns round-robin over
cost-sorted features, and dynamic re-balancing is off for oblique splits. With `in_worker_oblique.column_replication = r`,
each worker also holds a seeded random fraction r of the other workers' numerical columns; r = 1 means every worker
holds all of them. On `main` since 8ddb03f7: the knob, `vertex_build/train_cells.py` (many cells per CustomJob,
resumable), the launcher `column_replication/run_vertex_colrep.py`, `prep_cc18_fold0.py`, and the GCS/cache
speedups that cut a 374-cell job from ~2 h to ~10 min (all bit-identical).

**Results: branch `column-replication` @ b64346ca, `benchmarks/results/accuracy/distributed_spo_gbt/`.** Pre-fix (stale-bug): the
paper's appendix tab:column-replication reads the fixed rerun in `kdd28/appendix-earlier-settings/tab-column-replication/`.
`README.md` there indexes the folders:
- the Vertex run (`column_replication_vertex/`, run `full20261029b`, 1870 cells);
- the single-machine SPO-GBT baseline;
- the seed-bug run (`full20261029`, all seeds ran as 123456; not reportable);
- the EC2 in-process smoke sweep (not reportable);
- the speed work (`benchmarks/results/runtime/distributed_spo_gbt/vertex_speed_2026-10-29/REPORT.md`).

Raw cells: `gs://ariel_ydf_bucket/runs/colrep_full20261029b/`.

Setup:
- Cluster: n2-highmem-4, 1 manager + 4 workers.
- Model: 10 trees, depth 6, P = F per worker, density 1.5, binary weights, all columns presorted.
- Checkpoints off.
- Data: CC18 fold 0, numeric (policy-encoded), seeds 1-5, r = 0, 0.1, ..., 1.

Paper: KDD28 Table 2 (`tab:column-replication`) and the column-replication figure. Generators and cells are on the
branch; they reproduce the bodies on `main` byte-for-byte.

Known issue: the Python API's `random_seed` is dropped by the DGBT learner. The harness sets it through
`extra_training_config`. The C++ fix is 713c70f7 on the branch, not on `main`. Bug brief:
`vertex_speed_2026-10-29/BUG_dgbt_random_seed.md` on the branch.

## F. Naive column shipping / cross-network communication - results on branch `cross-worker-col-comms`

Baseline in which each worker samples projections over all columns and fetches foreign columns from their owners
every layer. Branch `cross-worker-col-comms` @ f4548d37 (was `naive-column-shipping` / `colship-vertex`):
- code;
- `benchmarks/results/runtime/distributed_spo_gbt/naive_column_shipping/` (on `main` since merge 719fb006, moved 2026-11-04 to
  `kdd28/outdated/naive_column_shipping_20261029/`, pre-fix) (`README.md`, Vertex run
  `vertex_hm4_20261029`: `results.csv`, `launch.sh`, raw eval.json + barrier CSVs);
- generators `make_cross_network_{table,fig}.py` for KDD28 Table 3 / Fig. 9.

That branch is not merged into `main` yet.


## G. Fixed reruns of the accuracy tables (2026-11-01) - code `dgbt-spo-full-fixes`, results in `kdd28/`

Branch `kdd27-fixed-reruns` does not exist (checked 2026-11-04, local and origin). The code is `dgbt-spo-full-fixes` @ 63d05422
plus `fixed_reruns/` on `main`. The results folder `fixed_reruns_2026-11-01/` was split over `benchmarks/results/kdd28/`
on 2026-11-04: `fixed_reruns/layout.py` maps every old subpath (paper inputs to their float dirs, raw runs to the float's
`raw/`, fix stages and paper_refs to `outdated/`, parity and comparisons to `analysis/`).

### Bugs fixed (each with a regression test or A/B; write-ups under `ariel_notes/results/2026-10-3*`/`2026-11-01_*`)

| Commit | Bug | Effect on published accuracy |
|---|---|---|
| 7c0eebd3 (`fix-dgbt-spo-discretization`) | The dataset cache bins numerical columns with (distinct − 1) <= `max_unique_values_for_discretized_numerical` and keeps only bin indices; in-worker SPO turned them back into bin midpoints/edges (`MaterializeNumericalColumn`), inference used raw values. `=1` still binned every 2-valued column. Fix: SPO forces the threshold to 0; a binned non-constant column is an error. | Large: Table 2 r=0 +1.1 pts, Table 3 +2.8–3.1 pts (Internet-Ads .146 → .976); Table 8 ≤0.001 AUC |
| ae5ed038 | Not a bug: new option `in_worker_oblique.split_projections_across_workers`. Without it, P = ceil(F^e) is per worker (W·F per node at r=1). | Parity only (see below) |
| dc1a8494 | Exact-scan VQSort on key-only `K32V32` left tied projected values in a per-process random order → last-bit label sums → random choice among exactly tied candidate splits: same seed, different model, any W. Fix: whole-word keys (value << 32 \| row), still VQSort. | Variance only (paired over 40 seeds, all \|t\| < 2) |
| 5e42cd20 | Slow path, generic fast engine and distributed scan let the compiler choose fma per file; on icx they differed at rows exactly on a threshold. Fix: `std::fma` everywhere. | None (all harnesses report the fast engine) |
| 10509539 | `train_distributed_gbt` reused a cache in `--work_dir` built from another `--train_csv`. Fix: signature file. | None (harnesses use fresh work dirs) |

Branches: the recorded Tables 2 and 8 ran at 10509539 (`dgbt-spo-full-fixes` has since gained 63d05422, a config
guard that no reported run triggers); `fix-disc-colcomms` 1ad2062f = the bin, sort and fma fixes on
`cross-worker-col-comms` (the code for Table 3). `fixed_reruns/` and the results are on `main`. Builds from either 10509539 or the branch head give identical models.

### Recreate

Everything is deterministic with these commits: on the same platform **and the same threads per worker**, a rerun
reproduces the committed `results.csv` cell for cell (checked by a fresh replication on 2026-11-01: Tables 2 and 3
1870+1870+68 cells identical; Table 8 identical once NCPU matched). Threads per worker change the model (root
statistics are summed in thread-count blocks): with NCPU=12 instead of 16, ~4% of Table 8 W<=8 cells moved by up to
0.014 AUC. Different CPU/compiler (Mac vs Linux, libc++ vs libstdc++ RNG) draws different projections; then compare
means, not cells.

**1. Data.**
- Table 8 (33 TabArena/TabReD binary datasets, 5 folds): pinned venv (`benchmarks/src/spaa27_experiments/REPLICABILITY.md` §4,
  §10) **plus `pip install huggingface_hub`** (not in its freeze; 2.4.0 worked), then
  `python benchmarks/src/distributed_spo_gbt/rebuild_table8_data.py <folds_dir>`; it prints the sha256 check against
  `speedup_map_by_dataset/MANIFEST/inputs.sha256` (all 351 must match; it reports 12 `train_nan.csv` files, not the
  13 that spaa27 §10 mentions, and the hashes still match).
- Tables 2-3 (34 CC18 binary datasets, fold 0, fully numeric): already at `gs://ariel_ydf_bucket/cc18_fold0/`. To
  rebuild: `python benchmarks/data/download_cc18_datasets.py`, then
  `python benchmarks/src/distributed_spo_gbt/column_replication/prep_cc18_fold0.py --repo <repo> --out <dir>` and
  `gcloud storage cp -r <dir>/* gs://<bucket>/cc18_fold0/` (pass `--data` to the launcher if not the default bucket).
  A venv with openml 0.15.1, pandas 3.1.0, numpy 2.5.4, scikit-learn 1.9.1 rebuilt the 68 CSVs byte-identical to GCS.

**2. Table 8 (one Linux box; the paper used AWS m7i.metal-24xl, Xeon 8488C, Intel icx; 45-60 min for 4 variants).**
Keep the script's default `NCPU=16` (threads per worker = 16 / W, as recorded) to reproduce cells exactly.
```
git checkout dgbt-spo-full-fixes   # + benchmarks/src/distributed_spo_gbt/fixed_reruns/ from main
source /opt/intel/oneapi/setvars.sh            # icx toolchain; before any `set -u`
bazel build //examples:train_distributed_gbt
benchmarks/src/distributed_spo_gbt/fixed_reruns/run_table8_fixed.sh <out> bazel-bin/examples/train_distributed_gbt <folds_dir>
```
Writes `<out>/{fullfix,fullfix_numeric,fullfix_rep,parity}/results.csv` (765/765/765/612 cells). Single-machine
arms are unchanged by the fixes (their recorded cells in `speedup_map_by_dataset/suite_results.csv` stand);
`run_table8_local.sh` with `ARMS="single:SPO single:AA"` reruns them (the SPO arm passes `--projection_density_factor 1.5`
since 2026-11-03, when the harness flag default became YDF's 2, as does `run_seed_sweep.sh single`).

**3. Tables 2-3 (Vertex AI; ~1.5 h, a few dollars).** Needs `gcloud` authenticated to project
`ydf-distributed-spo-gbt` and a Python with `google-cloud-aiplatform` (the orchestrator box uses `~/vertex_venv`).
```
PY=~/vertex_venv/bin/python benchmarks/src/distributed_spo_gbt/fixed_reruns/run_vertex_fixed.sh <out>
```
Defaults to the recorded commits (`T2_COMMIT` 10509539, `T3_COMMIT` 1ad2062f), whose images
`train-ydf:fullfix-10509539` and `train-ydf:colship-fullfix-1ad2062f` already exist in the project's Artifact
Registry: skip the build with `... <out> table2 table2_split table3`. The `build` step (~17 min on Cloud Build) is for
another commit or project. It then runs `table2/fullfix`, `table2/fullfix_splitbudget` (1870 cells each, ~20 min)
and `table3/fullfix` (68 cells). Steps can be run one at a time (`... <out> table3`). Uses a fresh cache root
(`gs://ariel_ydf_bucket/work/cache_cc18_fixed_<date>`; the recorded runs used `work/cache_cc18_discfix`, built
after 7c0eebd3): **never reuse a cache root built before 7c0eebd3** (e.g. `work/cache_cc18`), it holds
binned columns (SPO now refuses it with FAILED_PRECONDITION). Run it from a box that stays up (the
`ariel Small VertexAI Orchestrator` EC2); each launcher blocks until its jobs finish.

**4. Compare.**
`python benchmarks/src/distributed_spo_gbt/fixed_reruns/compare_fixed_reruns.py --results <dir>` where `<dir>` has the
old `fixed_reruns_2026-11-01/` layout (your `<out>/table2`, `<out>/table3`, `<out>/table8` subfolders next to a copy of
`kdd28/outdated/paper_refs_pre_7c0eebd3/` named `paper_refs/`). Without `--results` it reads the kdd28 tree via `layout.py`. Prints Tables 2, 3, 8 (paper vs fixed) and the parity deltas.

### Expected results (`kdd28/analysis/fixed_reruns_2026-11-01/README.md` has the full tables)
- Table 2 pooled accuracy, r = 0 / 0.5 / 1: paper .8449 / .8527 / .8518 → all fixes .8559 / .8586 / .8574.
- Table 3 mean accuracy in-worker / cross-network: paper .8487 / .8474 → .8795 / .8752.
- Table 8 mean AUC W = 1 / 4 / 16: paper .8312 / .8305 / .8272 → .8312 / .8299 / .8274 (W = 32: .8274).

### Distributed vs single-machine parity
Single-machine SPO-GBT (`train_oblique_forest`) evaluates P = ceil(F^e) projections per node over all F columns.
Distributed SPO differs by (1) P per worker, i.e. W·P per node, unless `--split_budget` /
`--sparse_oblique_split_budget`; (2) at r < 1 a projection only combines one worker's columns. With r = 1 and the split
budget the two match: Table 2 Δacc −0.17 ± 0.12 pts, Δlogloss +0.0000 ± 0.0008 (paired by dataset, 5 seeds);
Table 8 ΔAUC vs the 5-seed single-machine mean −0.00001 / −0.00022 / −0.00009 / −0.00021 at W = 4/8/16/32, and
W = 1 − single = +0.00001 ± 0.00028 (5 seeds each). Against the paper's single seed-1 run alone the gap looks like
−0.0008, but that run sits 0.0007 above its own seed mean: compare against multi-seed single-machine numbers.
Recreate the seed sweep on the Table 8 box with `fixed_reruns/run_seed_sweep.sh <single|w1> <seed> <threads> <out>`
(env `FOLDS`, `SINGLE`, `DIST` = binaries), then `fixed_reruns/analyze_seed_sweep.py <out>`. The split budget refuses
`num_projections_from_local_features` at r < 1 (63d05422): that combination divided an already-local count by W.
Analyses: `ariel_notes/results/2026-11-01_dist_vs_local_parity/report.md`, `.../2026-11-01_final_verify/report.md`. The Table 2 single-machine row
(`kdd28/outdated/paper_refs_pre_7c0eebd3/table2_single_machine_spo_gbt.csv`) was made at commit 2c0990f0 on an Apple-silicon Mac with the default
thread count; rerun it with `fixed_reruns/run_single_machine_cc18.sh <train_oblique_forest built at 2c0990f0>
<cc18_fold0 dir> <out>` (reproduced 170/170 cells on an M1 Pro). It is bit-reproducible only on such a Mac: the
single-machine finder's job split depends on the thread count. (`kdd28/outdated/paper_refs_pre_7c0eebd3/table2_single_machine_run_local.sh` is
the original, with paths from the author's machine.)

## H. Projection budget default (2026-11-02)

**Before:** `in_worker_oblique.split_projections_across_workers` defaulted to false, so every worker sampled
P = ceil(F^e) projections per node (W·P per node). **Now** it defaults to true: P = GetNumProjections(global F) per node
in total, worker w samples floor(P/W) + (w < P mod W) (`distributed_decision_tree::ProjectionsForWorker`). At r = 1
this is the single-machine SPO-GBT budget. `examples/train_distributed_gbt --sparse_oblique_split_budget` and
`vertex_build/train_cells.py` (`--no_split_budget` for legacy) follow the new default. The guard still rejects the split
budget with `num_projections_from_local_features` at r < 1.

Semantics of the recorded results and how to reproduce each:
- **Table 2 published cells** (`table2/fullfix`): per-worker global count, P = F per worker = 4F per node at W = 4.
  `train_cells.py --no_split_budget`, or `run_vertex_fixed.sh table2` (archived sources and image at a pre-change commit).
- **Table 8 / Table 9 cells**: per-worker local count, P = F per node in total only at r = 0. Driver flags
  `--sparse_oblique_local_projection_count=true --sparse_oblique_split_budget=false`; `run_table8_local.sh`,
  `run_dgbt_suite.sh`, `fixed_reruns/run_seed_sweep.sh` and `queue3.sh` pin them.
- **Parity runs** (`table2/fullfix_splitbudget`, Table 8 `parity`): split budget = the new default.
- **Column-replication sweep / smoke** (`run_replication_sweep.py`, `smoke_colrep.sh`): pin `--sparse_oblique_split_budget=false`.
- **§C in-process drivers** (`run_dgbt.sh`, hence `run_grid.sh`/`queue*.sh`, and `run_barrier_cell.sh`): pin
  `--sparse_oblique_split_budget=false` (the recorded `results.csv` SPO rows and the first barrier measurement are per-worker P).
- **§A Vertex grids**: cells before 2026-11-02 (incl. `runtime_reruns/run_vertex_grid_fixed.sh`, image `fullfix-10509539`) used a
  per-W exponent, P_TOTAL // W per worker, recorded in each cell's `eval.json["learner_kwargs"]`. `run_vertex_barriers.py --budget`
  selects the semantics: `split` (default) sends one exponent with ceil(F^e) = P_TOTAL and the library divides it; `per_worker`
  keeps the per-W exponent for old images. The launcher reads the image tag's YDF commit (first 8-hex token that is a commit of
  this repo) and exits when that commit's proto default disagrees with `--budget`, or when no commit is found. `eval.json["meta"]`
  records `budget`, `p_total` and `projections_per_worker` (the largest share under `split`).
- **Caveat (pre-existing, both semantics):** a worker with no numerical candidate for a node skips it and its share is not
  redistributed, so a node gets fewer than P when some worker owns no numerical column (F < W or skewed ownership). Not reached
  by the paper configurations (all features, F >= W).

Evidence (`ariel_notes/results/2026-11-02_dist_matches_single/`, CC18 fold 0, 34 datasets, 10 trees, depth 6, P = F,
seeds 1-5, Mac in-process workers): Δ vs single machine (dataset-paired, SE over datasets): W = 1 +0.08 ± 0.24 pts,
log loss +0.0003 ± 0.0008; W = 4, r = 1 −0.06 ± 0.15 pts, −0.0001 ± 0.0011; legacy W = 4 +0.27 ± 0.19 pts,
−0.0056 ± 0.0017 (27/34 datasets lower log loss). The default and legacy cells are bit-identical to the 2026-11-01
split-budget prototype and old-default cells (170/170 each).

## I. YDF SPO defaults for every new run (user decision, 2026-11-03)

Single-machine and distributed SPO-GBT share the YDF proto defaults: exponent 2, max 6000, density 2, min 1, so
P = min(F², 6000) per node, split across workers (split budget). No early stopping (single machine passes
`--ydf_defaults --validation_set_ratio 0 --early_stopping NONE`, the distributed learner has none). Profile `kdd28_ydf`
(arms `dist_spo`, `single_spo`) in `params/kdd28_params.json`. Tables 2 and 3 and the single-machine rows move to it
when rerun. `examples/train_distributed_gbt` and `vertex_build/train_cells.py` (`--num_trees` 300) now default to these
values. The recorded P = F profiles stay reproducible because their launchers pass the values explicitly:
`run_vertex_colrep.py` (table2: exponent 1, max 2147483647, density 1.5), `run_dgbt.sh` and `run_barrier_cell.sh`
(`--sparse_oblique_max_num_projections 1000`). Images built before this change reject the new `train_cells.py` flags.
Cost probe (`ariel_notes/results/2026-11-03_spo_budget_cost_probe/`, qsar-biodeg CC18 fold 0, F = 41, W = 4 in-process,
300 trees, seed 1, Mac, smoke only): tree training 0.99 s (P = F) vs 7.41 s (YDF), median of 3, test accuracy 0.8679 both.

## J. KDD28 accuracy reruns under `kdd28_ydf` on EC2 (user decision, 2026-11-03)

Box: EC2 `ariel-dist-accuracy-1` (renamed from `ariel-ydf-1`), i-0f90f7764082bf21a, m7i.metal-24xl, Xeon 8488C, 48 cores
online (SMT off), 377 GB. Distributed arms run in-process (MULTI_THREAD workers, `examples/train_distributed_gbt`) on this
box. This exception to §C covers these accuracy tables only. Timing stays Vertex only.

- Code: worktree `~/wt/kdd28-ydf-rerun` at e3279f20 plus the uncommitted profile arms and driver (`code.diff` in the
  results folder). Binaries `bazel build -c opt` (icx), copied to `bin/`.
- Driver: `ydf_reruns/run_kdd28_ydf_ec2.py`, one resumable queue for tab:dist-accuracy (33 TabArena/TabReD, 5 folds or
  1 holdout, single + W = 1/4/8/16/32, SPO and AA), tab:column-replication (34 CC18 fold 0, W = 4, r = 0..1, seeds 1-5,
  single row seeds 1-10), tab:cc18-aa-spo arms (a), (d), (e) (10 folds, seed 1). Arms (b) and (c) stay as recorded.
  Profile arms `kdd28_ydf.{distacc_*,colrep_*,cc18_*}`. Seed 1 everywhere except the seed lists. Threads: single 4,
  2 per worker. Argv golden: `params/golden/kdd28_ydf_ec2.argv`.
- Inputs: TabArena/TabReD folds `~/spo_vs_gbt/folds` (339 files match `MANIFEST/inputs.sha256`). CC18 fold 0 and the 10
  folds rebuilt on the box from `benchmarks/data/cc18_binary_csv`, then checked against the Mac copies
  (`2026-10-28_column_replication/data_cc18_fold0`, `2026-10-29_cc18_aa_spo_gbt/data`): all CSVs byte-identical after one
  climate-model train.csv (one value printed differently in the last digit) was replaced by the Mac copy.
- Results: box `~/results/2026-11-03_ydf_defaults_rerun/{distacc,colrep,colrep_single,cc18}/results.csv` (stamped rows,
  `models/`, `logs/`, `DONE` per table, `ALL_DONE`), mirrored by `ariel_notes/results/2026-11-03_ydf_defaults_rerun/mirror.sh`. Tracked copies of the result CSVs (2026-11-04):
  `kdd28/accuracy/tab-dist-accuracy-ydf/distacc/`, `kdd28/outdated/tab-column-replication-ydf-cc18/{colrep,colrep_single}/`
  (superseded by colrep_distacc), `kdd28/outdated/tab-cc18-aa-spo-ydf/cc18/` (float dropped).
- Determinism (tested 2026-11-03, `determinism/` there): same seed, same threads, two concurrent runs per cell for qsar-biodeg
  (single, W = 4), credit-g (single, W = 4) and heloc (single, W = 4, W = 16), 300 trees. All 7 pairs have byte-identical
  `nodes-*` and identical test metrics. Only `header.pb` (creation metadata) and `data_spec.pb` (label dictionary
  serialized in another order) differ. No worker-reply-order effect was seen in the distributed pairs.
- Not rerun on 2026-11-03: the cross-network arm of tab:cross-network (column shipping then lived only on
  `cross-worker-col-comms` / `fix-disc-colcomms`). Its in-worker arm equals the column-replication r = 0 cells.
- AA+SPO and arm (c) columns of tab:dist-accuracy (2026-11-04): profile arms `kdd28_ydf.distacc_single_aa_spo` and
  `kdd28_ydf.distacc_single_spo_harness_exact300` (the flags of `cc18_aa_spo.spo_harness_exact300`, P = min(F, 1000)),
  `run_kdd28_ydf_ec2.py --tables distacc_aaspo,distacc_exact300` (153 dataset-folds each, single machine, 4 threads,
  yesterday's binaries: no C++ change e3279f20..c3b01f56). Machine: the same instance resized to **c8i.metal-96xl**
  (Xeon 6975P-C, 384 threads, SMT and turbo on, 755 GiB). The earlier §J runs ran on m7i.metal-24xl with SMT off.
  Same-seed determinism was tested on m7i only. Rows carry a `machine` column. Tracked CSVs:
  `benchmarks/results/kdd28/accuracy/tab-dist-accuracy-ydf/distacc_aaspo/`. Argv goldens
  `params/golden/kdd28_ydf_distacc_{aaspo,exact300}.argv`.
- Column replication on the tab:dist-accuracy setup (2026-11-04, c8i.metal-96xl, SMT and turbo on): profile arm
  `kdd28_ydf.colrep_distacc_spo` (= distacc_spo at W = 4 plus `--column_replication r`, r = 0.1..1.0),
  `run_kdd28_ydf_ec2.py --tables colrep_distacc`, 1530 cells, box `~/results/2026-11-04_colrep_distacc/`, mirrored by
  `ariel_notes/results/2026-11-04_colrep_distacc/wait_and_mirror.sh`. r = 0 and `single` reuse the distacc W = 4 SPO and
  single SPO rows. Check: r = 0 rerun on 3 dataset-folds (blood-transfusion f1, diabetes f2, Fitness_Club f0), on m7i and
  again on c8i, gave `nodes-*` and header byte-identical to the distacc W = 4 models and equal AUC.
- Cross-network arm (added 2026-11-04, not yet run): `main` since merge 719fb006 of `cross-worker-col-comms` @ d97fe698
  (split budget applies to ship mode), profile arm `kdd28_ydf.cross_network_ship`, `run_kdd28_ydf_ec2.py --tables cross_network`
  (34 CC18 fold 0 x seeds 1-5, W = 4). Results box `~/results/2026-11-04_cross_network_ydf/`, mirrored by
  `ariel_notes/results/2026-11-04_cross_network_ydf/mirror.sh`. The arm definition is in main's kdd28_params.json.

## K. Eval suites: which data and metric code an accuracy table used (user decision, 2026-11-04)

Comparable KDD28 accuracy tables share eval suite `kdd28_tabarena33`, the setup of tab:dist-accuracy-ydf: 33
TabArena/TabReD datasets, 5 CV folds (1 chronological holdout for ecom-offers, homecredit-default, homesite-insurance),
seed 1 per fold, test ROC AUC, no validation split, no early stopping. Spec: `params/eval_suites.json` (datasets with
fold ids, file templates, label and positive-class rule, seeds, metric implementation, validation, early stopping).
Fold-file sha256: the TabArena suite reuses `benchmarks/results/runtime/speedup_map_by_dataset/MANIFEST/inputs.sha256`,
the CC18 suites `params/eval_suites/*.sha256` (hashed from the Mac copies, 2026-11-04).

- Metric `ydf_eval_v1` (both binaries, no sklearn): YDF's C++ evaluation, ROC of the last label-dictionary class (the
  least frequent training class, label 1 under majority = 0), trapezoid over distinct scores (ties count 1/2), all test
  rows. Sites: `examples/train_oblique_forest.cc:1777-1809`, `examples/train_distributed_gbt.cc:338-347`,
  `metric/metric.cc:1342` (BuildROCCurve), `metric/report.cc:488` (the `auc:` line the EC2 driver parses).
- Stamps: `run_kdd28_ydf_ec2.py` appends `eval_suite,data_sha,metric_impl` to new results CSVs (`data_sha` =
  `<train sha256[:16]>+<test sha256[:16]>`, hashes cached in `<out>/eval_sha_cache.json`). CSVs started earlier keep
  their header. `params.py backfill-eval <csv>` adds the columns from the manifest: done 2026-11-05 for
  `accuracy/tab-dist-accuracy-ydf/distacc/results.csv` (2026-11-03 rows), after all 153 dataset-folds the box read on
  2026-11-04 (colrep_distacc and distacc_aaspo stamps) matched the manifest hashes.
- Hard stops since 2026-11-05 (bulletproofing audit, `ariel_notes/results/2026-11-04_eval_bulletproofing/report.md`):
  an identity mismatch (dataset or fold not in the suite, file hash, metric implementation, seed, validation, early
  stopping, a row of another suite, and in generators a suite dataset or fold missing) prints `WARN eval-suite:` lines,
  then `STOP eval-suite:` and exits 2. The driver pre-flight checks each table's plan against its suite
  (`_driver_tables`, or `--eval_suite`) and also stops on unreadable fold files or a wrong label column. It skips
  coverage, since `--only` runs a subset. Generators declare `--eval_suite` (default from `_generators`).
  `--off_paper "<reason>"` (driver and generators) turns the stops into warnings, stamps driver rows
  `eval_suite=off-paper:<reason>` (which a strict generator refuses) and appends `| OFF-PAPER: <reason>` to the
  `% setup:` line. Unstamped rows stay a WARN (provenance, not identity). The `% setup:` line ends with
  `| eval suite <name>`.
- Effective-params since 2026-11-05: `train_distributed_gbt` also prints `spo_weights`, `max_unique_discretized`,
  `force_numerical`, `num_workers`, and `train_oblique_forest` prints `spo_weights`. The drift check compares them
  (weights must be BINARY, the others against the profile, `column_replication` and `column_shipping` always, W and r
  against the cell's own values in the EC2 driver). A binary built before then gets one WARN and those fields unchecked.
- Check by hand: `python3 params/params.py check-eval <results.csv> [--suite <name>]` (default: the reference).
  Rows without the new columns warn "unstamped eval" and are checked on dataset, fold, seed, validation and early stopping.
- The other suites are not comparable with the reference: `kdd28_cc18_fold0` (tab:column-replication-ydf,
  tab:cross-network) and `kdd28_cc18_10fold` (tab:cc18-aa-spo-ydf). Their fold-0 files differ byte-wise (label column
  name in all 34 datasets, cell values in climate-model-simulation-crashes, credit-approval, cylinder-bands,
  dresses-sales and jm1, measured 2026-11-04). The two datasets whose fold-0 seed-1 accuracy differs between the tables
  (cylinder-bands, dresses-sales) are among them.

## L. Results layout (user request 2026-11-04)

`benchmarks/results/kdd28/` mirrors the paper: `accuracy/`, `runtime/` (main-body sections), `appendix-earlier-settings/`,
`appendix/`, one dir per float named after its label (`tab:x` -> `tab-x`, a table and its chart share the table's dir),
plus `cells/` (raw Vertex cells shared by several floats), `analysis/` (evidence with no float) and `outdated/` (one-line
reason each). Index: `benchmarks/results/kdd28/README.md`. `python3 params/check_layout.py [main.tex]` fails on a data float
without a dir, a dir whose label is not in main.tex, README fields missing, absent `inputs.txt` entries, unnamed
`outdated/`/`analysis/` entries, and `_floats` labels main.tex lacks. It runs in `params.py selftest` (synthetic cases) and
in `check_argv_equivalence.sh` (the real tree, main.tex = `$KDD28_MAIN_TEX`, else the Overleaf clone, else the repo snapshot).
