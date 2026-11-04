# Runs and datasets — reference shard (moved out of CLAUDE.md 2026-11-01)

Not auto-loaded. The standing rules stay in `CLAUDE.md`; this file holds the measured facts,
inventories, recipes and history behind them. Full pre-trim text: `git show 24f08018:CLAUDE.md`.

## Dataset inventory (`benchmarks/data`, audited 2026-09-04)

- **Clean as-is (26):** HIGGS, haberman, processed_wise1_data; CC18 banknote-authentication,
  blood-transfusion-service-center, climate-model-simulation-crashes, wilt, PhishingWebsites,
  numerai28.6, Internet-Advertisements, churn, electricity, diabetes, pc4, pc3, kc2, kc1, pc1,
  spambase, Bioresponse, wdbc, phoneme, qsar-biodeg, madelon, nomao, ozone-level-8hr.
  PhishingWebsites and Internet-Advertisements are integer-coded categoricals/indicators —
  numeric on paper, not continuous.
- **Numeric but with NaNs (massage = train-fold mean-impute):** breast-w (160 cells, 1 col),
  jm1 (250 cells, 5 cols).
- **Non-numeric columns (massage = ordinal-encode):** ilpd (1/10), bank-marketing (9/16),
  credit-g (13/20); kr-vs-kp and tic-tac-toe are all-categorical — ordinal codes only, flag it in the report.
- **Both problems:** dresses-sales, cylinder-bands, credit-approval, adult.
- **YouTube-8M video-level (added 2026-09-21, `download_youtube8m.py`):** `youtube8m/youtube8m_video_{train,validate}.csv`,
  1152 fp32 features (1024 rgb + 128 audio, PCA-whitened, clipped to ±2), NaN-free, clean as-is;
  binary target = majority-vs-rest, entity id 0 ("Game", 20.3 % positive), full label lists in
  the `_labels.csv` sidecar next to it. Details in `benchmarks/SETUP_FRESH_BOX.md` step 8.
- **Large tabular (added 2026-09-21, `download_{criteo,nyc_taxi,airline,numerai}.py`, all clean after
  the script's massage; rows chronological; `<name>_meta.json` lists every dropped column, ordinal map
  and imputation mean):** `criteo/` day 2015-02-15, 195.8M × 39 (13 int + 26 hashed cats as ordinal
  codes; clicks 3.2 %); `nyc_taxi/` 2022-01..2024-06, 98.3M × 18 (target = trip duration > train median
  749 s); `airline/` 2018-2023, 37.9M × 43 (target = ArrDel15, arrival-side/diversion columns dropped as
  leakage, cancelled/diverted rows dropped for lack of label); `numerai/` v5.0 train 2.75M × 2376
  (target 0.5 vs rest). Regression targets were binarised as stated — user has not ruled on regression
  targets yet. 240-tree fit limits on the 377 GB m7i and the prefixes that fit: see
  `benchmarks/data/LARGE_DATASETS_240TREE_FIT.md` (RAM is dominated by the trained model: trees grow to
  purity, ≈25 kB per training row on noisy tall data — HIGGS 10.5M rows peaks at 267 GB).
- **Cardinality caveat (2026-09-22):** Numerai (5 levels per feature), Criteo and taxi have few distinct
  values per column, so VQSort's equal-key shortcut makes Exact 2–3× cheaper per row than on continuous
  data while the binner's cost is value-independent — Exact ties or beats Dynamic there. For HIGGS-like
  comparisons use the `*_dequant.csv` copies from `benchmarks/data/dequantize_dataset.py`.
- **Shifts weather / ClimSim / Jane Street (added 2026-09-24, `download_{shifts_weather,climsim,jane_street}.py`,
  targets binarised at the train median: `fact_temperature`, `cam_out_FLWDS`, `responder_6`):**
  `shifts_weather/` 3.13M × 127, `climsim/` 10.09M × 124 (HF `LEAP/subsampled_low_res`), `jane_street/`
  40.9M × 82 (Kaggle token in `~/.kaggle/access_token`; 240 trees fit only the 10,213,190-row prefix).
  Continuous enough that the VQSort tie shortcut does not apply.
- **Missing locally:** sick (no CSV; TBG column 100% NaN), TabArena (metadata only), TabReD,
  epsilon, `processed/` — the suites live on the m7i.
- Preprocessing reference implementation: `benchmarks/data/tabular_suite_prep.py` (encoding,
  `encode_binary_label`) + `benchmarks/src/spaa27_experiments/run_suite.py::make_folds` (per-fold
  imputation); documented in `benchmarks/results/runtime/speedup_map_by_dataset/README.md`. The CC18
  accuracy study (`benchmarks/results/accuracy/`) predates the rule and fed raw CSVs to the harness.

## What the timers measure (2026-09-09)

- `train_s` = the `random_forest.cc Training block took:` line (bootstrap + all trees + concurrent split
  manager). Right after it RF calls `exit(0)` unless `NO_EARLY_EXIT=1`; `train_oblique_forest.cc` sets
  that only with `--test_csv` / `--model_out_dir`. GBT has no shortcut. Accuracy cells record the
  post-training finalization separately as `train_post_s`.
- `pre_train_s` = "Loading/Init (pre-train)": CSV load ≈ 2.5 ns/cell warm (`fast_csv`; HIGGS 11M×29 0.8 s,
  EPSILON 400k×2001 1.8 s; cold page cache is EBS-bound, ~135 MB/s); in-process trunk generation ≈ 0.22 s
  per 1e9 cells. Paid once per (arm, depth) cell.
- Outside both timers: address-space teardown at the early `exit(0)` ≈ 0.105 s per 1e9 cells.

## Dynamic-histogram threshold history

256-bin breakeven sweep: `benchmarks/results/runtime/dynamic_histogram_breakeven/` (flat 600–1100,
minimum 800–1000). Harness default `-2` resolves from build macros only (250 for both vectorized
binners). `arms.py`: `_DYN64` = 250, `_DYN256` = 1000, `_DYN64_SCALAR` = 4600. Axis-aligned splits honour
the threshold since 2026-09-24 (`UseExactNumericalFinder` in `training.cc`); before that every AA
"Dynamic" run was plain Random. SPO-vs-GBT `spo_rf_dyn_scalar` rows before 2026-09-09 used 250 and render
as `--` until re-run (moved to `speedup_map_by_dataset/_tests/` by `strip_dyn_scalar_thr250.py`).

## Regression measurements (2026-09-24)

`--task regression` uses the same binner, VQSort exact finder and dynamic downgrade. Trees grown to purity
have 2–4× more nodes than the binarised classification tree (weather 1M: 3.5 s cls vs 7.4 s reg; ClimSim
1M: 2.7 s vs 9.8 s; 1 tree, 1 thread, Exact); Dyn/Exact drops from 1.2–1.4× to 1.11–1.13×; pure Random is
0.73–0.79× of Exact. Dataset scripts write `<name>_{train,test}_reg.csv` with `--also_regression`.

## Adding a dataset to the overall depth table (recipe, 2026-09-24)

`bash benchmarks/src/spaa27_experiments/run_b7_dataset.sh <NAME> <train.csv> <label_col>` inside tmux
(6 SPO-RF arms × depths {6,10,16,24,purity}, 240 trees, min_examples 1, 48 threads, 1 run; appends to
`speedup_map.csv`; `COMMIT=1` commits + pushes per stage; holds `run_all.lock`). Then add the dataset to
`DATASETS` in `make_overall_depth_table.py` and regenerate. Prereqs: SETUP_FRESH_BOX.md (bins in
`/home/ubuntu/spo_vs_gbt/bin/{default,scalar}`, venv `/home/ubuntu/gbt_venv`), apt timers masked. Report
each arm as speedup = exact_hwy / arm.

## Distributed GBT barrier timing (2026-10-22)

`DGBT_BARRIER_CSV=<path>` makes the manager write one row per request round (`internal::Monitoring` in
`learner/distributed_gradient_boosted_trees/`): stage, layer, barrier length, manager serial time,
fan-out, worker max/median/min runtime, idle worker-seconds, comm residual, bytes. Off by default.
Summarize with `benchmarks/src/distributed_spo_gbt/summarize_barriers.py`; taxonomy in
`benchmarks/results/kdd28/analysis/barrier_smoke_inprocess_20261022/README.md`.

## History behind standing rules

- **In-node SPO parallelism dropped (2026-10-25):** branches `in-node-parallelism-SPOGBT` @ 08c31fd0,
  `time-barriers-innode` @ 4df36453. With P = ceil(sqrt F) < 10 a node can use at most P threads.
  Reconsider only if P ≫ threads. The *distributed* (node, projection-chunk) worker fix + VQSort is a
  different change: 8.8× / 7.9× / 12.6× on SUSY / HIGGS / Epsilon (P=F, W=1, models byte-identical),
  merged on `main` (4d059967).
- **Vertex-only rule (2026-10-25):** `results.csv` / `suite_results.csv` rows before 2026-10-25 and the
  2026-10-22 APSFailure in-node/sort table were m7i in-process runs, mistaken for cluster runs.
- **Continuous-weights mistake:** the seven 300-tree distributed oblique cells of 2026-10-11 ran
  CONTINUOUS weights (`results.csv` rows with trees==300 and extra==""); not comparable on accuracy.
- **Projection budget:** the 2026-10-22 single-machine in-node thread sweep ran exponent 0.5 for GBT
  unnoticed; its table is kept as run and its caption says so.
- **apt timer incident (2026-09-24):** `apt-daily-upgrade.service` inflated one epsilon cell by 4 %.
- **Checkpoints:** Vertex cells before 2026-10-29 ran with DGBT checkpoints on.
- **Machine type:** Vertex cells before 2026-10-28 used n2-highmem-16.
