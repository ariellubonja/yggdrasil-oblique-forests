# Replicability notes — SPO vs XGBoost/LightGBM/CatBoost study + speedup map (Sept 2026)

Purpose: let anyone (the user, a reviewer, a future session) re-run **any single recorded
row** of the study and get **identical accuracy** and **timing within noise**, without
re-running the whole chain (~40 h). Section 6 is the spot-check recipe; the rest is what
the check depends on.

Companion files: `PROTOCOL.md` / `SPEC.md` (why each arm/setting was chosen; in
`/home/ubuntu/spo_vs_gbt/`), `replicate_check.py` (the spot-check driver),
`make_manifest.sh` (input hashes), `run_all.sh` (the full chain).

---

## 1. What a recorded row contains

Every row in the three result CSVs is self-describing:

| CSV (`/home/ubuntu/spo_vs_gbt/results/`, copied to `benchmarks/results/runtime/speedup_map_by_dataset/`) | Key columns | Reproduction handle |
|---|---|---|
| `suite_results.csv` (2295 rows = 33 datasets × folds × 15 arms) | `dataset, fold, rep, method, seed` | `cmd` column = exact harness argv (YDF arms) or the sklearn estimator `repr` (python arms); `engine_version` = git sha of the binary or library version |
| `large_results.csv` (HIGGS/SUSY/EPSILON; rep 0 = headline, rep 2 = timing repeat) | `dataset, rep, method, seed` | same |
| `speedup_map.csv` (B1/B2/B4/B5 cells) | `dataset, arm, max_depth, min_examples` | `cmd` column = exact harness argv, re-runnable verbatim |

Per-run logs (full harness stdout/stderr, or the python arm's fit log) are in
`/home/ubuntu/spo_vs_gbt/logs/runs/<dataset>_<fold>_<method>[_rep2][_seed2].log`
(2425 files). `suite_results.csv.provenance.txt` / `large_results.csv.provenance.txt`
record the CLI, host, python and library versions at the time each CSV was created.

## 2. Source state and binaries

- Repo `/home/ubuntu/yggdrasil-oblique-forests`, branch `rebased-main`, HEAD
  `576cc1dd` ("Update comment", on top of cherry-picked `bd85e57e` "v1 speedup GBT
  high-dimensional datasets"). Plus **one uncommitted patch** to
  `examples/train_oblique_forest.cc`: new flags `--min_examples` (default −1 = harness
  default: 1 for Bagging, 5 for Boosting) and `--num_candidate_attributes` (0 = ⌈√F⌉,
  −1 = all). The patch is captured in `MANIFEST/git_state.txt` (see §3) and must be
  applied before rebuilding. *Commit it before anything else drifts.*
- Three binaries in `/home/ubuntu/spo_vs_gbt/bin/`, built by `build_bins.sh`
  (icx 2025.2, `-c opt -O3 -march=native --config=skip_dead_axis_jobs`, plus per-binary
  config; since 2026-09-07 the skip is unconditional and that config is gone, so omit it
  when rebuilding from a newer tree): `default` (Highway VQSort exact + vectorized histograms), `exact_std_sort`
  (`--config=exact_std_sort`), `scalar` (`--config=disable_std_upper_bound_vectorization`).
  sha256 of each is in `MANIFEST/inputs.sha256`; `bin/<name>.gitsha` holds the source sha.
- **Two binary generations appear in `engine_version`:** `3752acb9` (pre-cherry-pick build;
  1008 suite rows, all RF-family) and `576cc1dd` (post-cherry-pick; every YDF GBT row was
  dropped and re-run with it, plus all large/speedup-map rows). The cherry-pick only
  removes dead axis-aligned jobs in GBT split search; RF accuracy is bit-identical between
  the two builds (the re-run GBT rows also reproduced their pre-fix accuracies exactly).
  To replicate everything from one build, rebuild from `576cc1dd` + the patch.
- Python arms: venv `/home/ubuntu/gbt_venv` — Python 3.12.3, xgboost 3.4.1, lightgbm
  4.7.0, catboost 1.2.10, scikit-learn 1.9.0, numpy 2.5.2, pandas 3.0.5
  (`MANIFEST/pip_freeze.txt`).

## 3. Input manifest (hashes)

`bash benchmarks/src/spo_vs_gbt/make_manifest.sh [--large]` writes
`/home/ubuntu/spo_vs_gbt/MANIFEST/`:

- `inputs.sha256` — sha256 of the 3 binaries, every TabArena/TabReD source CSV +
  `meta.json` + `train_nan.csv`, every cached fold CSV + `folds.json` under
  `/home/ubuntu/spo_vs_gbt/folds/`, and the driver scripts/cells files (438 files).
- `large_inputs.sha256` — HIGGS/SUSY/EPSILON train/test CSVs (`--large`, ~28 GB,
  run only when the box is idle).
- `git_state.txt` (HEAD, status, the uncommitted diff), `pip_freeze.txt`, `machine.txt`.

Before a replication run, `sha256sum -c MANIFEST/inputs.sha256 --quiet` must pass. A
changed fold CSV means the folds were regenerated with different code and *nothing
downstream is comparable*.

## 4. Data and folds — how they were made (deterministic)

- TabArena (30 binary datasets): `benchmarks/data/tabarena_binary_csv/<name>/train.csv`,
  encoded by `benchmarks/data/tabular_suite_prep.py` (numeric as-is, bool→0/1,
  datetime→epoch s, categorical→ordinal codes over sorted uniques, label→0/1 with
  majority=0). `train_nan.csv` beside it keeps the NaNs so imputation can be per-fold.
- TabReD (ecom-offers, homecredit-default, homesite-insurance): 40k-row HF mirror,
  which is a **chronological prefix** ⇒ one chronological 80/20 holdout
  (`--chrono-holdout`), first 80 % of rows in file order = train.
- Folds: `StratifiedKFold(n_splits=5, shuffle=True, random_state=0)` on the file
  row order; each fold's NaNs imputed with the **train-fold** column mean from
  `train_nan.csv` (`run_suite.make_folds`). The exact row indices per fold are stored in
  `folds/<dataset>/folds.json`; the materialized `fold{k}_{train,test}.csv` files are
  what every arm actually read (same bytes for YDF and python arms) and are hashed.
- Huge: HIGGS 10.5M/500k (`benchmarks/data/HIGGS_{train_10500k,test_500k}.csv`), SUSY
  4.5M/500k, EPSILON 400k train (`epsilon_normalized_train.csv`) / official 100k test
  (`/home/ubuntu/spo_vs_gbt/data/epsilon_test_100k.csv`, HF `jxie/epsilon-normalized`).
  HIGGS is only ever used at full size (CLAUDE.md rule).
- Speedup map trunk cells: `--input_mode trunk --rows=1000000 --cols=C`, generated
  in-process by the harness from the seed (no file).

## 5. What "identical" and "close" mean

Measured on the rep-0 vs rep-2 pairs of the huge tier (22 pairs, all 11 headline arms):

- **Accuracy: bit-identical** (test_acc, test_auc, test_logloss) for every pair, YDF
  *and* xgboost/lightgbm/catboost. YDF is deterministic at a fixed `--seed` and
  `--num_threads` (RF trees are seeded per tree; GBT split search is deterministic at a
  fixed thread count). The python libraries are deterministic at a fixed
  `random_state`/`random_seed` and `n_jobs`/`thread_count` = 48 on identical versions.
  ⇒ the spot check uses `--acc-tol 0`.
- **Timing:** YDF arms ≤ 2.4 % relative (median 0.2 %); xgboost/catboost ≤ 3.6 %;
  lightgbm up to 13 % on a 20 s run (its 48-thread scaling on HIGGS is noisy).
  ⇒ default `--time-tol 0.15`; runs under 2 s are not time-checked.
- Preconditions for the timing bound: the box is otherwise idle (no `run_all.sh` lock,
  no other training binary; `replicate_check.py` refuses to start if the lock is held),
  48 threads, same instance type (m7i.metal-24xl, Xeon 8488C, 48 cores, 377 GB). A
  different core count changes YDF RF times roughly ∝ cores and changes nothing in
  accuracy for RF; for GBT and python arms a different thread count may change
  accuracy in the last digits.

**Validation performed 2026-09-05** (`results/replication/replication_20260905T153531Z.csv`,
run with `--allow-concurrent` while the chain was still training): 6 diabetes rows
(spo_rf_dyn_vec, spo_rf_exact_stdsort, xgboost, lightgbm, catboost, lightgbm_rf) all
reproduced test_acc / test_auc / test_logloss **exactly**. Timings were meaningless in that
run (xgboost 0.14 s → 72 s, lightgbm_rf 2 s → 728 s: OpenMP oversubscription against the
chain's 48 threads) — which is exactly why the lock guard exists; `--allow-concurrent`
proves accuracy only, never time.

**Incident, same day:** that concurrent validation run contaminated three EPSILON rep-2
timings in the chain that was still running (spo_rf_dyn_vec 50 → 91 s, aa_rf_exact 52 → 97 s,
spo_rf_exact_hwy 59.0 → 59.8 s). Those rows were dropped (backup in
`results/_tests/large_results.before_epsilon_rep2_redo.csv`) and re-run alone
(`logs/rep2_epsilon_redo.log`). Lesson, again: never run anything with `--allow-concurrent`
while a timing stage is active; even sub-second python arms spin 48 OpenMP threads.

## 6. Spot-check recipe (what the user will ask for)

```bash
cd /home/ubuntu/yggdrasil-oblique-forests/benchmarks/src/spo_vs_gbt
sha256sum -c /home/ubuntu/spo_vs_gbt/MANIFEST/inputs.sha256 --quiet && echo inputs-OK

# random sample: N rows from each of suite / large / speedup, seeded
/home/ubuntu/gbt_venv/bin/python replicate_check.py --n 5 --sample-seed 0 --dry-run   # list only
/home/ubuntu/gbt_venv/bin/python replicate_check.py --n 5 --sample-seed 0             # run

# cheap sample only (skip the multi-hour library RF-mode rows)
/home/ubuntu/gbt_venv/bin/python replicate_check.py --n 8 --exclude-slow --max-train-s 600

# explicit rows: dataset,fold,method[,rep[,seed]]  (large tier: fold 0)
/home/ubuntu/gbt_venv/bin/python replicate_check.py \
    --rows "credit-g,3,spo_rf_dyn_vec" --rows "HIGGS,0,xgboost" --rows "EPSILON,0,spo_gbt_dyn_vec"
```

The script re-runs each row through the **same functions** the study used
(`run_suite.run_ydf`, `run_suite.run_python` in a spawned child with the untimed warm-up,
`run_speedup_map.run_cell`), asserts the rebuilt argv equals the recorded `cmd`, and writes
`/home/ubuntu/spo_vs_gbt/results/replication/replication_<utc>.csv` with recorded vs new
values and `PASS`/`FAIL:<fields>` per row (exit 1 on any failure). Logs go to
`logs/replication/`.

Manual equivalent for a YDF row: copy the `cmd` cell and run it; parse
`Training block took: X s` (= `train_s`) and the `test-accuracy:… test-auc:… test-logloss:…` line.
For a python row: `python -c` with the recorded estimator `repr`, fit on
`folds/<dataset>/fold<k>_train.csv`, evaluate on `fold<k>_test.csv` (label column from
`folds.json`); time only `.fit()`.

Cost guide for sampling: suite rows are seconds to minutes (largest: NATICUSdroid /
Diabetes130US ~1–3 min for YDF arms, lightgbm_rf up to ~10 min); huge rows: SPO-RF ~1.5–6
min, SPO-GBT ~1–27 min, XGBoost/LightGBM/CatBoost ≤1 min, **xgboost_rf/lightgbm_rf 40–50
min** (HIGGS/SUSY) — use `--exclude-slow` unless those are wanted.

## 7. Things that are *not* replicable as-is, and why

- `xgboost_rf` and `lightgbm_rf` on EPSILON: `OOM` rows (status recorded), not accuracy.
- Timing across boxes: only the same instance type reproduces the absolute numbers; the
  *ratios* between arms are what the paper reports.
- Suite RF rows built at `3752acb9` vs a fresh `576cc1dd` build: accuracy identical,
  timing within noise (the diff does not touch the RF path).
- The Overleaf tables/figures are regenerated by `analyze.py` from the three CSVs; a
  changed CSV changes them — never hand-edit `table_*.tex`.
- The HIGGS row prefixes under `/home/ubuntu/spo_vs_gbt/data/higgs_train_*.csv` are
  unused (user directive: full HIGGS only) and are not part of any recorded row.

## 8. Re-running a whole stage instead of a row

`rm /home/ubuntu/spo_vs_gbt/<stage>_DONE` then `tmux new-session -d -s spo_run 'bash
run_all.sh'` (or `STAGES=<stage> bash run_all.sh`). Skip-existing is keyed on
`(dataset, fold, rep, method, seed)`; delete the rows you want recomputed from the CSV
first (keep a copy under `results/_tests/`), or the stage will skip them.

## 9. Speedup-map stages added after the panel (B6, 256-bin arms, B7 depth table)

All three append to `/home/ubuntu/spo_vs_gbt/results/speedup_map.csv` (copied to
`benchmarks/results/runtime/speedup_map_by_dataset/speedup_map.csv` at the end of each stage/depth) via
`run_speedup_map.py`, keyed `(dataset, max_depth, min_examples, arm)`, status OK ⇒ skipped
on re-run. Same binaries (`bin/{default,scalar,exact_std_sort}`, sha in `bin/*.gitsha`),
48 threads, seed 1, 240 trees, `--compute_oob_performances=false`; the recorded `train_s`
is the harness "Training block took" timer, `cmd` is the exact argv. Every stage takes the
study lock (`flock` on `run_all.lock`) so no two trainings overlap.

- **B6 trunk row-column map (2026-09-07):** `run_b6.sh` → `cells_b6.json` (12 shapes ×
  {exact_stdsort, exact_hwy, dyn_vec}, depth −1, min_examples 1). Trunk cells are generated
  in-process from the seed (`--input_mode trunk --rows R --cols C`), no file. 36/36 OK, 6 h 21 m.
- **256-bin arms (2026-09-07):** `run_256.sh` → suite + large for `spo_rf_rand256_vec`,
  `spo_rf_dyn256_vec`, `spo_gbt_dyn256_vec` (`arms.py`; the `default` binary picks the
  AVX-512 256-threshold kernel). Smoke test first: `compare_models.sh` on `--model_out_dir`
  trees from the `default` vs `scalar` binaries, bit-identical (`/home/ubuntu/spo_vs_gbt/smoke256/`).
- **B7 overall depth table (started 2026-09-08 20:58 UTC):** `run_b7.sh [REP]` →
  `cells_b7_{d6,d10,d16,d24,full}.json`, depth-major. 19 entries (HIGGS 10.5M, SUSY, EPSILON,
  GiveMeSomeCredit full `train.csv`, trunk 1M×{32,512,2048}, the 12 B6 shapes) × 6 SPO-RF
  arms (exact_stdsort, exact_hwy, rand_scalar, rand_vec, dyn_scalar, dyn_vec) × depths
  {6,10,16,24,−1}, min_examples 1. Natural datasets are explicit-`csv` cells (paths/label
  cols in the cells files; HIGGS via the default rows→CSV map). REP=1 appends to
  `speedup_map.csv` reusing existing purity cells; REP>1 writes
  `results/speedup_map_rep<REP>.csv` with `--no-skip-existing`. Per-depth logs
  `logs/b7_<tag>[_rep<k>].log`, markers `b7_<tag>[_rep<k>]_DONE`. Wall ≫ train_s because
  trunk generation / CSV parse is redone per cell (depth 6: 5.3 h wall for 1.9 h training).
  Table: `make_overall_depth_table.py` → `overleaf-spaa27/table_overall_depth.tex`, 17 rows
  (D8-rev2 drops trunk 15k×4096 and 15k×40k), median ± sample std over the rep files present.
- **Headline-speed rule D8-rev2 (2026-09-08):** `analyze.py` counts a dataset for runtime
  only if rows > 100,000 OR features > 100,000 (`MIN_ROWS/MIN_COLS_FOR_HEADLINE_SPEED`).

To replicate one B7 cell: take its `cmd` from `speedup_map.csv` and run it under the lock
on an idle box; expect train_s within a few % (timing repeats on this box are <1 % at
300 s, larger for sub-second cells such as GiveMeSomeCredit at depth 6).
- **B7 amendment (2026-09-09):** `spo_rf_dyn_scalar` removed from `cells_b7_{d16,d24,full}.json`
  mid-run (threshold mis-tuned for the scalar binner, see PROTOCOL.md); the depth-16 run was
  restarted with `run_b7.sh 1` (depths 6/10 and finished depth-16 cells skip via skip-existing).
- **D15 dyn_scalar re-run (2026-09-09):** `spo_rf_dyn_scalar` threshold corrected 250 → 4600
  (`arms.py _DYN64_SCALAR`; harness default now build-resolved). Old rows are in
  `results/_tests/dyn_scalar_thr250_{speedup_map,large_results,suite_results}.csv`
  (`strip_dyn_scalar_thr250.py`); `run_b7_dynscalar.sh` re-ran large, `cells_b7_dynscalar.json`
  (19 entries × 5 depths) and suite under the lock. Row `cmd` shows the threshold used.
