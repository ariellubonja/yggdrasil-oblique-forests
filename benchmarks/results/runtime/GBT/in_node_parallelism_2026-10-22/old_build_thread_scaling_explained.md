# Why the Old SPO-GBT build speeds up 1→4 threads on EPSILON/YouTube-8M but not HIGGS

(Final report of a Fable max-effort explanation agent, 2026-10-24 ~00:50 UTC, saved verbatim except that
scratchpad paths in the Commands section were shortened. Raw run logs, scripts and the two rebuilt Old binaries
are on the m7i in `ariel_notes/results/2026-10-22_spo_gbt_in_node/explain_old_thread_scaling/` (gitignored);
"this folder" below means that directory. Old = commit 23053344; New = branch in-node-parallelism-SPOGBT.)

Provenance: the box rebooted at ~00:16 UTC (uptime 3 min at session start), so the earlier `bin_base` binary, `run_sm_ds.sh` and `scratchpad/runs/` were gone. I rebuilt commit 23053344 in a detached worktree (`git worktree add … 23053344`, removed afterwards): plain `-c opt` was a 100 % disk-cache hit (846/846 compile actions ⇒ identical build inputs to the earlier bin_base) → `scratchpad/explain/bin_old/`; `--config=coarse_chrono_profile` → `scratchpad/explain/bin_old_chrono/`. Same command line as the table; all numbers below are re-measured and replicate the caller's table within 2 % (EPSILON 11.88/3.80/3.14 s at 1/4/8 thr vs 11.8/3.80/3.12; HIGGS 55.53/54.97 vs 56.6/55.9; YT8M 100.22/33.75 vs 100.2/33.8). Chrono overhead ≤0.5 % of the block. Main repo working tree untouched (only the pre-existing `training.cc` diff). apt timers verified masked.

## 1. Plain-language explanation (Old build)

1. **What one GBT iteration does.** GBT grows one tree at a time on the main thread (gbt.cc:1566-1568 → `decision_tree::Train`). Per node (NodeTrain, training.cc:5330): sum the gradients/hessians of the node's rows for the leaf value (loss_utils.cc:84-118), search for the best split (training.cc:5451-5455), partition the node's rows in place (training.cc:5476-5481). After the tree: walk every row through it to update predictions (gbt.cc:1623-1625, loss_utils.cc:214-229), recompute gradients (gbt.cc:1491-1494). Depth 6 ⇒ 31 split searches per tree, each depth re-processing all rows (no subsampling: `subsample` default 1 ⇒ iota, gbt.cc:3382-3386); depth-6 nodes are leaves.
2. **Where the time goes at 1 thread (measured).** The sparse-oblique split search ("oblique") is EPSILON 94 %, YT8M 93 %, HIGGS 71 % of the training block. Everything else is serial bookkeeping: updating predictions (0.39 / 3.9 / 8.8 s), partitioning rows (0.21 / 1.9 / 4.4 s), leaf sums (0.04 / 0.6 / 1.9 s). Gradient + training-loss computation is <1 % (EPSILON 0.024 s, YT8M 0.20 s, HIGGS 0.54 s).
3. **Inside the oblique search**: ApplyProjection (gathering the rows' feature values and forming Σw·x) is 24-29 %; the Exact finder (sort the projected values + scan thresholds, VQSort path `FindBestSplitFlatHighway`, splitter_scanner.h:1862-1864/1806, reached via oblique.cc:577 → training.cc:3144) is 71-77 %. Cost is uniform: 12.5 / 12.6 / 14.2 ns per (row × projection) on EPSILON / HIGGS / YT8M, i.e. work per tree ≈ 5 depths × rows × P.
4. **The "single core per node" belief is only half right.** With >1 thread the Old build splits a node's P projections into `jobs = min(num_threads, ceil(P/10))` worker jobs (training.cc:1742-1750, `min_projections_per_request = 10`), each job of `ceil(P/jobs)` projections, the last job the remainder (training.cc:1865-1874), submitted to the "SplitFinder" worker pool while the tree thread blocks in `GetResult()` (training.cc:1909). The axis-aligned jobs are skipped entirely (`AxisAlignedJobsAreNoop`, training.cc:1796-1802). So: HIGGS P=6 → ceil(6/10)=1 job at every thread count (truly single-core); EPSILON P=45 → up to 5 jobs; YT8M P=34 → up to 4 jobs.
5. **The 1→4 drop is exactly that split.** EPSILON at 4 threads: jobs of [12,12,12,9] projections ⇒ the node takes as long as a 12-projection job: ideal 45/12 = 3.75× on the oblique part; measured oblique 11.22 s → 3.14 s wait (3.57×), while total worker CPU stays 11.33 s (work conserved, nothing duplicated except the per-job setup, 0.02 → 0.10 s, oblique.cc:239-250 `ExtractLabels` per job). YT8M: [9,9,9,7] ⇒ ideal 3.78×; measured 94.1 → 27.1 s (3.47×; worker CPU 99.7 s, +6 % from 4 concurrent gathers and 4× setup). HIGGS: [6] ⇒ 1×; measured 39.8 → 40.2 s.
6. **Why it stops where it stops: the job cap.** EPSILON can never have more than 5 jobs: 5, 8, 16, 24, 48 threads produce byte-identical models (same md5 of `nodes-00000-of-00001`, because each job's RNG seed is drawn per job, training.cc:1854/1327) and the same time (3.08-3.27 s); the 5→8-thread step is only the 4→5-job step ([12,12,12,9] → [9,9,9,9,9]: 3.80 → 3.08 s). YT8M caps at 4 jobs ⇒ flat from 4 threads (caller: 33.8/33.8/34.1/34.4/34.2). HIGGS caps at 1 job ⇒ flat everywhere.
7. **The serial floor caps the whole-block speedup.** Block − oblique = EPSILON 0.66 s, YT8M 6.8 s, HIGGS 15.5 s at 4 threads, none of it parallel (prediction update loops over all rows on the main thread, loss_utils.cc:214-229; row partition and leaf sums run on the tree thread). Hence 4-thread speedups 3.13× / 2.97× / 1.01× vs oblique-only 3.57× / 3.47× / 0.99×.
8. **Gradients and loss ARE parallel but irrelevant.** `GBTLossThreadpool` (gbt.cc:1424-1429, `ConcurrentForLoop` in loss_imp_binomial.cc:165-175, 247-265) cuts HIGGS update_gradients 0.31 → 0.08 s and training-loss eval 0.23 → 0.06 s: 0.4 s of 55.8 s. It is not where the N-thread speedup comes from.
9. **Beyond the cap, more threads cost a little.** EPSILON 8 → 48 threads: worker CPU 11.54 → 11.67 s (+1 %), manager wait 2.50 → 2.57 s, prediction update 0.37 → 0.41 s ⇒ block 3.14 → 3.27 s (+4 %). The caller's 3.38 s at 24 threads is in the same direction; on this boot the plain binary at exactly 24 threads is reproducibly 4.59 s (3 runs: 4.586/4.604/4.585; 23 thr 3.23, 25 thr 3.39, chrono binary at 24 thr 3.25; model identical; +2 s user CPU ⇒ slower worker compute). Unexplained layout/thread-count artefact, not the mechanism; `perf stat` unavailable (`perf_event_paranoid=4`).
10. **Consistency with the New build (derived from the caller's New numbers, not re-run).** New uses `min(threads, P)` jobs ⇒ 6/45/34 jobs at 48 threads. Floor + oblique/jobs gives HIGGS 15.5 + 39.8/6 = 22.1 s (measured 28.9), EPSILON 0.66 + 11.2/45 = 0.91 s (1.46), YT8M 6.8 + 94.1/34 = 9.6 s (14.4): the remaining gap is per-job overhead/imbalance of 1-projection jobs, and the serial floor is now the dominant term on HIGGS.

## 2. Is the 1-thread run a different code path? Yes; it costs ≈ nothing.

- `CreateSplitterFinderStreamProcessor` returns `nullptr` for `num_threads <= 1` (training.cc:5259-5262); `FindBestConditionManager` then takes `FindBestConditionSingleThreadManager` (training.cc:2035-2046), which runs the entire oblique search inline on the tree thread under `kObliqueSplitSearch` (training.cc:1494-1505) with the tree's RNG. With ≥2 threads `FindBestConditionConcurrentManager` builds jobs seeded by `(*random)()` (training.cc:1854), the worker re-seeds its own engine (training.cc:1327) ⇒ a different model even with a single job (HIGGS 1-thr vs 4-thr model md5 differ). `GBTLossThreadpool` is also only created for >1 thread (gbt.cc:1424-1429).
- Cost of the hand-off, isolated on HIGGS (1 job either way): inline 39.78 s vs worker 40.17 s (+1.0 %, ≈0.6 ms per node incl. cross-core cache effects); manager setup/submit/process bookkeeping 0.003 s total over 630 nodes. Savings from the loss pool −0.40 s. Net ≈ 0: plain 55.53 s (1 thr) vs 54.97 s (4 thr). Total CPU is conserved: `/usr/bin/time` user 75.00 s (1 thr) vs 74.97 s (4 thr) on HIGGS (includes the always-48-thread CSV parser), 57.4/57.3/57.5 s on EPSILON at 1/4/8 thr.

## 3. Evidence

Training-block seconds (Old, this session; T = threads; "jobs" = Old oblique jobs per node):

| Dataset (rows, P) | T1 | T2 | T3 | T4 | T5 | T8 | T12 | T16 | T20 | T23 | T24 | T25 | T32 | T48 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EPSILON (400k, 45; jobs 1/2/3/4/5/5…) | 11.88 | 6.45 | 4.57 | 3.80 | 3.08 | 3.14, 3.14 | 3.19 | 3.20, 3.21 | 3.23 | 3.23 | 4.59 ×3 | 3.39 | 3.25 | 3.27 |
| HIGGS (10.5M, 6; 1 job always) | 55.53 | | | 54.97 | | | | | | | | | | |
| YT8M (3.89M, 34; jobs 1/…/4 cap) | 100.22 | | | 33.75 | | | | | | | | | | |

Ideal oblique speedup from the job split (P / max projections per job): EPSILON T2 1.96×, T3 3.00×, T4 3.75×, T≥5 5.00×; YT8M T4+ 3.78×; HIGGS 1.00×. Measured block speedups: EPSILON 1.84 / 2.60 / 3.13 / 3.85 (T5) / 3.79 (T8)×; YT8M 2.97×; HIGGS 1.01×.

Chrono breakdown (seconds; "oblique" = `ObliqueSplitSearch` at 1 thr, `SplitManagerWait` at N thr; "worker CPU" summed over SplitFinder threads):

| Run | block | oblique (share) | worker CPU = AP + finder + setup | split rows | leaf | upd_pred | upd_grad | loss eval |
|---|---|---|---|---|---|---|---|---|
| EPSILON T1 | 11.91 | 11.22 (94 %) = AP 3.22 + finder 7.96 + setup 0.02 | – | 0.21 | 0.04 | 0.39 | 0.012 | 0.012 |
| EPSILON T4 | 3.80 | 3.14 (83 %) | 11.33 = 3.19 + 8.02 + 0.10 | 0.21 | 0.04 | 0.38 | 0.006 | 0.007 |
| EPSILON T8 | 3.16 | 2.50 (79 %) | 11.54 = 3.28 + 8.08 + 0.15 | 0.21 | 0.04 | 0.37 | 0.006 | 0.007 |
| EPSILON T16 / T24 / T48 | 3.23 / 3.25 / 3.27 | 2.56 / 2.57 / 2.57 | 11.57 / 11.63 / 11.67 | 0.21 | 0.04 | 0.38 / 0.40 / 0.41 | 0.006 | 0.007 |
| HIGGS T1 | 55.83 | 39.78 (71 %) = 9.58 + 28.68 + 1.48 | – | 4.38 | 1.89 | 8.82 | 0.311 | 0.232 |
| HIGGS T4 | 55.67 | 40.17 (72 %) | 40.11 = 9.60 + 28.73 + 1.71 | 4.35 | 1.80 | 8.86 | 0.078 | 0.063 |
| YT8M T1 | 100.73 | 94.07 (93 %) = 20.96 + 72.79 + 0.30 | – | 1.86 | 0.57 | 3.93 | 0.115 | 0.089 |
| YT8M T4 | 33.87 | 27.07 (80 %) | 99.72 = 21.30 + 76.94 + 1.45 | 1.84 | 0.76 | 4.02 | 0.029 | 0.027 |

Ideal wait from measured worker CPU: EPSILON T4 12/45 × 11.33 = 3.02 s (measured 3.14, +4 %), T8 9/45 × 11.54 = 2.31 s (2.50, +8 % imbalance/hand-off); YT8M T4 9/34 × 99.72 = 26.4 s (27.07, +2.5 %).

Other facts: models — EPSILON T5/T8/T16/T24/T48 `nodes-00000-of-00001` md5 identical (d5df7ca0…), T1..T4 all different; plain and chrono binaries give identical models at the same T. Per depth (EPSILON tree 0, 1 thr): depths 1-5 each 0.20-0.29 s on all 400k rows, depth 6 = leaves (0 s). Dataset load (outside the timer): EPSILON cold ≈2 min / warm 1.8 s, HIGGS cold 58 s / warm ≈1 s, YT8M cold 435 s / warm 8.0 s.

## Commands

Build (in the worktree, now removed): `set +u; source /opt/intel/oneapi/setvars.sh; cd <scratchpad>/explain/wt_old; bazel build -c opt --disk_cache=~/.cache/bazel-disk //examples:train_oblique_forest` then the same with `--config=coarse_chrono_profile` (script `build_old.sh`, log `build_old.log`, both in this folder; binaries in `bins/`).

Each run (`run_old.sh BINSET DATASET THREADS 10`), one at a time:
`cd /home/ubuntu/yggdrasil-oblique-forests && /usr/bin/time -v <BIN> --input_mode csv --train_csv <CSV> --label_col <L> --ensemble_method Boosting --num_trees 10 --tree_depth 6 --feature_split_type Oblique --numerical_split_type Exact --num_threads <T> --seed 1 --model_out_dir <run dir>/model`
with HIGGS `benchmarks/data/HIGGS_train_10500k.csv`/`class`, EPSILON `benchmarks/data/epsilon_normalized_train.csv`/`label`, YT8M `benchmarks/data/youtube8m/youtube8m_video_train.csv`/`class`. Logs: `runs/sm_<DS>_<bin_old|bin_old_chrono>_t<T>_n10/log.txt` (re-runs suffixed `_run2`, `_run3`), one-line summaries in `runs/summary.txt`. Analysis: `summarize.py` (timing table + job split), `parse_chrono.py <log>` (per-scope sums), `shares.py` (share table above). Harness facts used: `--num_threads` → `deploy_config.num_threads` (train_oblique_forest.cc:1237-1251), Boosting sets `validation_set_ratio=0`, early stopping NONE (1276-1289), depth 6 / `min_examples` 5 (1300-1308), `Exact` → `NumericalSplit::EXACT` (1398-1400). Timer = gbt.cc:1181 → 1825.

No repo files were modified; no commits.
