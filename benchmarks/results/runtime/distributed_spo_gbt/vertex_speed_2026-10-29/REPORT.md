# Vertex distributed GBT: wall-time cuts (2026-10-29)

Branch `colrep-speed` (from `column-replication` @ dc568281), pushed to origin; commits 5ae9026f, 2f1fa5ba,
6067d48c, fc60c458, de84ad27, 1d16d3e6, c988e2d0, 76c81faa. Images (`us-east1-docker.pkg.dev/ydf-distributed-spo-gbt/ydf-repo/`):
- measured: wheel `ydf-wheel:speed-de84ad27`, train image `train-ydf:speed-de84ad27` (C++ at de84ad27, scripts at 1d16d3e6);
- final (= HEAD 76c81faa): `ydf-wheel:speed-76c81faa`, `train-ydf:speed-76c81faa`. Differs from the measured one only by the
  verification fixes below (a GCS-writer error path and the Python lock protocol), none on the measured paths.
- baseline: `train-ydf:speed-base-34d64d3d` = the old wheel (34d64d3d) + the new `train_cells.py` with every knob off, so
  the harness is held constant and only the C++/knobs differ.

Setup for every number below: `n2-highmem-4`, 1 manager + 4 workers (real GRPC cluster, one VM per node), seed 1,
10 trees, depth 6, P = F projections per worker per node, CC18 fold-0 ozone (2280 x 72) and Bioresponse
(3375 x 1776) from `gs://ariel_ydf_bucket/cc18_fold0`. Cells: `gs://ariel_ydf_bucket/runs/speed_{base1,cacheioD,allC,finalF,sweepG}`,
copied to `vertex/<run>/cells/`; `summary_cells.tsv` = all cells; `vertex/*/logs/` = manager timelines.
Phase columns (parsed from the manager log, manager clock): `cache_s` = "Dataset cache created in",
`ready_s` = "All workers ready to train in" (worker Setup = column loads), `ckpt_s` = sum of "Checkpoint created in",
`loop_s` = "Start tree training" -> "Training done" minus the checkpoints inside it (= the 10-tree loop).

## Headline (reference cells; before = speed_base1, after = speed_finalF, every lever on)

| cell | before: cell wall | after | x | cache build before -> after | workers-ready | checkpoints | 10-tree loop |
|---|---|---|---|---|---|---|---|
| ozone r=0.5 (warm) | 3.2 s | **0.48 s** | 6.7x | - | 1.3 -> 0.03 s | 1.4 -> 0 | 0.33 -> 0.35 s (unchanged) |
| ozone r=0.1 cold (incl. cache build) | 128.9 s | **2.3 s** (base knobs) / ~1.6 s | 56x | 125.7 -> **0.95 s** (132x) | | | |
| Bioresponse r=0.1 (warm) | 16.3 s | **2.46 s** | 6.6x | - | 12.2 -> 0.03 s | 1.5 -> 0 | 2.29 -> 2.31 s (unchanged) |
| Bioresponse r=0.5 (warm) | 27.7 s | **2.55 s** | 10.9x | - | 23.5 -> 0.03 s | 1.7 -> 0 | 2.4 -> 2.4 s |
| Bioresponse r=0.1 cold | 481.9 s | **17.8 s** (base knobs) / ~14 s | 27x | 465.6 -> **11.7 s** (40x) | | | |

Bit-identity: every cell of every run reproduces the reference metrics exactly: Bioresponse s1 r0.1
acc 0.6728723404255319 / auc 0.7946306429548562, ozone s1 r0.5 acc 0.9409448818897638 / auc 0.9432773109243682
(and r0.1 / r0.5 of the other dataset agree across all five runs and all knob variants). In-process A/B on the EC2 box
(old binary 34d64d3d vs new, `train_distributed_gbt`, 4 workers, ozone + Bioresponse, r=0.1): model files
(`nodes-*`, `gradient_boosted_trees_header.pb`) byte-identical. No lever changes model semantics.

## Lever table

| lever | what changed | before -> after (reference cells) | identity | patch |
|---|---|---|---|---|
| 1. GCS `Match` listed the whole bucket | `filesystem_default_gcs.cc Match()`: a path without wildcard used an EMPTY list prefix, so every dataset-path expansion (`ExpandInputShards`, called by the manager and 2x by the worker that loads the CSV) listed the entire bucket: 35 s per call, growing with the bucket. Prefix = whole object name now. | cold build Bioresponse 121.4 -> 11.7 s, ozone 110.2 -> 0.95 s (allC -> finalF; ~105 s per cold dataset) | same | de84ad27 |
| 2. One GCS client per process | `GetGCSClient()` built a new `google::cloud::storage::Client` per call (no connection reuse, fresh credentials per op). Now one static client, pool 64; `YDF_GCS_CLIENT_PER_OP=1` restores. Small objects (<= 32 MiB) buffered and sent as one `InsertObject` instead of a 2-call resumable upload; `YDF_GCS_STREAMING_WRITES=1` restores. | workers-ready Bioresponse r0.1 12.2 -> 3.0 s, r0.5 23.5 -> 3.6 s; checkpoint 0.75 -> 0.32 s each; cache build 160.8 -> 121.4 s (cacheioD -> allC, both still paying the listing bug) | same | 5ae9026f |
| 3. Cache creation I/O structure | worker: write column files straight to their final `gs://` path (object writes are atomic; tmp + Rewrite + Delete + prefix-delete per file before), 32 I/O threads on object stores; manager: the per-column `FileExists` (61 s serial before the first sort result) and per-result `SetBinaryProto` resume markers (243 s serial) run in 32/16-thread pools; sort jobs per worker 5 -> clamp(2 GiB / column bytes, 5, 20). `YDF_CACHE_LEGACY_IO=1` restores all three. Local filesystems keep tmp + rename. | cache build Bioresponse 465.6 -> 160.8 s, ozone 125.7 -> 114.4 s with the legacy client (base1 -> cacheioD); sort phase 304 -> 7 s, separate phase 116 -> 4 s with everything on | same (cache_test / reader_test pass; same metrics) | 5ae9026f, 2f1fa5ba |
| 4. Checkpoints off | proto `create_checkpoints` (default true): false skips the iteration-0 and final checkpoints (~10 serial GCS ops each). `train_cells.py --no_checkpoints`. | 2 checkpoints 1.4-1.7 s -> 0 (after lever 2: 0.6-0.7 s -> 0) | same | 5ae9026f |
| 5. Workers keep columns in memory | proto `reuse_dataset_cache_in_workers` (default false): the worker's `DatasetCacheReader` takes columns from a process-wide pool keyed by (cache path, cache metadata bytes) and loads only missing ones, so the level sweep (every level changes each worker's column set) reloads nothing once a column was seen; identical (path, options, features) reuses the reader object. `train_cells.py --reuse_reader`. | workers-ready 3.0 / 3.6 s -> 0.03 s (second visit), partial hits in between (r0.5 after r0.1: 2.1 s instead of 3.6) | same (reader_test ColumnPoolAndSharedReader) | 6067d48c |
| 6. One cache per dataset shared across jobs/seeds | proto `dataset_cache_path` (default = `<work dir>/dataset_cache`); `train_cells.py --cache_root gs://...`: cache at `<root>/<dataset>/<sha1(key version, csv path, label, feature list, discretization)>/dataset_cache`, built once under a GCS lock object (`if_generation_match=0`), other jobs poll the `done` marker (stale lock 45 min). Work dirs stay per training (`resume_training=False`), no `rm checkpoint` dance. | saves one cold build per dataset per extra job: with 5 seed jobs 4 x 34 builds (now ~1-12 s each; was 2-8 min each) | same cache bytes | 5ae9026f |
| 7. Worker readiness ping | GRPC manager slept 10 s after a failed first ping (first manager of a job). 1 s now. | 10 s once per job | same | de84ad27 |
| 8. Harness | one `fs.find` listing instead of one HEAD per cell per pass (374 HEADs ~ 40 s/pass); phase timings + barrier CSV per cell; in-job knob variants (`--variants`) and repeats for A/B. | | | 5ae9026f, 6067d48c |

Per-tree time: unchanged by design (nothing in the tree loop was touched). The barrier recorder on the final cells
shows where the loop time goes: Bioresponse 10 trees = 2.3 s, FindSplits 87 % (worker compute: P = 1776 projections
per worker per open node, 3375 rows), transport (`comm_residual`) 6.8 %, manager serial 1.3 %, straggler idle 15 % of
worker time; ozone 10 trees = 0.33 s, 22 rounds per tree at 0.7-2.7 ms each, transport 38-42 % of the loop
(= 0.5-0.6 ms per round, the GRPC floor between VMs). A manager-relay redesign of ShareSplits would save at most
~0.1 s per cell; not pursued.

## Where the baseline time went (manager-clock timeline of the Bioresponse cold cell, speed_base1)

| t (s) | phase | cause |
|---|---|---|
| 0 -> 10 | first manager waits for the workers | failed first ping + 10 s sleep (lever 7) |
| 10 -> 45 | "Found 1777 column(s)" -> "Found 1 shard(s)" | `Match` lists the whole bucket (lever 1) |
| 45 -> 162 | one worker separates 1777 columns | 2 more bucket listings inside `LoadVerticalDataset` + tmp/rename/delete per column with a fresh client each (levers 1-3) |
| 162 -> 223 | "Start sorting" -> first result | manager's serial `FileExists` x 1776 (lever 3) |
| 223 -> 465 | receive loop | serial `SetBinaryProto` per column, 137 ms each (levers 2-3) |
| 466 -> 478 | workers load 577 columns x 3 objects | fresh client per read (levers 2, 5) |
| 478 -> 482 | 2 checkpoints + 10 trees | levers 4 (2.3 s is the tree loop) |

After all levers (speed_finalF, same cell): listing 0.03 s, separate 4.2 s, sort 7.3 s (cache 11.7 s), ready 2.9 s
(first visit) / 0.03 s (later), trees 2.3 s.

## Ablation runs (all in `summary_cells.tsv`)

| run | wheel | env / knobs | Bioresponse cache | ozone cache | Bioresponse warm r0.1 / r0.5 | ozone warm r0.1 / r0.5 |
|---|---|---|---|---|---|---|
| base1 | 34d64d3d (old) | - | 465.6 s | 125.7 s | 16.3 / 27.7 s | 3.1 / 3.2 s |
| cacheioD | 2f1fa5ba | legacy client (`YDF_GCS_CLIENT_PER_OP=1 YDF_GCS_STREAMING_WRITES=1`), new cache I/O | 160.8 s | 114.4 s | 16.5 / 27.7 s | 2.8 / 3.1 s |
| allC base | fc60c458 | shared client + new cache I/O, knobs off | 121.4 s | 110.2 s | 6.2 / 6.7-7.0 s | 1.7 / 1.6 s |
| allC nockpt | | + create_checkpoints=false | | | 4.5-6.1 / 6.1-6.5 s | 0.7 / 0.9 s |
| allC reuse (2nd visit) | | + column pool | | | 3.1 / 3.2 s | 1.05 / 1.1 s |
| allC all (2nd visit) | | both | | | 2.4-2.6 / 2.5-2.6 s | 0.44-0.48 / 0.43-0.51 s |
| finalF base | de84ad27 | + Match fix, 1 s ping | **11.7 s** | **0.95 s** | 6.1 / 6.7-7.0 s | 1.3 / 1.4-1.5 s |
| finalF all | | | | | **2.46 / 2.55 s** | **0.48 / 0.48 s** |

## Projection for the 374-cell column-replication job (34 datasets x 11 levels, one seed)

Before: ~1 h of cache builds + ~1 h of warm cells (2 h per job). After: cache builds 34 x (1-12 s) ~ 3 min
(and only in the first job when `--cache_root` is shared), warm cells 374 x (0.5-2.5 s) ~ 5-15 min; the tree loops
(real compute) are now most of it. `sweepG` (below) measures the full 11-level sweep on the two reference datasets.

## Full 11-level sweep on the two reference datasets (speed_sweepG, every lever on, one job)

Manager clock from `pass 0` to `ALL_CELLS_DONE`: **73 s for 22 cells** (Bioresponse 11 levels 62 s including the CSV
download + dataspec and the 12.9 s cold build; ozone 11 levels 11 s including the 0.95 s build). Per cell
(`train_s`): Bioresponse 17.6 s cold then 3.1-4.0 s per level (the pool fills as levels add replicated columns:
ready 2.1 -> 1.6 -> 0.6-1.0 s; the 10-tree loop is 2.3-2.5 s of it), ozone 1.7 s cold then 0.56-0.99 s. Baseline
projection for the same 22 cells from speed_base1 (482 + 10 x 16-45 s; 129 + 10 x 3.2 s): ~15 min, i.e. ~12x.
Metrics per level identical to the 2026-10-28 runs (e.g. Bioresponse r=0 acc 0.7393617 / auc 0.7944881).

## How to run it

    # one-off: pre-build the shared caches (optional; the first job to reach a dataset builds it anyway)
    IMG=us-east1-docker.pkg.dev/ydf-distributed-spo-gbt/ydf-repo/train-ydf:speed-76c81faa
    python column_replication/run_vertex_colrep.py --image $IMG --git 76c81faa --job_prefix prebuild \
        --seeds 1 --levels 0 --num_trees 1 --extra_args "--cache_root gs://ariel_ydf_bucket/work/cache_cc18 --no_checkpoints"
    # the sweep (5 seed jobs share the caches; a job that arrives first builds, the others wait on the done marker)
    python column_replication/run_vertex_colrep.py --image $IMG --git 76c81faa \
        --extra_args "--cache_root gs://ariel_ydf_bucket/work/cache_cc18 --no_checkpoints --reuse_reader"

`--cache_root` keys the cache by (csv path, label, feature list, discretization, key version `v1`): bump
`CACHE_KEY_VERSION` in `train_cells.py` if the cache format changes. Without `--cache_root` the old per-run
work-dir cache (resume_training + `rm checkpoint`) is used. `--barrier_csv` adds `barriers.csv` per cell
(`summarize_barriers.py`). `summarize_speed.py <run dirs>` prints the per-cell phase table.

## Design decisions

- One process-wide `google::cloud::storage::Client` (copies share pool + credential cache), pool size 64 (>= the 32-thread
  export pool); per-op client kept behind `YDF_GCS_CLIENT_PER_OP=1` for A/B only.
- Object writes <= 32 MiB are buffered and sent as one `InsertObject`; larger ones stream (resumable) exactly as before;
  the cutoff keeps the per-writer buffer bounded (32 threads x 32 MiB worst case).
- `Match` without wildcard uses the whole object name as list prefix; the regex filter after it is unchanged.
- Direct final-path writes only when the output is `gs://` (atomic object creation); local filesystems keep tmp + rename,
  so upstream semantics on disk are untouched. Re-sent jobs overwrite identical files, as the rename did.
- Manager-side existence checks and resume-marker writes keep their resume semantics (markers still written per column),
  only their latency is taken off the critical path (pools of 32 / 16); the receive loop never blocks on a write.
- Sort-job parallelism per worker = clamp(2 GiB / (rows x 8 B), 5, 20): 20 for anything below 13M rows, the historical 5
  for huge columns (RAM bound).
- `create_checkpoints` defaults to true (upstream behaviour); the harness turns it off. Without checkpoints a worker
  restart fails the training (CancelledError) instead of resuming; the harness's retry pass covers that.
- `dataset_cache_path` is an opt-in override; the cache dir is only ever read once its `done` marker exists, and the
  builder is serialized by an atomic lock object with a stale-lock takeover (45 min) so a dead builder cannot block forever.
- `reuse_dataset_cache_in_workers` defaults to false. The pool is keyed by cache path + serialized cache metadata and
  holds one dataset at a time (identity change drops the previous columns before loading: peak = max, not sum).
  Reader-object reuse needs identical path/options/features and no non-blocking load in flight; ranking (which releases
  groups in place) never reuses. Feature-loading time is reported as 0 on reuse so the load balancer is not misled.
- Levers are isolated: each env gate / proto knob is independently switchable; ablations ran on the same machine type,
  cluster shape, seed, datasets and image recipe; the baseline uses the old wheel with the new harness.
- Nothing in the tree loop, split finders, sampling or merge order was touched; identity is checked per cell by
  comparing accuracy/AUC against the reference values and in-process by comparing model files byte for byte.

## Verification pass (independent Fable reviewer, max effort; Opus subagents for the empirical checks)

VERDICT PASS on the C++ (no model-semantics change). Re-done by the reviewer: bazel tests on the EC2 box (cache tests pass;
the oblique test's `SparseOblique_1worker` serving-engine mismatch and a flaky `SparseOblique_5workers` fail identically on
the base commit); an 18-way in-process A/B (old binary vs new vs new+`YDF_CACHE_LEGACY_IO=1` vs repeat; ozone/Bioresponse;
discretized and presorted cache paths): model files and dataset caches byte-identical, resume paths (missing `done`,
missing per-column markers, legacy gate) identical; all 94 Vertex cells give one (acc, auc, logloss) per (dataset, level)
across the 9 configurations, equal to the 2026-10-28 reference. Three findings, all fixed in 76c81faa:
(1) `CacheLock` probed the lock with a `Blob` whose generation got pinned by `reload()`, so after a takeover it could spin
without sleeping or timing out -> a fresh `Blob` per probe, 1 s retry when the lock vanished (8-scenario offline test
`test_cachelock_offline.py` here: acquire, done short-circuit, wait, stale takeover, replacement mid-wait, vanish-and-retry, 2x45 min
timeout, release never deletes another holder's lock); (2) the lock could leak if the learner constructor threw before the
`try/finally` -> the whole cell after acquisition is under the `finally`; (3) a failed overflow-to-streaming `Write` left
the buffer, so the `OutputFileCloser` destructor's `Close()` would have uploaded a truncated object -> a failed write poisons
the stream (`Close()` returns an error, uploads nothing). Not re-measured: identical hot paths.

## Tried / considered, did not pay or not pursued

- Reader-object reuse alone (first version): useless for a level sweep because each level changes every worker's
  column set; superseded by the per-column pool (kept as a fast path).
- Sharding the input CSV so all workers separate columns: unnecessary once the separate phase is 4 s for 1777 columns,
  and it would multiply the raw-shard reads at every column load.
- Fewer files per column / several columns per object: would change the cache format; the per-file cost is now ~25 ms
  and parallel, so the format was left alone.
- Per-round GRPC redesign (manager relaying split evaluations instead of the 2-round peer plan): transport is
  0.5-1 ms per round, <= 0.1-0.16 s per 10-tree cell; not worth a protocol change.
- VQSort in the cache sort job: the sort phase is I/O bound (3375-row columns); std::sort there is not a hot path.
- Smaller/cheaper VMs for tiny datasets: not measured (machine type is the user's decision; one type per comparison).
- Worker-side C++ logs: `ydf.verbose(2)` in the worker process does not surface them in Cloud Logging (still invisible);
  the manager timeline was enough.
