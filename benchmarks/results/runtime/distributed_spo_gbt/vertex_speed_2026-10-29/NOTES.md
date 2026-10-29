# 2026-10-29 Vertex DGBT speed work — running notes

Worktree: /Users/ariellubonja/prog/randals_lab/oblique_forests/wt-colrep-speed (branch colrep-speed, from origin/column-replication dc568281).
EC2 build worktree: ~/wt/speed on ec2-3-92-30-82 (disk cache ~/.cache/bazel-disk, icx, --per_file_copt boringssl@-O0).

## Baseline (smoke28v3, n2-highmem-4 1+4, image train-ydf:colrep-34d64d3d-v3, logs in baseline_logs/)
- ozone (1847x72): cold cell 119.8 s (cache build 117.2 s), warm r0.10 2.78 s, r0.50 3.21 s
  (StartTraining 1.31 s, 2 checkpoints 1.42 s, 10 trees ~0.3 s)
- Bioresponse (3375x1776): cold 481 s (cache build 467 s: separate ~200 s + sort ~240 s),
  warm r0.10 16.8 s (StartTraining 12.6 s = workers re-load 577 cols x 3 objects each,
  2 checkpoints 1.7 s, 10 trees 3.1 s; FindSplits 40 ms/layer is real compute: P=F=1776
  projections per worker per node), r0.50 28.7 s.
- Reference accuracies: Bioresponse s1 r0.1 acc 0.6728723404255319 auc 0.7946306429548562;
  ozone s1 r0.5 acc 0.9409448818897638 auc 0.9432773109243682.

## Code reading: where the GCS ops are
- utils/filesystem_default_gcs.cc: `GetGCSClient()` built a NEW google::cloud::storage::Client per
  call (every FileExists/Rename/Open) -> no connection reuse, fresh credentials per op; writes were
  resumable uploads (2 HTTP calls) even for 13 KB objects.
- dataset_cache_worker.cc: every column file written to tmp/ then Rename (GCS: rewrite + delete),
  then RecursivelyDelete(tmp) per job; 10 I/O threads; sort jobs 5 per worker.
- dataset_cache.cc SortNumericalColumns: manager did a serial FileExists per column before dispatch
  and a serial SetBinaryProto per result -> 2 x 1776 serial GCS ops on Bioresponse.
- worker.cc Setup: DatasetCacheReader::Create per training (new worker object per manager uid);
  3 object reads per numerical column, 20 threads.
- Checkpoints: iteration 0 (ShouldCreateCheckpoint: iter 0 always) and final are unconditional;
  each ~10 serial GCS ops (worker shard write + rename, model.Save 5-6 files, checkpoint proto,
  snapshot).

## Job A = speed_base1 (base image = old wheel 34d64d3d + new train_cells.py, knobs off; 02:47 UTC)
Timestamps (manager clock) of the Bioresponse cold cell, 465.6 s cache build:
- 0.1 -> 10.1 s  first manager waits for worker pings (10 s sleep after one failed ping; once per job)
- 10.1 -> 45.4 s "Found 1777 column(s)" -> "Found 1 shard(s)": file::Match on the plain csv path
  lists the WHOLE BUCKET (GCS Match used an empty prefix when the path has no wildcard) -> 35 s,
  and the worker repeats it inside LoadVerticalDataset (another ~35 s inside the 116 s separate phase)
- 45.5 -> 161.8 s one worker separates 1777 columns (tmp write + rewrite + delete per column, 10 thr)
- 161.8 -> 222.7 s manager serial FileExists x 1776 before the first sort result (61 s)
- 222.7 -> 465.4 s receive loop: serial SetBinaryProto per column, 137 ms each (243 s)
ozone cold: 36.6 s listing + 76.5 s separate (73 cols) + 12.4 s sort = 125.7 s.
Warm cells: ozone 3.1-3.2 s (ready 0.9-1.3, 2 checkpoints 1.4-1.7, trees 0.3);
Bioresponse r0.1 16.3 s (ready 12.2), r0.5 27.7 s (ready 23.5); 10 trees 2.3 s.
