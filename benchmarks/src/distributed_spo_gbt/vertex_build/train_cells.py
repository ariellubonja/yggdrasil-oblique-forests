#!/usr/bin/env python3
"""Many small distributed-GBT cells in one Vertex CustomJob (column-replication study).

Pool 0 = manager, pool 1 = W workers (real GRPC distribution). The manager loops over
(dataset, replication) cells with one seed, reusing the workers; a cell whose eval.json
already exists in --out is skipped, so resubmitting the job resumes it.

Speed knobs (all off = the 2026-10-28 behaviour; see ariel_notes/results/2026-10-29_vertex_speed):
  --cache_root     share one dataset cache per dataset across jobs/seeds (lock + done marker)
  checkpoints are always off (create_checkpoints=false, CLAUDE.md 2026-10-29); --checkpoints re-enables (A/B)
  --reuse_reader   workers keep the loaded columns between trainings
  --barrier_csv    record the manager's per-round barrier CSV next to eval.json
"""
import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import traceback

import gcsfs
import ydf
from yggdrasil_decision_forests.learner import abstract_learner_pb2
from yggdrasil_decision_forests.learner.distributed_gradient_boosted_trees import (
    distributed_gradient_boosted_trees_pb2 as dgbt_pb2,
)

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="gs:// root with <dataset>/{train,test}.csv")
parser.add_argument("--out", required=True, help="gs:// run dir; cells at <out>/<dataset>/r<RR>/")
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--levels", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
parser.add_argument("--datasets", default="", help="comma filter; empty = all under --data")
parser.add_argument("--num_trees", type=int, default=10)
parser.add_argument("--max_depth", type=int, default=6)
parser.add_argument("--label", default="label")
parser.add_argument("--meta", default="{}")
parser.add_argument("--cache_root", default="",
                    help="gs:// dir of shared dataset caches; empty = per-run cache under /work/")
parser.add_argument("--checkpoints", action="store_true", help="keep YDF checkpoints (A/B only)")
parser.add_argument("--no_checkpoints", action="store_true", help="no-op: checkpoints are off by default")
parser.add_argument("--reuse_reader", action="store_true")
parser.add_argument("--barrier_csv", action="store_true")
parser.add_argument("--variants", default="",
                    help="comma list of knob sets run per (dataset, level) for in-job A/B: "
                         "base | nockpt | reuse | all; cells land in r<RR>_<variant>/")
parser.add_argument("--repeat", type=int, default=1,
                    help="run every cell this many times (timing); repeats land in r<RR>_rep<k>/")
parser.add_argument("--force", action="store_true", help="rerun cells whose eval.json exists")
args = parser.parse_args()
args.data, args.out = args.data.rstrip("/"), args.out.rstrip("/")

CACHE_KEY_VERSION = "v1"  # bump when the cache format or its config changes
LOCK_STALE_S = 45 * 60

# C++ log lines that bound the phases (manager clock, first match wins).
PHASES = {
    "cache_begin": "Create dataset cache in",
    "cache_created": "Dataset cache created in",
    "workers_ready": "All workers ready to train",
    "tree_training_start": "Start tree training",
    "training_done": "Training done",
}


class LogTee:
  """Tees fd 2 (the C++ logs) into a buffer, timestamping every line on the manager clock."""

  def __init__(self):
    self.lines = []
    self.marks = {}

  def __enter__(self):
    self.t0 = time.time()
    sys.stderr.flush()
    self.saved = os.dup(2)
    r, w = os.pipe()
    os.dup2(w, 2)
    os.close(w)
    self.thread = threading.Thread(target=self._pump, args=(r,), daemon=True)
    self.thread.start()
    return self

  def _pump(self, r):
    with os.fdopen(r, "r", errors="replace") as f:
      for line in f:
        t = time.time() - self.t0
        for key, pat in PHASES.items():
          if key not in self.marks and pat in line:
            self.marks[key] = round(t, 3)
        self.lines.append(f"[{t:9.3f}] {line}")
        os.write(self.saved, line.encode(errors="replace"))

  def __exit__(self, *exc):
    sys.stderr.flush()
    os.dup2(self.saved, 2)
    self.thread.join(timeout=30)
    os.close(self.saved)

  def text(self):
    return "".join(self.lines)


def parse_duration(s):
  """absl::Duration text ("7m46.98s", "918.987ms", "1.06s", "12us") -> seconds."""
  total, unit = 0.0, {"h": 3600, "m": 60, "s": 1, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
  for num, u in re.findall(r"([0-9.]+)(h|ms|us|ns|m|s)", s):
    total += float(num) * unit[u]
  return total


def phase_summary(tee):
  """Per-phase seconds from the C++ log: cache build, worker setup, checkpoints, tree loop."""
  txt, m = tee.text(), tee.marks
  out = {"phase_marks_s": dict(m)}
  cache = re.search(r"Dataset cache created in ([0-9a-z.]+)", txt)
  out["cache_build_s"] = round(parse_duration(cache.group(1)), 3) if cache else 0.0
  ready = re.search(r"All workers ready to train in ([0-9a-z.]+)", txt)
  out["workers_ready_s"] = round(parse_duration(ready.group(1)), 3) if ready else None
  cps = re.findall(r"Checkpoint created in ([0-9a-z.]+)", txt)
  out["checkpoints_s"] = round(sum(parse_duration(c) for c in cps), 3)
  out["num_checkpoints"] = len(cps)
  if "tree_training_start" in m and "training_done" in m:
    out["tree_training_s"] = round(m["training_done"] - m["tree_training_start"], 3)
  return out


_DS_CACHE = {}


def dataset_ctx(fs, dataset):
  """Local CSV copies + dataspec, computed once per dataset (the cells differ only in r)."""
  if dataset not in _DS_CACHE:
    loc = f"/tmp/{dataset}"
    os.makedirs(loc, exist_ok=True)
    for f in ("train.csv", "test.csv"):
      fs.get(f"{args.data[5:]}/{dataset}/{f}", f"{loc}/{f}")
    header = open(f"{loc}/train.csv").readline().strip().split(",")
    feats = [(c, ydf.Semantic.NUMERICAL) for c in header if c != args.label]
    spec = None
    try:
      spec = ydf.create_vertical_dataset(
          f"csv:{loc}/train.csv",
          columns=feats + [(args.label, ydf.Semantic.CATEGORICAL)]).data_spec()
    except Exception:  # noqa: BLE001  fall back to the learner's own GCS scan
      traceback.print_exc()
    _DS_CACHE.clear()  # keep one dataset resident
    _DS_CACHE[dataset] = (feats, spec, loc)
  return _DS_CACHE[dataset]


def shared_cache_dir(dataset, feats):
  """<cache_root>/<dataset>/<key>: the key pins everything the cache content depends on."""
  key = hashlib.sha1("\n".join([
      CACHE_KEY_VERSION, f"{args.data}/{dataset}/train.csv", args.label,
      ",".join(c for c, _ in feats), "max_unique_values_for_discretized_numerical=1",
  ]).encode()).hexdigest()[:12]
  return f"{args.cache_root.rstrip('/')}/{dataset}/{key}"


class CacheLock:
  """Single builder per shared cache dir: atomic lock object (generation 0), done-marker polling."""

  def __init__(self, cache_dir):
    from google.cloud import storage  # noqa: PLC0415
    self.bucket_name, self.prefix = cache_dir[5:].split("/", 1)
    self.bucket = storage.Client().bucket(self.bucket_name)
    self.done = self.bucket.blob(f"{self.prefix}/dataset_cache/done")
    self.lock = self.bucket.blob(f"{self.prefix}/builder.lock")
    self.held = False

  def acquire_or_wait(self):
    """Returns once the cache is complete (False) or this process holds the build lock (True)."""
    from google.api_core import exceptions  # noqa: PLC0415
    t0 = time.time()
    while True:
      if self.done.exists():
        return False
      try:
        self.lock.upload_from_string(json.dumps({"host": os.uname().nodename, "t": time.time()}),
                                     if_generation_match=0)
        self.held = True
        return True
      except exceptions.PreconditionFailed:
        pass
      # Fresh Blob per probe: a reloaded Blob pins its generation and would 404 forever
      # once another process replaced the lock. self.lock keeps our own upload generation.
      probe = self.bucket.blob(self.lock.name)
      try:
        probe.reload()
        age = time.time() - probe.updated.timestamp()
      except exceptions.NotFound:
        age = None  # released or taken over between our two calls
      if age is not None and age > LOCK_STALE_S:
        print(f"stale cache lock ({age:.0f}s) on {self.prefix}; taking over", flush=True)
        try:
          probe.delete(if_generation_match=probe.generation)
        except (exceptions.PreconditionFailed, exceptions.NotFound):
          pass
      if time.time() - t0 > 2 * LOCK_STALE_S:
        raise TimeoutError(f"cache {self.prefix} not built after {2 * LOCK_STALE_S}s")
      time.sleep(1 if age is None else 5)

  def release(self):
    from google.api_core import exceptions  # noqa: PLC0415
    if self.held:
      try:
        self.lock.delete()
      except exceptions.NotFound:
        pass
      self.held = False


VARIANTS = {"base": {"checkpoints"}, "nockpt": set(), "reuse": {"checkpoints", "reuse_reader"},
            "all": {"reuse_reader"}}


def run_cell(fs, cluster, dataset, level, out_cell, variant=""):
  feats, spec, loc = dataset_ctx(fs, dataset)
  knobs = VARIANTS[variant] if variant else set()
  no_checkpoints = not (args.checkpoints or "checkpoints" in knobs)
  reuse_reader = args.reuse_reader or "reuse_reader" in knobs

  tc = abstract_learner_pb2.TrainingConfig()
  # DGBT drops the generic random_seed hyperparameter (it is applied to a temporary GBT
  # learner and only the GBT extension is copied back); the extra config merges after it.
  tc.random_seed = args.seed
  ext = tc.Extensions[dgbt_pb2.distributed_gradient_boosted_trees_config]
  ext.in_worker_oblique.column_replication = level
  ext.in_worker_oblique.num_projections_from_local_features = False
  if no_checkpoints:
    ext.create_checkpoints = False
  if reuse_reader:
    ext.reuse_dataset_cache_in_workers = True

  lock = None
  if args.cache_root:
    # Shared, deterministic cache; the work dir (checkpoints, tmp) stays per training.
    cache_dir = shared_cache_dir(dataset, feats)
    ext.dataset_cache_path = f"{cache_dir}/dataset_cache"
    work_dir = f"{args.out.replace('/runs/', '/work/')}/{dataset}"
    resume = False
    lock = CacheLock(cache_dir)
    t_lock = time.time()
    building = lock.acquire_or_wait()
    lock_wait_s = round(time.time() - t_lock, 2)
    try:
      return _train_cell(fs, cluster, dataset, level, out_cell, variant, feats, spec, loc, ext,
                         tc, work_dir, resume, building, lock_wait_s, no_checkpoints, reuse_reader)
    finally:
      lock.release()  # also on any failure after the lock was taken
  else:
    # Stable work dir + resume_training=True: dataset_cache/ is built once per dataset and
    # reused across levels. checkpoint/ must go, or the next level resumes this level's trees.
    work_dir = f"{args.out.replace('/runs/', '/work/')}/{dataset}"
    resume = True
    building, lock_wait_s = None, 0.0
    try:
      fs.rm(f"{work_dir}/checkpoint"[5:], recursive=True)
    except FileNotFoundError:
      pass
    return _train_cell(fs, cluster, dataset, level, out_cell, variant, feats, spec, loc, ext, tc,
                       work_dir, resume, building, lock_wait_s, no_checkpoints, reuse_reader)


def _train_cell(fs, cluster, dataset, level, out_cell, variant, feats, spec, loc, ext, tc,
                work_dir, resume, building, lock_wait_s, no_checkpoints, reuse_reader):
  train = f"csv:{args.data}/{dataset}/train.csv"
  barrier_path = None
  if args.barrier_csv:
    barrier_path = f"/tmp/barriers_{dataset}_{level:.2f}.csv"
    os.environ["DGBT_BARRIER_CSV"] = barrier_path

  learner = ydf.DistributedGradientBoostedTreesLearner(
      label=args.label,
      features=feats,
      data_spec=spec,
      split_axis="SPARSE_OBLIQUE",
      sparse_oblique_weights="BINARY",
      sparse_oblique_num_projections_exponent=1.0,
      sparse_oblique_projection_density_factor=1.5,
      sparse_oblique_max_num_projections=2147483647,
      max_unique_values_for_discretized_numerical=1,
      num_trees=args.num_trees,
      max_depth=args.max_depth,
      random_seed=args.seed,
      worker_logs=False,
      workers=cluster.workers,
      working_dir=work_dir,
      resume_training=resume,
      extra_training_config=tc,
  )
  t0 = time.time()
  with LogTee() as tee:
    model = learner.train(train)
  train_s = round(time.time() - t0, 2)
  try:
    ev = model.evaluate(f"csv:{loc}/test.csv", use_slow_engine=True)
  except TypeError:
    ev = model.evaluate(f"csv:{loc}/test.csv")
  held = ";".join(f"{o}+{r}" for o, r in
                  re.findall(r"worker \d+: (\d+) owned \+ (\d+) replicated", tee.text()))
  summary = {
      "dataset": dataset, "seed": args.seed, "replication": level,
      "workers": len(cluster.workers), "trees": args.num_trees, "max_depth": args.max_depth,
      "test_acc": ev.accuracy, "test_logloss": ev.loss,
      "test_auc": ev.characteristics[0].roc_auc if ev.characteristics else None,
      "num_trees_trained": model.num_trees(), "held_cols": held, "train_s": train_s,
      "wall_s": round(time.time() - t0, 2), "ydf_version": ydf.__version__,
      "variant": variant,
      "knobs": {"cache_root": args.cache_root, "no_checkpoints": no_checkpoints,
                "reuse_reader": reuse_reader, "built_cache": building,
                "cache_lock_wait_s": lock_wait_s},
      **phase_summary(tee),
      **json.loads(args.meta),
  }
  fs.pipe(f"{out_cell}/eval.json"[5:], json.dumps(summary, indent=2).encode())
  fs.pipe(f"{out_cell}/manager.log"[5:], tee.text().encode())
  if barrier_path and os.path.exists(barrier_path):
    fs.put(barrier_path, f"{out_cell}/barriers.csv"[5:])
  print(f"CELL_DONE {dataset} r={level:.2f} {variant} acc={ev.accuracy:.4f} {train_s}s "
        f"cache={summary['cache_build_s']} ready={summary['workers_ready_s']} "
        f"ckpt={summary['checkpoints_s']} trees={summary.get('tree_training_s')}", flush=True)


def main():
  cluster = ydf.util.get_vertex_ai_cluster_spec()
  print("cluster:", cluster, flush=True)
  if cluster.is_worker:
    ydf.verbose(2)  # worker-side C++ logs (cache export, column loads) reach Cloud Logging
    while True:  # serve managers forever; restart if a training kills the worker loop
      try:
        ydf.start_worker(cluster.port)
      except Exception:
        traceback.print_exc()
      time.sleep(2)

  ydf.verbose(2)
  fs = gcsfs.GCSFileSystem()
  only = [d for d in args.datasets.split(",") if d]
  datasets = sorted(p.rstrip("/").rsplit("/", 1)[-1] for p in fs.ls(args.data[5:])
                    if fs.exists(p.rstrip("/") + "/train.csv"))
  if only:
    datasets = [d for d in datasets if any(o in d for o in only)]
  levels = [float(x) for x in args.levels.split(",")]
  variants = [v for v in args.variants.split(",") if v] or [""]
  cells = [(d, r, v, k) for d in datasets for r in levels for v in variants
           for k in range(args.repeat)]
  print(f"{len(cells)} cells: {len(datasets)} datasets x {len(levels)} levels x "
        f"{len(variants)} variant(s) x {args.repeat} repeat(s), seed {args.seed}", flush=True)

  def cell_dir(d, r, v, k):
    return f"{args.out}/{d}/r{r:.2f}" + (f"_{v}" if v else "") + (f"_rep{k}" if k else "")

  failed = {}
  for attempt in range(3):
    # One listing instead of one HEAD request per cell (374 cells = ~40 s per pass).
    done = set()
    try:
      fs.invalidate_cache(args.out[5:])
      done = {f"gs://{p}" for p in fs.find(args.out[5:]) if p.endswith("/eval.json")}
    except FileNotFoundError:
      pass
    todo = [c for c in cells
            if (args.force and attempt == 0) or f"{cell_dir(*c)}/eval.json" not in done]
    if not todo:
      break
    print(f"pass {attempt}: {len(todo)} cells to run", flush=True)
    for i, (d, r, v, k) in enumerate(todo, 1):
      try:
        run_cell(fs, cluster, d, r, cell_dir(d, r, v, k), v)
        failed.pop((d, r, v, k), None)
      except Exception as e:  # noqa: BLE001  keep sweeping; retried next pass
        traceback.print_exc()
        failed[(d, r, v, k)] = f"{type(e).__name__}: {e}"
        print(f"CELL_FAIL {d} r={r:.2f} {v} ({i}/{len(todo)})", flush=True)
        time.sleep(5)
  if failed:
    fs.pipe(f"{args.out}/FAILED.json"[5:],
            json.dumps({f"{d}|r{r:.2f}|{v}|{k}": m for (d, r, v, k), m in failed.items()},
                       indent=2).encode())
    print(f"DONE_WITH_FAILURES {len(failed)}", flush=True)
    sys.exit(1)
  print("ALL_CELLS_DONE", flush=True)


if __name__ == "__main__":
  main()
