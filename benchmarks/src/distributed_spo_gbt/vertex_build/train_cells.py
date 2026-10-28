#!/usr/bin/env python3
"""Many small distributed-GBT cells in one Vertex CustomJob (column-replication study).

Pool 0 = manager, pool 1 = W workers (real GRPC distribution). The manager loops over
(dataset, replication) cells with one seed, reusing the workers; a cell whose eval.json
already exists in --out is skipped, so resubmitting the job resumes it.
"""
import argparse
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
args = parser.parse_args()


class LogTee:
  """Tees fd 2 (the C++ logs) into a buffer so the replication lines can be parsed."""

  def __init__(self):
    self.lines = []

  def __enter__(self):
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
        self.lines.append(line)
        os.write(self.saved, line.encode(errors="replace"))

  def __exit__(self, *exc):
    sys.stderr.flush()
    os.dup2(self.saved, 2)
    self.thread.join(timeout=30)
    os.close(self.saved)

  def text(self):
    return "".join(self.lines)


def run_cell(fs, cluster, dataset, level, out_cell):
  train = f"csv:{args.data}/{dataset}/train.csv"
  test = f"csv:{args.data}/{dataset}/test.csv"
  # Stable work dir + resume_training=True: dataset_cache/ is built once per dataset and
  # reused across levels. checkpoint/ must go, or the next level resumes this level's trees.
  work_dir = f"{args.out.replace('/runs/', '/work/')}/{dataset}"
  try:
    fs.rm(f"{work_dir}/checkpoint"[5:], recursive=True)
  except FileNotFoundError:
    pass
  with fs.open(f"{args.data[5:]}/{dataset}/train.csv".replace("gs://", ""), "r") as f:
    header = f.readline().strip().split(",")
  feats = [(c, ydf.Semantic.NUMERICAL) for c in header if c != args.label]

  tc = abstract_learner_pb2.TrainingConfig()
  ext = tc.Extensions[dgbt_pb2.distributed_gradient_boosted_trees_config]
  ext.in_worker_oblique.column_replication = level
  ext.in_worker_oblique.num_projections_from_local_features = False

  learner = ydf.DistributedGradientBoostedTreesLearner(
      label=args.label,
      features=feats,
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
      resume_training=True,
      extra_training_config=tc,
  )
  t0 = time.time()
  with LogTee() as tee:
    model = learner.train(train)
  train_s = round(time.time() - t0, 2)
  try:
    ev = model.evaluate(test, use_slow_engine=True)
  except TypeError:
    ev = model.evaluate(test)
  held = ";".join(f"{o}+{r}" for o, r in
                  re.findall(r"worker \d+: (\d+) owned \+ (\d+) replicated", tee.text()))
  summary = {
      "dataset": dataset, "seed": args.seed, "replication": level,
      "workers": len(cluster.workers), "trees": args.num_trees, "max_depth": args.max_depth,
      "test_acc": ev.accuracy, "test_logloss": ev.loss,
      "test_auc": ev.characteristics[0].roc_auc if ev.characteristics else None,
      "num_trees_trained": model.num_trees(), "held_cols": held, "train_s": train_s,
      "wall_s": round(time.time() - t0, 2), "ydf_version": ydf.__version__,
      **json.loads(args.meta),
  }
  fs.pipe(f"{out_cell}/eval.json"[5:], json.dumps(summary, indent=2).encode())
  fs.pipe(f"{out_cell}/manager.log"[5:], tee.text().encode())
  print(f"CELL_DONE {dataset} r={level:.2f} acc={ev.accuracy:.4f} {train_s}s", flush=True)


def main():
  cluster = ydf.util.get_vertex_ai_cluster_spec()
  print("cluster:", cluster, flush=True)
  if cluster.is_worker:
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
  cells = [(d, r) for d in datasets for r in levels]
  print(f"{len(cells)} cells: {len(datasets)} datasets x {len(levels)} levels, seed {args.seed}",
        flush=True)
  failed = {}
  for attempt in range(3):
    todo = [(d, r) for d, r in cells
            if not fs.exists(f"{args.out}/{d}/r{r:.2f}/eval.json"[5:])]
    if not todo:
      break
    print(f"pass {attempt}: {len(todo)} cells to run", flush=True)
    for i, (d, r) in enumerate(todo, 1):
      out_cell = f"{args.out}/{d}/r{r:.2f}"
      try:
        run_cell(fs, cluster, d, r, out_cell)
        failed.pop((d, r), None)
      except Exception as e:  # noqa: BLE001  keep sweeping; retried next pass
        traceback.print_exc()
        failed[(d, r)] = f"{type(e).__name__}: {e}"
        print(f"CELL_FAIL {d} r={r:.2f} ({i}/{len(todo)})", flush=True)
        time.sleep(5)
  if failed:
    fs.pipe(f"{args.out}/FAILED.json"[5:],
            json.dumps({f"{d}|r{r:.2f}": m for (d, r), m in failed.items()}, indent=2).encode())
    print(f"DONE_WITH_FAILURES {len(failed)}", flush=True)
    sys.exit(1)
  print("ALL_CELLS_DONE", flush=True)


if __name__ == "__main__":
  main()
