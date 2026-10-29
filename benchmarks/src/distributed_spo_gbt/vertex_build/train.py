import argparse
import json
import os
import sys
import threading
import time
import ydf
from yggdrasil_decision_forests.learner import abstract_learner_pb2
from yggdrasil_decision_forests.learner.distributed_gradient_boosted_trees import (
    distributed_gradient_boosted_trees_pb2 as dgbt_pb2,
)

parser = argparse.ArgumentParser()
# Path to training dataset. Should be prefixed with the dataset type e.g. 'csv':.
# See the supported formats at https://ydf.readthedocs.io/en/latest/dataset_formats/
parser.add_argument("--train_ds", type=str, required=True)
# Path to validation dataset. If empty, the model is trained without validation.
parser.add_argument("--valid_ds", type=str)
# Path to test dataset. If empty, the model is not evaluated.
parser.add_argument("--test_ds", type=str)
# Path to save the model.
parser.add_argument("--model", type=str, required=True)
# Work directory containing the temporary working data. Only used for distributed training.
parser.add_argument("--work_dir", default="", type=str)
# Label column to predict.
parser.add_argument("--label", type=str, required=True)
# Is the training distributed, or on a single machine?
parser.add_argument("--distributed", action="store_true")
# AXIS_ALIGNED (stock GBT) or SPARSE_OBLIQUE (this fork's in-worker SPO-GBT).
parser.add_argument("--split_axis", default="AXIS_ALIGNED",
                    choices=["AXIS_ALIGNED", "SPARSE_OBLIQUE"])
# Number of trees; unset = the learner's default.
parser.add_argument("--num_trees", type=int, default=None)
# SPO projections per worker per node = ceil(F ** exponent); unset = learner defaults (exponent 2, density 2).
parser.add_argument("--sparse_oblique_num_projections_exponent", type=float, default=None)
parser.add_argument("--sparse_oblique_projection_density_factor", type=float, default=None)
# AA numerical columns: cache keeps columns with <= this many unique values as lossless bins,
# the rest presorted (default 16000); 1 = everything presorted. force = approximate bins everywhere.
parser.add_argument("--max_unique_values_for_discretized_numerical", type=int, default=None)
parser.add_argument("--force_numerical_discretization", action="store_true")
# JSON object copied verbatim into eval.json (e.g. rows / workers / experiment tags).
parser.add_argument("--meta", default="{}", type=str)
# Where to write the evaluation summary (JSON). Defaults to <model>_eval.json.
parser.add_argument("--eval_json", default="", type=str)
args = parser.parse_args()

# First C++ log line containing the value marks the phase boundary (manager clock).
PHASES = {
    "cache_created": "Dataset cache created in",
    "workers_ready": "All workers ready to train",
    "tree_training_start": "Start tree training",
    "training_done": "Training done",
}


class LogCapture:
  """Tees fd 2 (the C++ logs) through a pipe and timestamps every line locally."""

  def __init__(self, path):
    self.path = path
    self.marks = {}

  def __enter__(self):
    self.t0 = time.time()
    sys.stderr.flush()
    self.saved = os.dup(2)
    r, w = os.pipe()
    os.dup2(w, 2)
    os.close(w)
    self.file = open(self.path, "w")
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
        self.file.write(f"[{t:9.3f}] {line}")
        os.write(self.saved, line.encode(errors="replace"))

  def __exit__(self, *exc):
    sys.stderr.flush()
    os.dup2(self.saved, 2)  # closes the pipe's write end; the pump thread sees EOF
    self.thread.join(timeout=60)
    os.close(self.saved)
    self.file.close()

  def timing(self, train_seconds):
    m = self.marks
    out = {"train_seconds": round(train_seconds, 2), "phase_marks_s": dict(m)}
    if "tree_training_start" in m:
      out["setup_s"] = m["tree_training_start"]
      if "training_done" in m:
        out["tree_training_s"] = round(m["training_done"] - m["tree_training_start"], 3)
    return out


def learner_kwargs():
  """Learner kwargs: split axis + trees; everything else keeps the YDF defaults."""
  kw = {"split_axis": args.split_axis}
  if args.split_axis == "SPARSE_OBLIQUE":
    # The distributed learner does not default the weights; the paper uses binary.
    kw["sparse_oblique_weights"] = "BINARY"
    for k in ("sparse_oblique_num_projections_exponent", "sparse_oblique_projection_density_factor"):
      if getattr(args, k) is not None:
        kw[k] = getattr(args, k)
  if args.num_trees is not None:
    kw["num_trees"] = args.num_trees
  if args.max_unique_values_for_discretized_numerical is not None:
    kw["max_unique_values_for_discretized_numerical"] = args.max_unique_values_for_discretized_numerical
  if args.force_numerical_discretization:
    kw["force_numerical_discretization"] = True
  return kw


BARRIER_CSV = "/tmp/barriers.csv"


def gcs_upload(path, data, content_type, quiet=False):
  from google.cloud import storage
  assert path.startswith("gs://"), path
  bucket, blob = path[5:].split("/", 1)
  storage.Client().bucket(bucket).blob(blob).upload_from_string(data, content_type=content_type)
  if not quiet:
    print("Wrote", path)


def eval_dir():
  return (args.eval_json or args.model.rstrip("/") + "_eval.json").rsplit("/", 1)[0]


def _proc_stat():
  """(busy, total) jiffies for the node and per vCPU from /proc/stat (idle+iowait count as idle)."""
  agg, per = None, []
  for line in open("/proc/stat"):
    if not line.startswith("cpu"):
      break
    f = line.split()
    v = [int(x) for x in f[1:]]
    busy_total = (sum(v) - v[3] - v[4], sum(v))
    if f[0] == "cpu":
      agg = busy_total
    else:
      per.append(busy_total)
  return agg, per


def host_info():
  """CPU model / cache line of /proc/cpuinfo: N2 VMs land on Cascade Lake or Ice Lake hosts."""
  info = {}
  for line in open("/proc/cpuinfo"):
    if ":" in line:
      k, v = line.split(":", 1)
      k = k.strip()
      if k in ("model name", "cache size", "cpu MHz") and k not in info:
        info[k] = v.strip()
  info["nproc"] = os.cpu_count()
  return info


def start_cpu_sampler(role, period=2.0, upload_every=20.0):
  """Daemon thread: node busy fraction + vCPUs >50 % busy per period -> <eval dir>/cpu_<role>.json."""
  dest = f"{eval_dir()}/cpu_{role}.json"
  samples = []
  host = host_info()
  print("host:", json.dumps(host))

  def run():
    t0 = time.time()
    prev, last_up = _proc_stat(), t0
    while True:
      time.sleep(period)
      cur = _proc_stat()
      db, dt = cur[0][0] - prev[0][0], cur[0][1] - prev[0][1]
      busy_cpus = sum(1 for (b1, t1), (b0, t0_) in zip(cur[1], prev[1]) if t1 > t0_ and (b1 - b0) / (t1 - t0_) > 0.5)
      samples.append([round(time.time() - t0, 1), round(db / max(dt, 1), 3), busy_cpus])
      prev = cur
      if time.time() - last_up >= upload_every:
        last_up = time.time()
        try:
          gcs_upload(dest, json.dumps({"role": role, "ncpu": len(cur[1]), "period_s": period, "host": host, "samples": samples}),
                     "application/json", quiet=True)
        except Exception as e:  # noqa: BLE001  sampling must never kill training
          print("cpu sampler upload failed:", type(e).__name__)

  threading.Thread(target=run, daemon=True).start()


def train(learner):
  """Trains with the C++ log captured; returns (model, timing dict, local log path)."""
  log_path = "/tmp/manager.log"
  t0 = time.time()
  with LogCapture(log_path) as cap:
    model = learner.train(args.train_ds, valid=args.valid_ds)
  return model, cap.timing(time.time() - t0), log_path


def finish(model, timing, log_path, num_workers):
  print("Save model in", args.model)
  model.save(args.model)
  print(model.describe("text"))

  summary = {
      "split_axis": args.split_axis,
      "ydf_version": ydf.__version__,
      "num_trees": model.num_trees(),
      "num_workers": num_workers,
      "model": args.model,
      "learner_kwargs": learner_kwargs(),
      "create_checkpoints": None if num_workers == 0 else False,  # None = not distributed
      **timing,
      **json.loads(args.meta),
  }
  if args.test_ds:
    print("Evaluate model on", args.test_ds)
    evaluation = model.evaluate(args.test_ds)
    print(evaluation)
    summary.update({"accuracy": evaluation.accuracy, "loss": evaluation.loss})
    if evaluation.characteristics:
      summary["auc"] = evaluation.characteristics[0].roc_auc
  print("SUMMARY", json.dumps(summary))

  eval_json = args.eval_json or args.model.rstrip("/") + "_eval.json"
  gcs_upload(eval_json, json.dumps(summary, indent=2), "application/json")
  gcs_upload(eval_json.rsplit("/", 1)[0] + "/manager.log", open(log_path).read(), "text/plain")
  if os.path.exists(BARRIER_CSV):
    gcs_upload(eval_json.rsplit("/", 1)[0] + "/barriers.csv", open(BARRIER_CSV).read(), "text/csv")


def main_in_process():
  ydf.verbose(2)
  print("Train model in process on", args.train_ds)
  learner = ydf.GradientBoostedTreesLearner(label=args.label, **learner_kwargs())
  finish(*train(learner), num_workers=0)


def no_checkpoints_config():
  """Checkpoints off (CLAUDE.md 2026-10-29): their GCS I/O falls inside tree_training_s."""
  tc = abstract_learner_pb2.TrainingConfig()
  tc.Extensions[dgbt_pb2.distributed_gradient_boosted_trees_config].create_checkpoints = False
  return tc


def main_distributed():
  ydf.verbose(2)

  # Gather the manager and workers configuration.
  cluster_config = ydf.util.get_vertex_ai_cluster_spec()
  print("cluster_config:\n", cluster_config)

  if cluster_config.is_worker:
    # This machine is running a worker.
    start_cpu_sampler("worker%d" % json.loads(os.environ.get("CLUSTER_SPEC", "{}")).get("task", {}).get("index", 0))
    ydf.start_worker(cluster_config.port)
    return
  start_cpu_sampler("manager")

  # Manager-side barrier recorder (one CSV row per request round), uploaded next to eval.json.
  os.environ["DGBT_BARRIER_CSV"] = BARRIER_CSV
  print("Train model with distribution on", args.train_ds)
  learner = ydf.DistributedGradientBoostedTreesLearner(
      label=args.label,
      workers=cluster_config.workers,
      working_dir=args.work_dir,
      resume_training=True,
      extra_training_config=no_checkpoints_config(),
      **learner_kwargs(),
  )
  finish(*train(learner), num_workers=len(cluster_config.workers))


if __name__ == "__main__":
  print("Arguments:\n", args)
  print("ydf version:", ydf.__version__)
  if args.distributed:
    main_distributed()
  else:
    main_in_process()
