#!/usr/bin/env python3
"""Column-replication accuracy sweep on Vertex AI: one CustomJob per seed, cells inside the job.

Each job = 1 manager + --workers workers (train_cells.py loops the (dataset, r) grid, resuming
on resubmit). Jobs are scheduled inside the region's N2 custom-training vCPU quota; a failed job
is resubmitted up to twice. Finished cells are collected into <out_root>/results.csv.
  python run_vertex_colrep.py --image .../train-ydf:colrep-34d64d3d-v1 [--run_id X] [--dry_run]
"""
import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "params"))
import params as kdd28  # noqa: E402  (params/kdd28_params.json)

P = kdd28.load_profile("table2")  # defaults below = the recorded Table 2 setting
PROJECT, REGION, BUCKET = "ydf-distributed-spo-gbt", "us-east1", "ariel_ydf_bucket"
FIELDS = ["dataset", "seed", "replication", "workers", "threads", "trees", "max_depth", "rc",
          "test_acc", "test_auc", "test_logloss", "num_conditions", "num_oblique", "num_multi_col",
          "held_cols", "train_s", "wall_s", "cache_build_s", "workers_ready_s",
          "checkpoints_s", "tree_training_s", "ship_bytes_in", "ship_bytes_out", "ship_cols_in",
          "ship_s_max", "ship_s_sum", "cross_worker_splits", "local_projection_count", "git", "log"]
STAMP_COLS = kdd28.stamp_columns(FIELDS)
FIELDS += STAMP_COLS

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
ap.add_argument("--run_id", default="")
ap.add_argument("--seeds", default=",".join(map(str, P["seeds"])))
ap.add_argument("--levels", default=",".join(map(str, P["column_replication"])))
ap.add_argument("--datasets", default="", help="comma filter passed to train_cells.py")
ap.add_argument("--data", default=f"gs://{BUCKET}/cc18_fold0")
ap.add_argument("--machine", default=P["machine"])
ap.add_argument("--workers", type=int, default=P["workers"][0])
ap.add_argument("--num_trees", type=int, default=P["trees"])
ap.add_argument("--max_depth", type=int, default=P["max_depth"])
ap.add_argument("--profile", default="table2.dist_split",
                help="profile name stamped into results.csv; name the override (e.g. table2.dist_per_worker) when --extra_args changes the setting")
ap.add_argument("--git", default="34d64d3d")
ap.add_argument("--quota", type=int, default=96, help="N2 custom-training vCPUs in the region")
ap.add_argument("--out_root", default=os.path.join(
    HERE, "../../../..", "ariel_notes/results/2026-10-28_column_replication/vertex_hm4"))
ap.add_argument("--job_prefix", default="colrep",
                help="display name <prefix>-s<seed>-<run_id>, outputs runs/<prefix>_<run_id>")
ap.add_argument("--env", action="append", default=[],
                help="KEY=VALUE set in every container (manager and workers); repeatable")
ap.add_argument("--extra_args", default="",
                help="extra train_cells.py flags, e.g. '--cache_root gs://b/work/cache --no_checkpoints'")
ap.add_argument("--dry_run", action="store_true", help="print every job's container argv and exit")
args = ap.parse_args()

SEEDS = [int(s) for s in args.seeds.split(",")]
VCPUS = int(args.machine.rsplit("-", 1)[1])
JOB_VCPUS = (1 + args.workers) * VCPUS
RUN_ID = args.run_id or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = f"gs://{BUCKET}/runs/{args.job_prefix}_{RUN_ID}"
now = lambda: datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S")  # noqa: E731

print(f"RUN_ID {RUN_ID} -> {RUN_DIR} | {len(SEEDS)} jobs x (1+{args.workers}) x {args.machine} "
      f"= {JOB_VCPUS} vCPUs/job, quota {args.quota} | trees {args.num_trees} depth {args.max_depth}")


def job_args(seed):
    """Container argv of one seed's job (train_cells.py flags)."""
    meta = {"machine": args.machine, "run_id": RUN_ID, "git": args.git, "threads": VCPUS}
    a = [f"--data={args.data}", f"--out={RUN_DIR}/s{seed}", f"--seed={seed}",
         f"--levels={args.levels}", f"--num_trees={args.num_trees}",
         f"--max_depth={args.max_depth}", f"--meta={json.dumps(meta)}"]
    if args.datasets:
        a.append(f"--datasets={args.datasets}")
    # Recorded P = F budget, explicit since train_cells.py defaults to YDF (2026-11-03). --extra_args may override.
    a += [f"--sparse_oblique_num_projections_exponent={P['spo_exponent']}",
          f"--sparse_oblique_max_num_projections={P['spo_max_projections']}",
          f"--sparse_oblique_projection_density_factor={P['spo_density']}"]
    return a + args.extra_args.split()


if args.dry_run:
    for s in SEEDS:
        print("argv", f"s{s}", json.dumps(job_args(s)))
    raise SystemExit(0)

from google.cloud import aiplatform  # noqa: E402
import gcsfs  # noqa: E402

aiplatform.init(project=PROJECT, location=REGION, staging_bucket=f"gs://{BUCKET}/staging")
fs = gcsfs.GCSFileSystem(project=PROJECT)
ACTIVE = ("JOB_STATE_QUEUED", "JOB_STATE_PENDING", "JOB_STATE_RUNNING")


def make_job(seed):
    a = job_args(seed)
    env = [{"name": k, "value": v} for k, v in (e.split("=", 1) for e in args.env)]
    pool = lambda n: {"machine_spec": {"machine_type": args.machine}, "replica_count": n,  # noqa: E731
                      "container_spec": {"image_uri": args.image, "env": env,
                                         "command": ["python3", "train_cells.py"], "args": a}}
    return aiplatform.CustomJob(display_name=f"{args.job_prefix}-s{seed}-{RUN_ID}",
                                worker_pool_specs=[pool(1), pool(args.workers)])


def in_use():
    """vCPUs held by every active CustomJob in the region, so launchers share the quota."""
    total = 0
    try:
        for j in aiplatform.CustomJob.list(filter=" OR ".join(f'state="{s}"' for s in ACTIVE)):
            for p in j._gca_resource.job_spec.worker_pool_specs:
                total += int(p.machine_spec.machine_type.rsplit("-", 1)[1]) * int(p.replica_count or 1)
    except Exception as e:  # noqa: BLE001
        print(now(), "job list failed", type(e).__name__)
        return sum(JOB_VCPUS for s in running)
    return total


def job_state(job):
    for attempt in range(30):
        try:
            return job.state.name
        except Exception as e:  # noqa: BLE001
            print(now(), "state failed, retry", attempt, type(e).__name__)
            time.sleep(30)
    return "JOB_STATE_FAILED"


LAST = args.levels.split(",")[-1]
LAST_TAG = "ship" if LAST == "ship" else f"r{float(LAST):.2f}"


def seed_done(seed):
    return not fs.exists(f"{RUN_DIR[5:]}/s{seed}/FAILED.json") and bool(
        fs.glob(f"{RUN_DIR[5:]}/s{seed}/*/{LAST_TAG}/eval.json"))


pending, running, attempts, failed_seeds = list(SEEDS), {}, {s: 0 for s in SEEDS}, []
active = {j.display_name: j for j in aiplatform.CustomJob.list(
    filter=" OR ".join(f'state="{s}"' for s in ACTIVE))}
for s in list(pending):
    if (job := active.get(f"{args.job_prefix}-s{s}-{RUN_ID}")) is not None:
        running[s] = job
        pending.remove(s)
        print(now(), "adopting", s, job.resource_name.split("/")[-1])

while pending or running:
    for s in list(pending):
        if in_use() + JOB_VCPUS > args.quota:
            break
        job = make_job(s)
        for attempt in range(10):
            try:
                job.submit()
                break
            except Exception as e:  # noqa: BLE001
                print(now(), "submit failed, retry", attempt, type(e).__name__)
                time.sleep(60)
        running[s] = job
        attempts[s] += 1
        pending.remove(s)
        print(now(), f"submitted seed {s} (attempt {attempts[s]})",
              job.resource_name.split("/")[-1], flush=True)
        time.sleep(15)
    for s, job in list(running.items()):
        state = job_state(job)
        if state.endswith(("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED")):
            del running[s]
            err = f" error: {job.error.message[:200]}" if job.error and job.error.message else ""
            print(now(), f"seed {s} {state}{err}", flush=True)
            if not state.endswith("SUCCEEDED") and attempts[s] < 3:
                pending.append(s)  # resubmit: finished cells are skipped
            elif not state.endswith("SUCCEEDED") and not seed_done(s):
                failed_seeds.append(s)
    if pending or running:
        time.sleep(60)

os.makedirs(args.out_root, exist_ok=True)
rows = []
for p in sorted(fs.glob(f"{RUN_DIR[5:]}/s*/*/*/eval.json")):
    e = json.loads(fs.cat(p))
    rows.append({k: e.get(k, "") for k in FIELDS} | {
        "rc": 0, "threads": e.get("threads", VCPUS), "git": args.git,
        "log": "/".join(p.split("/")[-4:-1])}
        | {col: e.get("stamp", {}).get(c, "") for col, c in zip(STAMP_COLS, kdd28.STAMP)}
        # train_cells.py stamps "<profile>!drift" when it ran under ALLOW_DRIFT=1: keep the mark, generators refuse it
        | {STAMP_COLS[0]: args.profile + (kdd28.DRIFT if str(e.get("stamp", {}).get("profile", "")).endswith(kdd28.DRIFT) else "")})
with open(os.path.join(args.out_root, "results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, FIELDS)
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} cells collected -> {args.out_root}/results.csv; failed seeds: {failed_seeds}")
subprocess.run([os.sys.executable, os.path.join(HERE, "summarize_replication.py"), args.out_root],
               check=False)
print("LAUNCHER_DONE" if not failed_seeds else f"LAUNCHER_FAILED {failed_seeds}", flush=True)
