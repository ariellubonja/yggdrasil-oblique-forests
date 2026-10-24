#!/usr/bin/env python3
"""Barrier-timing grid on Vertex AI: one CustomJob per (arm, workers) cell on a natural dataset.

Each cell = 1 manager + W workers of --machine, training image from vertex_build/ (time-barriers
branch: the manager writes barriers.csv next to eval.json). Cells are scheduled inside the region's
N2 custom-training vCPU quota, finished cells are downloaded to
benchmarks/results/runtime/distributed_spo_gbt/barriers/vertex/<RUN_ID>_<dataset>/<arm>_w<W>/ and
summarized with summarize_barriers.py. Re-running with --run_id reuses finished cells.

  python run_vertex_barriers.py --dataset epsilon --workers 1,2,4,8 --arms aa,spo [--dry_run]
"""
import argparse, datetime, hashlib, json, math, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT, REGION, BUCKET = "ydf-distributed-spo-gbt", "us-east1", "ariel_ydf_bucket"
AR = f"{REGION}-docker.pkg.dev/{PROJECT}/ydf-repo"
ARMS = {"aa": "AXIS_ALIGNED", "spo": "SPARSE_OBLIQUE"}
SPO_DENSITY = 1.5  # paper's SPO-GBT arm (E[nnz] = 1.5); binary weights are set by train.py
DATASETS = {  # label + numeric column count drive the projection exponent
    "epsilon": dict(train=f"csv:gs://{BUCKET}/big/EPSILON/epsilon_normalized_train.csv",
                    test=f"csv:gs://{BUCKET}/big/EPSILON/epsilon_test_100k.csv", label="label", cols=2000, rows=400_000),
    # label column holds "1.000000000000000000e+00"/"0.0..." tokens; classification makes it CATEGORICAL
    "higgs": dict(train=f"csv:gs://{BUCKET}/big/HIGGS/HIGGS_train_10500k.csv",
                  test=f"csv:gs://{BUCKET}/big/HIGGS/HIGGS_test_500k_floatlabel.csv", label="class", cols=28, rows=10_500_000),
    "susy": dict(train=f"csv:gs://{BUCKET}/big/SUSY/SUSY_train_4500k.csv",
                 test=f"csv:gs://{BUCKET}/big/SUSY/SUSY_test_500k_floatlabel.csv", label="class", cols=18, rows=4_500_000),
    "climsim": dict(train=f"csv:gs://{BUCKET}/big/CLIMSIM/train/shard_*.csv",
                    test=f"csv:gs://{BUCKET}/big/CLIMSIM/test/shard_*.csv", label="class", cols=124, rows=10_091_520),
    # 41 GB (AA) / 59 GB (SPO) per worker at W=1 (MEMORY_BUDGET.md): on n2-highmem-4 run W >= 4 only
    "youtube8m": dict(train=f"csv:gs://{BUCKET}/big/YOUTUBE8M/youtube8m_video_train.csv",
                      test=f"csv:gs://{BUCKET}/big/YOUTUBE8M/youtube8m_video_validate.csv", label="class", cols=1152, rows=3_888_919),
    # m7i sets, uploaded as 16/4 shards by vertex_build/shard_csv.sh; W=1 SPO needs 98/119/50/23 GB (MEMORY_BUDGET.md)
    "numerai": dict(train=f"csv:gs://{BUCKET}/big/NUMERAI/train/shard_*.csv",
                    test=f"csv:gs://{BUCKET}/big/NUMERAI/test/shard_*.csv", label="class", cols=2376, rows=2_746_268),
    "criteo": dict(train=f"csv:gs://{BUCKET}/big/CRITEO/train/shard_*.csv",
                   test=f"csv:gs://{BUCKET}/big/CRITEO/test/shard_*.csv", label="class", cols=39, rows=195_841_983),
    "jane_street": dict(train=f"csv:gs://{BUCKET}/big/JANE_STREET/train/shard_*.csv",
                        test=f"csv:gs://{BUCKET}/big/JANE_STREET/test/shard_*.csv", label="class", cols=82, rows=40_852_762),
    "airline": dict(train=f"csv:gs://{BUCKET}/big/AIRLINE/train/shard_*.csv",
                    test=f"csv:gs://{BUCKET}/big/AIRLINE/test/shard_*.csv", label="class", cols=43, rows=37_892_905),
}
QUOTA_IDS = {"n2": "CustomModelTrainingN2CPUsPerProjectPerRegion", "c2": "CustomModelTrainingC2CPUsPerProjectPerRegion"}
RESULTS = os.path.join(HERE, "../../results/runtime/distributed_spo_gbt/barriers/vertex")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
ap.add_argument("--workers", default="1,2,4,8")
ap.add_argument("--arms", default="aa,spo")
ap.add_argument("--machine", default="n2-highmem-4", help="worker machine type")
ap.add_argument("--manager_machine", default=None, help="manager machine type; default = --machine (it only holds the test set)")
ap.add_argument("--trees", type=int, default=50)
ap.add_argument("--p_total", type=int, default=None, help="total SPO projections per node over all workers; default ceil(sqrt(cols))")
ap.add_argument("--image", default=None, help="training image; default <AR>/train-ydf:<commit8>-<build hash> like the notebooks")
ap.add_argument("--run_id", default="", help="reuse a run id: finished cells are skipped, active jobs adopted")
ap.add_argument("--dry_run", action="store_true")
args = ap.parse_args()

ds = DATASETS[args.dataset]
WORKERS = [int(w) for w in args.workers.split(",")]
ARM_LIST = args.arms.split(",")
VCPUS = int(args.machine.rsplit("-", 1)[1])
MGR_MACHINE = args.manager_machine or args.machine
MGR_VCPUS = int(MGR_MACHINE.rsplit("-", 1)[1])
P_TOTAL = args.p_total or math.ceil(math.sqrt(ds["cols"]))
now = lambda: datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S")


def spo_exponent(w):
    """ceil(cols ** e) == P_TOTAL // w per worker (projection count uses the global feature count)."""
    p = max(1, P_TOTAL // w)
    return math.log(p - 0.5) / math.log(ds["cols"]) if p > 1 else 0.0


def default_image():
    build = os.path.join(HERE, "vertex_build")
    commit = open(os.path.join(build, "wheel/Dockerfile")).read().split("ARG YDF_COMMIT=")[1].split()[0][:8]
    files = [os.path.join(build, f) for f in ("Dockerfile", "gen_trunk.py", "train.py")]
    h = hashlib.sha1(b"".join(open(f, "rb").read() for f in files)).hexdigest()[:8]
    return f"{AR}/train-ydf:{commit}-{h}"


def training_cpu_quota():
    qid = QUOTA_IDS.get(args.machine.split("-")[0], "CustomModelTrainingCPUsPerProjectPerRegion")
    token = subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()
    url = f"https://cloudquotas.googleapis.com/v1/projects/{PROJECT}/locations/global/services/aiplatform.googleapis.com/quotaInfos/{qid}"
    info = json.loads(subprocess.check_output(["curl", "-s", "-H", f"Authorization: Bearer {token}", url]))
    return next(int(d["details"]["value"]) for d in info["dimensionsInfos"] if d.get("dimensions", {}).get("region") == REGION)


IMAGE = args.image or default_image()
RUN_ID = args.run_id or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = f"gs://{BUCKET}/runs/barriers_{RUN_ID}_{args.dataset}"
tag_of = lambda arm, w: f"{arm}_w{w}"
vcpus = lambda w: MGR_VCPUS + w * VCPUS
CELLS = [(arm, w) for w in WORKERS for arm in ARM_LIST]

print(f"dataset {args.dataset} ({ds['rows']} x {ds['cols']}, label {ds['label']}) | workers {args.machine} ({VCPUS} vCPU), manager {MGR_MACHINE} | "
      f"{args.trees} trees | SPO total projections {P_TOTAL}, density {SPO_DENSITY}\nimage {IMAGE}\nRUN_ID {RUN_ID} -> {RUN_DIR}")
print(f"{'cell':8} {'nodes':>5} {'vCPUs':>5} {'proj/worker':>11} {'exponent':>9}")
for arm, w in CELLS:
    proj = f"{P_TOTAL // w:11d} {spo_exponent(w):9.6f}" if arm == "spo" else f"{'-':>11} {'-':>9}"
    print(f"{tag_of(arm, w):8} {1 + w:5d} {vcpus(w):5d} {proj}")
if args.dry_run:
    sys.exit(0)

from google.cloud import aiplatform  # noqa: E402  (slow import; after --dry_run)
import gcsfs  # noqa: E402

aiplatform.init(project=PROJECT, location=REGION, staging_bucket=f"gs://{BUCKET}/staging")
fs = gcsfs.GCSFileSystem(project=PROJECT)
CPU_BUDGET = training_cpu_quota()
print(f"{args.machine}: quota {CPU_BUDGET} vCPUs in {REGION}; largest cell needs {max(vcpus(w) for _, w in CELLS)}")


def make_job(arm, w):
    tag, out = tag_of(arm, w), f"{RUN_DIR}/{tag_of(arm, w)}"
    meta = {"dataset": args.dataset, "rows": ds["rows"], "cols": ds["cols"], "workers": w, "arm": arm, "machine_type": args.machine,
            "manager_machine_type": MGR_MACHINE, "tag": tag, "trees": args.trees, "image": IMAGE,
            "projections_per_worker": P_TOTAL // w if arm == "spo" else None, "p_total": P_TOTAL if arm == "spo" else None}
    # fresh cache per cell (never resume a stale one), under work/ so the bucket lifecycle rule deletes it after 2 days
    work_dir = f"gs://{BUCKET}/work/{RUN_ID}_{args.dataset}/{tag}_{datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S')}"
    a = [f"--train_ds={ds['train']}", f"--test_ds={ds['test']}", f"--model={out}/model", f"--work_dir={work_dir}",
         f"--eval_json={out}/eval.json", "--distributed", f"--label={ds['label']}", f"--split_axis={ARMS[arm]}",
         f"--num_trees={args.trees}", f"--meta={json.dumps(meta)}"]
    if arm == "spo":
        a += [f"--sparse_oblique_num_projections_exponent={spo_exponent(w):.6f}", f"--sparse_oblique_projection_density_factor={SPO_DENSITY}"]
    pool = lambda n, m: {"machine_spec": {"machine_type": m}, "replica_count": n, "container_spec": {"image_uri": IMAGE, "args": a}}
    return aiplatform.CustomJob(display_name=f"bar-{args.dataset}-{tag}-{RUN_ID}", worker_pool_specs=[pool(1, MGR_MACHINE), pool(w, args.machine)])


def job_state(job):
    for attempt in range(30):
        try:
            return job.state.name
        except Exception as e:  # noqa: BLE001  transport blips must not kill the run
            print(now(), "state failed, retry", attempt, type(e).__name__); time.sleep(30)
    raise RuntimeError("job.state kept failing")


def collect(arm, w):
    """Download eval.json / manager.log / barriers.csv and summarize the barriers."""
    cell_gcs, local = f"{RUN_DIR[5:]}/{tag_of(arm, w)}", os.path.join(RESULTS, f"{RUN_ID}_{args.dataset}", tag_of(arm, w))
    os.makedirs(local, exist_ok=True)
    for f in ("eval.json", "manager.log", "barriers.csv"):
        if fs.exists(f"{cell_gcs}/{f}"):
            fs.get(f"{cell_gcs}/{f}", os.path.join(local, f))
    if os.path.exists(os.path.join(local, "barriers.csv")):
        with open(os.path.join(local, "summary.txt"), "w") as out:
            subprocess.run([sys.executable, os.path.join(HERE, "summarize_barriers.py"), os.path.join(local, "barriers.csv")], stdout=out, stderr=subprocess.STDOUT)
    if os.path.exists(os.path.join(local, "eval.json")):
        e = json.load(open(os.path.join(local, "eval.json")))
        print(now(), tag_of(arm, w), f"tree_training_s={e.get('tree_training_s')} setup_s={e.get('setup_s')} acc={e.get('accuracy')} -> {local}")


pending = sorted(CELLS, key=lambda c: (c[1], c[0] != "spo"))  # small W first: the references
running, done = {}, {}
for c in list(pending):
    if fs.exists(f"{RUN_DIR[5:]}/{tag_of(*c)}/eval.json"):
        done[c] = None; pending.remove(c); print("reusing", tag_of(*c)); collect(*c)
ACTIVE = ("JOB_STATE_QUEUED", "JOB_STATE_PENDING", "JOB_STATE_RUNNING")
active = {j.display_name: j for j in aiplatform.CustomJob.list(filter=" OR ".join(f'state="{s}"' for s in ACTIVE))}
for c in list(pending):
    if (job := active.get(f"bar-{args.dataset}-{tag_of(*c)}-{RUN_ID}")) is not None:
        running[c] = job; pending.remove(c); print("adopting", tag_of(*c), job.resource_name.split("/")[-1])


def in_use():
    """vCPUs held by every active CustomJob in the region (not just ours), so several launchers can share the quota."""
    total = 0
    try:
        for j in aiplatform.CustomJob.list(filter=" OR ".join(f'state="{s}"' for s in ACTIVE)):
            for p in j._gca_resource.job_spec.worker_pool_specs:
                total += int(p.machine_spec.machine_type.rsplit("-", 1)[1]) * int(p.replica_count or 1)
    except Exception as e:  # noqa: BLE001  fall back to our own accounting on a transport blip
        print(now(), "job list failed", type(e).__name__); return sum(vcpus(w) for (_, w) in running)
    return max(total, sum(vcpus(w) for (_, w) in running))


while pending or running:
    for c in list(pending):
        if in_use() + vcpus(c[1]) <= CPU_BUDGET:
            job = make_job(*c)
            for attempt in range(10):
                try:
                    job.submit(); break
                except Exception as e:  # noqa: BLE001
                    print(now(), "submit failed, retry", attempt, type(e).__name__); time.sleep(60)
            running[c] = job; pending.remove(c)
            print(now(), "submitted", tag_of(*c), f"({vcpus(c[1])} vCPUs, {in_use()} in use)", job.resource_name.split("/")[-1])
    for c, job in list(running.items()):
        state = job_state(job)
        if state.endswith(("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED")):
            done[c] = job; del running[c]
            err = f" error: {job.error.message[:300]}" if job.error and job.error.message else ""
            print(now(), tag_of(*c), state + err); collect(*c)
    if pending or running:
        time.sleep(30)
print(now(), "all", len(done), "cells finished ->", os.path.join(RESULTS, f"{RUN_ID}_{args.dataset}"))
