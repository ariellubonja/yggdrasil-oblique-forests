#!/usr/bin/env python3
"""Per-cell phase timings of speed-probe runs (train_cells.py eval.json files).

  python summarize_speed.py gs://ariel_ydf_bucket/runs/speed_base1 [more run dirs...]
  python summarize_speed.py <local dir with s*/<dataset>/r*/eval.json>
Prints one row per cell: dataset, level, variant, repeat, cache build, workers-ready,
checkpoints, tree loop, total train_s, accuracy/AUC (for the bit-identity check).
"""
import glob
import json
import os
import sys


def load(run):
  rows = []
  if run.startswith("gs://"):
    import gcsfs  # noqa: PLC0415
    fs = gcsfs.GCSFileSystem()
    paths = sorted(fs.glob(f"{run[5:]}/s*/*/r*/eval.json"))
    read = lambda p: json.loads(fs.cat(p))  # noqa: E731
  else:
    paths = sorted(glob.glob(os.path.join(run, "s*", "*", "r*", "eval.json")))
    read = lambda p: json.load(open(p))  # noqa: E731
  for p in paths:
    e = read(p)
    cell = p.split("/")[-2]
    parts = cell.split("_")
    e["_variant"] = next((x for x in parts[1:] if not x.startswith("rep")), "")
    e["_rep"] = next((int(x[3:]) for x in parts[1:] if x.startswith("rep")), 0)
    e["_run"] = run.rstrip("/").rsplit("/", 1)[-1]
    rows.append(e)
  return rows


def main():
  rows = [r for run in sys.argv[1:] for r in load(run)]
  # loop_s = "Start tree training" -> "Training done" minus the checkpoints inside it.
  hdr = ("run", "dataset", "r", "variant", "rep", "train_s", "cache_s", "ready_s", "ckpt_s",
         "loop_s", "acc", "auc")
  print("\t".join(hdr))
  for e in rows:
    loop = (round(e["tree_training_s"] - e.get("checkpoints_s", 0), 3)
            if e.get("tree_training_s") is not None else None)
    print("\t".join(str(x) for x in (
        e["_run"], e["dataset"].split("_", 2)[-1][:18], e["replication"], e["_variant"] or "-",
        e["_rep"], e.get("train_s"), e.get("cache_build_s"), e.get("workers_ready_s"),
        e.get("checkpoints_s"), loop,
        round(e["test_acc"], 6) if e.get("test_acc") is not None else None,
        round(e["test_auc"], 6) if e.get("test_auc") is not None else None)))


if __name__ == "__main__":
  main()
