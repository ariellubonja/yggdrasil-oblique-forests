"""Writes trunk CSV shards to GCS (run as a Vertex job: in-region upload, one numpy version for all sizes).

Trunk = MakeTrunkDataset of examples/train_oblique_forest.cc: class 1 at +mu, class 0 at -mu with
mu_j = 1/sqrt(j+1) on the first 256 columns, unit-variance Gaussian noise. numpy RNG, rows shuffled.
Each smaller size is a prefix of the largest one. Overwrites existing shards.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True, help="gs://bucket/prefix; sets land in <out>/<rows>x<cols>/train/")
parser.add_argument("--rows", type=int, nargs="+", required=True)
parser.add_argument("--cols", type=int, required=True)
parser.add_argument("--shards", type=int, default=16)
parser.add_argument("--test_rows", type=int, default=100_000)
parser.add_argument("--test_shards", type=int, default=4)
parser.add_argument("--label", default="label")
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()


def trunk(rows: int, cols: int, seed: int):
  rng = np.random.default_rng(seed)
  y = (np.arange(rows) >= rows // 2).astype(np.int8)
  mu = np.zeros(cols, np.float32)
  k = min(256, cols)
  mu[:k] = 1 / np.sqrt(np.arange(1, k + 1, dtype=np.float32))
  x = rng.standard_normal((rows, cols), dtype=np.float32)
  x += np.where(y[:, None] == 1, mu, -mu)
  perm = rng.permutation(rows)
  return x[perm], y[perm]


def write_shards(x, y, prefix: str, shards: int):
  t0 = time.time()
  if not prefix.startswith("gs://"):
    os.makedirs(prefix, exist_ok=True)  # local smoke tests only
  names = [f"f{j}" for j in range(x.shape[1])]
  n = (len(y) + shards - 1) // shards
  for s in range(shards):
    df = pd.DataFrame(x[s * n:(s + 1) * n], columns=names)
    df[args.label] = y[s * n:(s + 1) * n]
    df.to_csv(f"{prefix}/shard_{s:02d}.csv", index=False, float_format="%.7g")
  print(f"wrote {prefix} {x.shape} in {time.time() - t0:.0f}s", flush=True)


x, y = trunk(max(args.rows), args.cols, args.seed)
for rows in sorted(args.rows):
  write_shards(x[:rows], y[:rows], f"{args.out}/{rows}x{args.cols}/train", args.shards)
xt, yt = trunk(args.test_rows, args.cols, args.seed + 1)
write_shards(xt, yt, f"{args.out}/test_{args.test_rows}x{args.cols}", args.test_shards)
print("DONE", flush=True)
