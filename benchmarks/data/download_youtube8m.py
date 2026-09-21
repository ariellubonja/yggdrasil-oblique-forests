#!/usr/bin/env python3
"""Download the YouTube-8M (2018, v2) *video-level* features and convert them to
all-numeric binary-classification CSVs for the oblique-RF harness.

Source: https://research.google.com/youtube8m/download.html  (CC BY 4.0)
  * 3,844 TFRecord shards per partition (train / validate / test), ~31 GB total,
    served from http://<mirror>.data.yt8m.org/2/video/<partition>/<name>.tfrecord
    with an MD5 per shard in http://data.yt8m.org/2/download_plans/video_<partition>.json.
  * Each record is a tf.train.Example: `id` (bytes), `labels` (int64 list, multi-label
    over 3,862 entities, ids sorted by decreasing frequency), `mean_rgb` (1024 fp32),
    `mean_audio` (128 fp32). No NaNs. `test` has no labels and is skipped by default.

This script re-implements the upstream `download.py` (same plan JSON, same shard
renaming `train<2 chars>` -> `train%04d`, same MD5 check) but downloads shards in
parallel, and parses the TFRecords with a ~60-line protobuf decoder so TensorFlow is
NOT a dependency (only numpy).

Binary task: class = 1 if `--target_label` (default 0, the most frequent entity,
"Game" in the v2 vocabulary; ~21 % of videos) is among the video's labels, else 0.
The full multi-label list is kept in a sidecar CSV so other targets can be derived
without re-downloading.

  benchmarks/data/download_youtube8m.py                       # train + validate
  benchmarks/data/download_youtube8m.py --partitions train --max_shards 40   # smoke test

Outputs (all in benchmarks/data/youtube8m/, gitignored):
  youtube8m_video_<partition>.csv          class,rgb_0..rgb_1023,audio_0..audio_127  (%.9g, fp32-exact)
  youtube8m_video_<partition>_labels.csv   id,labels (space-separated entity ids), same row order
  tfrecord_temp/<partition>/               the verified shards (delete after conversion if space matters)
The binary target follows the repo rule for multi-class / multi-label data (CLAUDE.md, dataset
policy): the MAJORITY label vs the rest. Entity 0 is the most frequent label in both
partitions (20.3 % of videos; entity 1 is next at 13.9 %), so class=1 iff 0 in labels.
Rows are in shard-index order (train0000, train0001, ...), record order within a shard.
"""
import argparse
import hashlib
import io
import itertools
import json
import multiprocessing as mp
import os
import struct
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "youtube8m")        # everything this script writes lives here
TEMP_DIR = os.path.join(OUT_DIR, "tfrecord_temp")
PLAN_URL = "http://data.yt8m.org/2/download_plans/video_{partition}.json"
SHARD_URL = "http://{mirror}.data.yt8m.org/2/video/{partition}/{name}"
N_RGB, N_AUDIO = 1024, 128

# Upstream shard naming: "train" + 2 chars from [a-zA-Z0-9] -> "train%04d" by index.
_VOCAB = [chr(c) for c in itertools.chain(range(97, 123), range(65, 91), range(48, 58))]
_FILE_INDEX = {"".join(p): i for i, p in enumerate(itertools.product(_VOCAB, repeat=2))}


def local_shard_name(remote_name):
    stem, ext = remote_name.split(".")
    return "%s%04d.%s" % (stem[:-2], _FILE_INDEX[stem[-2:]], ext)


# ----------------------------------------------------------------------------- download
def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_plan(partition):
    plan_path = os.path.join(TEMP_DIR, "video_%s_download_plan.json" % partition)
    if not os.path.exists(plan_path):
        os.makedirs(TEMP_DIR, exist_ok=True)
        with urllib.request.urlopen(PLAN_URL.format(partition=partition), timeout=60) as r:
            data = r.read()
        with open(plan_path, "wb") as f:
            f.write(data)
    with open(plan_path) as f:
        files = json.load(f)["files"]            # remote name -> md5
    # (local path, remote name, md5), sorted by local shard index.
    shards = sorted((os.path.join(TEMP_DIR, partition, local_shard_name(n)), n, m)
                    for n, m in files.items())
    return shards


def download_shard(local, remote, md5, partition, mirror, retries=5):
    if os.path.exists(local) and md5sum(local) == md5:
        return local, 0
    url = SHARD_URL.format(mirror=mirror, partition=partition, name=remote)
    part = local + ".part"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
                for chunk in iter(lambda: r.read(1 << 20), b""):
                    f.write(chunk)
            if md5sum(part) != md5:
                raise IOError("MD5 mismatch for %s" % remote)
            os.replace(part, local)
            return local, os.path.getsize(local)
        except Exception as e:                      # noqa: BLE001
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError


def download_partition(partition, shards, mirror, jobs):
    os.makedirs(os.path.join(TEMP_DIR, partition), exist_ok=True)
    t0, done, new_bytes = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(download_shard, l, r, m, partition, mirror) for l, r, m in shards]
        for fut in as_completed(futs):
            _, nb = fut.result()
            done += 1
            new_bytes += nb
            if done % 200 == 0 or done == len(shards):
                print("  [%s] %d/%d shards, %.1f GB new, %.0f s" % (
                    partition, done, len(shards), new_bytes / 1e9, time.time() - t0), flush=True)


# ----------------------------------------------------------------------------- TFRecord / protobuf
def _read_varint(buf, pos):
    result, shift = 0, 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def _iter_fields(buf, start, end):
    """Yield (field_number, wire_type, value) of the protobuf message buf[start:end].
    Length-delimited values are (start, end) offsets into buf."""
    pos = start
    while pos < end:
        key, pos = _read_varint(buf, pos)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, pos = _read_varint(buf, pos)
        elif wt == 2:
            ln, pos = _read_varint(buf, pos)
            v = (pos, pos + ln)
            pos += ln
        elif wt == 5:
            v, pos = buf[pos:pos + 4], pos + 4
        elif wt == 1:
            v, pos = buf[pos:pos + 8], pos + 8
        else:
            raise ValueError("unsupported wire type %d" % wt)
        yield fn, wt, v


def parse_example(buf, start, end):
    """tf.train.Example -> {name: bytes list | int list | float32 ndarray}."""
    out = {}
    for fn, _, (fs, fe) in _iter_fields(buf, start, end):        # Example.features = 1
        if fn != 1:
            continue
        for fn2, _, (ms, me) in _iter_fields(buf, fs, fe):        # Features.feature = 1 (map entry)
            if fn2 != 1:
                continue
            name, val = None, None
            for fn3, _, v in _iter_fields(buf, ms, me):           # entry.key = 1, entry.value = 2
                if fn3 == 1:
                    name = buf[v[0]:v[1]].decode()
                elif fn3 == 2:
                    for fn4, _, (ls, le) in _iter_fields(buf, v[0], v[1]):  # Feature.{bytes,float,int64}_list
                        if fn4 == 1:
                            val = [buf[a:b] for _, _, (a, b) in _iter_fields(buf, ls, le)]
                        elif fn4 == 2:
                            parts = [np.frombuffer(buf[a:b], dtype="<f4") for _, w, (a, b) in _iter_fields(buf, ls, le)]
                            val = np.concatenate(parts) if parts else np.zeros(0, "<f4")
                        elif fn4 == 3:
                            val = []
                            for _, w, v5 in _iter_fields(buf, ls, le):
                                if w == 2:                        # packed varints
                                    p = v5[0]
                                    while p < v5[1]:
                                        x, p = _read_varint(buf, p)
                                        val.append(x)
                                else:
                                    val.append(v5)
            out[name] = val
    return out


def iter_tfrecord(path):
    """Yield (buf, start, end) per record. Layout: u64 length, u32 masked crc, data, u32 crc."""
    with open(path, "rb") as f:
        buf = f.read()
    pos, n = 0, len(buf)
    while pos < n:
        (ln,) = struct.unpack_from("<Q", buf, pos)
        pos += 12
        yield buf, pos, pos + ln
        pos += ln + 4


# ----------------------------------------------------------------------------- conversion
_TARGET = 0


def _init_worker(target_label):
    global _TARGET
    _TARGET = target_label


def convert_shard(path):
    """One shard -> (n_rows, feature CSV bytes, labels CSV bytes, n_positive)."""
    feats, classes, ids, labels = [], [], [], []
    for buf, s, e in iter_tfrecord(path):
        ex = parse_example(buf, s, e)
        rgb, audio = ex["mean_rgb"], ex["mean_audio"]
        if len(rgb) != N_RGB or len(audio) != N_AUDIO:
            raise ValueError("%s: bad feature sizes %d/%d" % (path, len(rgb), len(audio)))
        feats.append(np.concatenate([rgb, audio]))
        classes.append(1 if _TARGET in ex["labels"] else 0)
        ids.append(ex["id"][0].decode())
        labels.append(" ".join(map(str, ex["labels"])))
    if not feats:
        return 0, b"", b"", 0
    x = np.stack(feats)
    if not np.isfinite(x).all():
        raise ValueError("%s: non-finite feature values" % path)
    bio = io.BytesIO()
    np.savetxt(bio, x, fmt="%.9g", delimiter=",")          # %.9g round-trips fp32 exactly
    body = bio.getvalue().splitlines(keepends=True)
    out = b"".join(b"%d,%s" % (c, line) for c, line in zip(classes, body))
    lab = "".join("%s,%s\n" % (i, l) for i, l in zip(ids, labels)).encode()
    return len(feats), out, lab, sum(classes)


def convert_partition(partition, shards, target_label, workers):
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "youtube8m_video_%s.csv" % partition)
    lab_path = os.path.join(OUT_DIR, "youtube8m_video_%s_labels.csv" % partition)
    header = ("class," + ",".join("rgb_%d" % i for i in range(N_RGB)) + ","
              + ",".join("audio_%d" % i for i in range(N_AUDIO)) + "\n").encode()
    t0, rows, pos = time.time(), 0, 0
    tmp_csv, tmp_lab = csv_path + ".part", lab_path + ".part"
    with open(tmp_csv, "wb") as fo, open(tmp_lab, "wb") as fl, \
            mp.Pool(workers, initializer=_init_worker, initargs=(target_label,)) as pool:
        fo.write(header)
        fl.write(b"id,labels\n")
        for k, (n, body, lab, npos) in enumerate(
                pool.imap(convert_shard, [s[0] for s in shards], chunksize=4), 1):
            fo.write(body)
            fl.write(lab)
            rows += n
            pos += npos
            if k % 500 == 0 or k == len(shards):
                print("  [%s] %d/%d shards -> %d rows (%.2f%% positive), %.0f s" % (
                    partition, k, len(shards), rows, 100.0 * pos / max(rows, 1), time.time() - t0), flush=True)
    os.replace(tmp_csv, csv_path)
    os.replace(tmp_lab, lab_path)
    print("Wrote %s (%d rows x %d features, %.1f GB) and %s" % (
        csv_path, rows, N_RGB + N_AUDIO, os.path.getsize(csv_path) / 1e9, lab_path), flush=True)
    return rows, pos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partitions", nargs="+", default=["train", "validate"],
                    choices=["train", "validate", "test"])
    ap.add_argument("--mirror", default="us", choices=["us", "eu", "asia"])
    ap.add_argument("--target_label", type=int, default=0,
                    help="entity id whose presence defines class=1 (default 0 = most frequent)")
    ap.add_argument("--jobs", type=int, default=16, help="parallel shard downloads")
    ap.add_argument("--workers", type=int, default=max(1, min(os.cpu_count() or 1, 32)),
                    help="parallel shard->CSV converters")
    ap.add_argument("--max_shards", type=int, default=None, help="only the first N shards (smoke test)")
    ap.add_argument("--skip_convert", action="store_true", help="download and verify only")
    args = ap.parse_args()

    for partition in args.partitions:
        shards = fetch_plan(partition)
        if args.max_shards:
            shards = shards[:args.max_shards]
        print("== %s: %d shards" % (partition, len(shards)), flush=True)
        download_partition(partition, shards, args.mirror, args.jobs)
        if args.skip_convert:
            continue
        if partition == "test":
            print("  test has no labels; converting with class=0 for every row", flush=True)
        convert_partition(partition, shards, args.target_label, args.workers)


if __name__ == "__main__":
    sys.exit(main())
