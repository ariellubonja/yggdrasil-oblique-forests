#!/usr/bin/env python3
"""MP4 of the per-feature split histogram, one frame per GBT tree (1..T), pooled over the 10 folds; same axes as
fig:cc18-tree-feature-hist (x = features in header order, y = share %, y range fixed per dataset over all trees)."""
import argparse, collections, csv, io, re, tarfile
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
import numpy as np

here = Path(__file__).resolve().parents[2] / "results" / "misc" / "cc18_gbt_feature_study"
ap = argparse.ArgumentParser()
ap.add_argument("--tar", default=here / "nodes_all.tar.gz")
ap.add_argument("--per_dataset", default=here / "per_dataset.csv")
ap.add_argument("--datasets", default="diabetes,breast-w,tic-tac-toe,dresses-sales,pc4,Bioresponse")
ap.add_argument("--seconds", type=float, default=20.0)
ap.add_argument("--out", default=here / "movie" / "tree_feature_hist.mp4")
ap.add_argument("--fold", type=int, default=-1, help="only this CV fold (0-based); -1 = pool all folds")
ap.add_argument("--rows", type=int, choices=[1, 2], default=2, help="2 = 2x3 grid (1080p), 1 = one row of 6")
ap.add_argument("--bar_max_f", type=int, default=60, help="bars up to this many features, thin lines above")
args = ap.parse_args()

F = {r["dataset"]: int(float(r["F"])) for r in csv.DictReader(open(args.per_dataset))}
want = args.datasets.split(",")
chosen = sorted([ds for ds in F if re.sub(r"^task_\d+_", "", ds) in want], key=lambda ds: (F[ds], ds))
assert len(chosen) == len(want), chosen
counts = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))  # ds -> tree -> attr
with tarfile.open(args.tar) as tf:
    for m in tf.getmembers():
        mm = re.search(r"runs/(.+)__fold(\d+)/nodes\.csv$", m.name)
        if not mm or mm.group(1) not in chosen or (args.fold >= 0 and int(mm.group(2)) != args.fold):
            continue
        for r in csv.DictReader(io.TextIOWrapper(tf.extractfile(m), encoding="utf-8")):
            counts[mm.group(1)][int(r["tree"]) + 1][int(r["attribute"])] += 1
T = min(max(counts[ds]) for ds in chosen)
share = {}  # ds -> (T, F) array of split share in %
for ds in chosen:
    a = np.zeros((T, F[ds]))
    for t in range(1, T + 1):
        for attr, v in counts[ds][t].items():
            a[t - 1, attr] = v
    share[ds] = 100 * a / np.maximum(a.sum(axis=1, keepdims=True), 1)  # a split-less tree stays all-zero

fps = T / args.seconds
one = args.rows == 1
fig, axes = plt.subplots(args.rows, len(chosen) // args.rows, figsize=(19.2, 4.8 if one else 10.8), dpi=100, squeeze=False)
blue = "#3B6EA8"
artists = []
for ax, ds in zip(axes.flat, chosen):
    f, x = F[ds], np.arange(1, F[ds] + 1)
    ax.set_title(re.sub(r"^task_\d+_", "", ds), fontsize=16 if one else 18)
    ax.set_xlim(0.5, f + 0.5); ax.set_ylim(0, 1.08 * share[ds].max())
    ax.set_xticks(list(x) if f <= (9 if one else 12) else [1, f])
    ax.tick_params(labelsize=12 if one else 14)
    if f <= args.bar_max_f:
        artists.append(("bar", ax.bar(x, share[ds][0], width=0.8, color=blue)))
    else:
        artists.append(("line", ax.add_collection(LineCollection([], colors=blue, linewidths=1.2))))
for ax in axes[:, 0]:
    ax.set_ylabel("share of splits (%)", fontsize=15)
for ax in axes[-1]:
    ax.set_xlabel("feature index" if one else "feature index (header order)", fontsize=15)
sup = fig.suptitle("", fontsize=20 if one else 24, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.91 if one else 0.94))

out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
with writer.saving(fig, str(out), dpi=100):
    for t in range(T):
        sup.set_text(f"Tree {t + 1} of {T}   (axis-aligned GBT, " + (f"CV fold {args.fold + 1} only)" if args.fold >= 0 else "pooled over 10 CV models)"))
        for (kind, art), ds in zip(artists, chosen):
            h = share[ds][t]
            if kind == "bar":
                for b, v in zip(art, h):
                    b.set_height(v)
            else:
                art.set_segments([[(i + 1, 0), (i + 1, v)] for i, v in enumerate(h) if v > 0])
        writer.grab_frame()
print(out, f"{T} frames @ {fps:.2f} fps = {T / fps:.1f} s")
