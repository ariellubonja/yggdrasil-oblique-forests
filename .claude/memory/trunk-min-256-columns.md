---
name: trunk-min-256-columns
description: "Synthetic trunk datasets must have at least 256 columns; shrink rows, not columns, to fit a time budget"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 8bdbde36-6c81-454f-8256-46a8a978436e
  modified: 2026-10-22T19:18:46.776Z
---

User directive (2026-10-22): every generated trunk dataset uses **at least 256 columns**. To make a
run shorter, reduce rows (or trees), never the column count below 256.

**Why:** the trunk generator (`MakeTrunkDataset` in `examples/train_oblique_forest.cc`, mirrored by
`benchmarks/src/distributed_spo_gbt/vertex_build/gen_trunk.py`) puts the class signal
mu_j = 1/sqrt(j+1) on the first 256 columns; fewer columns truncates the signal and changes the
problem, and the feature-distributed GBT work needs enough columns per worker to be meaningful.

**How to apply:** e.g. a quick distributed smoke = 100k rows x 256 cols, 4 workers (64 cols each),
not 400k x 32. State the shape in the run report. Related: [[no-dataset-slicing]].
