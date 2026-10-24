---
name: no-dataset-slicing
description: Never cut row prefixes/tails out of a real dataset for a run; use a smaller whole dataset or generate a trunk of the needed size
metadata:
  node_type: memory
  type: feedback
  originSessionId: 8bdbde36-6c81-454f-8256-46a8a978436e
  modified: 2026-10-22T19:17:42.223Z
---

User directive (2026-10-22): do not slice datasets (no `head`/`tail` prefixes of SUSY, HIGGS or
anything else) to make a run fit a time budget. Either pick a smaller *whole* dataset or generate a
synthetic trunk of the right shape (`benchmarks/src/distributed_spo_gbt/vertex_build/gen_trunk.py`
for CSV; the harness trunk mode for in-process runs).

**Why:** a prefix is neither the real dataset nor a controlled synthetic one, so its numbers cannot
be compared to either; it also generalises the existing HIGGS-full-size-only rule.

**How to apply:** when a quick run is needed, choose trunk shape by rows × cols (e.g. 100k rows per
worker × 32 cols, the Vertex weak-scaling cell) and say so in the run report. Related:
[[always-vqsort-not-std-sort]].
