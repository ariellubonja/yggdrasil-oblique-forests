---
name: distributed-runs-only-on-vertex
description: User directive 2026-10-25 - every reported distributed GBT number is a Vertex AI run; the in-process train_distributed_gbt driver and its scripts are smoke tests only
metadata:
  type: feedback
---

Real distributed SPO-GBT / AA-GBT experiments run on Vertex AI (CustomJob, one node per manager and
worker, `run_vertex_barriers.py`). The in-process driver `examples/train_distributed_gbt` and
`run_dgbt.sh` / `run_dgbt_suite.sh` / `run_barrier_cell.sh` are for smoke tests only (correctness,
bit-identity, builds, small shapes).

**Why:** the paper's distributed runtime table, the accuracy suite and the 2026-10-22 APSFailure
in-node/sort table were all m7i in-process runs ("workers" = threads) and were taken for cluster runs.
Thread count on Vertex is the node's vCPU count, not a flag.

**How to apply:** when asked to "run" a distributed experiment, build the image (wheel from a pushed
commit + train image) and launch Vertex cells; never substitute a local run, even if the m7i is handy.
Related: [[vertex-machine-n2-highmem-32]], [[directives-go-in-repo-claude-md]].
