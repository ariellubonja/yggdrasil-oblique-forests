---
name: vertex-machine-n2-highmem-32
description: All Vertex AI distributed GBT experiments use n2-highmem-32 for manager and workers (user decision 2026-10-24); warn before adding a dataset that does not fit one worker
metadata:
  node_type: memory
  type: feedback
  originSessionId: 8bdbde36-6c81-454f-8256-46a8a978436e
  modified: 2026-10-24T15:27:09.437Z
---

User decision (2026-10-24): every Vertex AI experiment, all datasets and all arms, runs on
**n2-highmem-32** (32 vCPU, 256 GB) for both the manager pool and the worker pool. No mixing of
machine types across cells or datasets; earlier n2-highmem-4 results (Epsilon barrier grid
20261024T003657Z) are not comparable and get re-run.

**Why:** the user wants one instance type behind every number in the paper. 256 GB per worker fits
every current dataset at W=1 by the MEMORY_BUDGET.md formula (largest: Criteo SPO 119 GB = 47 %,
Numerai SPO 98 GB, YouTube-8M SPO 59 GB), so the W=1 reference exists for all of them.

**How to apply:** `run_vertex_barriers.py --machine n2-highmem-32` (manager defaults to the same).
Quota: 96 N2 vCPUs in us-east1 = 3 nodes = W <= 2 per cell (192 pending = W <= 5). **Warn the user
before launching** when a new dataset's W=1 per-worker need (rows x cols / 1 x B_cell + SPO copy, see
MEMORY_BUDGET.md) exceeds ~70 % of 256 GB = 180 GB; that is the point where the machine choice has
to be revisited, and it is their call. Related: [[trunk-min-256-columns]], [[no-dataset-slicing]].
