---
name: vertex-machine-n2-highmem-32
description: Vertex AI machine policy — one machine type per comparison; n2-highmem-4 for the W=1..16 grids (user decision 2026-10-24, superseding the same-day n2-highmem-32 choice); warn before anything that would change the type
metadata:
  node_type: memory
  type: feedback
  originSessionId: 8bdbde36-6c81-454f-8256-46a8a978436e
  modified: 2026-10-24T22:44:03.976Z
---

User decisions (both 2026-10-24): every Vertex AI comparison runs all cells, datasets and arms on **one
machine type** (manager = worker). In the morning that type was n2-highmem-32 (256 GB fits every
dataset at W=1); by midday the user **reverted to n2-highmem-4** (4 vCPU, 32 GB) and asked for the
worker curve W = 1, 2, 4, 8, 16 now, with W = 32 and 48 kept in the planned table as a reminder.

**Why:** the N2 custom-training quota in us-east1 is 96 vCPUs (192 pending, 256 auto-denied). On -32
that is W <= 2 per cell; on -4 it is W <= 16 (68 vCPUs). The scaling curve matters more than a W=1
reference for the big sets, so cells that do not fit 70 % of 32 GB (YouTube-8M W <= 2, Numerai and
Criteo W <= 4, Jane Street W <= 2; SPO one step further) are simply marked OOM.

**How to apply:** `run_vertex_barriers.py --machine n2-highmem-4` (manager defaults to the same);
threads per worker = YDF default (`determine_optimal_num_threads`, i.e. the node's vCPUs), never set
by hand. W=32 needs 132 vCPUs, W=48 needs 196 (a 200 request). Planned-runtime table: Overleaf
`tables/table_planned_runtime.tex` (generator in the session scratchpad `plan_table.py`; anchors:
Epsilon -4 grid 20261024T003657Z, cache 14.7 MB/s per creating worker, AA 89 ns/cell/tree, SPO
320 ns/row/projection/tree). **Warn the user before any launch that would change the machine type.**
Related: [[trunk-min-256-columns]], [[no-dataset-slicing]], [[directives-go-in-repo-claude-md]].
