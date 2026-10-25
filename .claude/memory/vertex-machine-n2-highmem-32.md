---
name: vertex-machine-n2-highmem-32
description: Vertex AI machine policy — one machine type per comparison; n2-highmem-16 since 2026-10-25 (after -32 and -4 on 2026-10-24); no 70 % RAM OOM rule (user dropped it 2026-10-25)
metadata:
  node_type: memory
  type: feedback
  originSessionId: 8bdbde36-6c81-454f-8256-46a8a978436e
  modified: 2026-10-24T22:53:55.548Z
---

User decisions (both 2026-10-24): every Vertex AI comparison runs all cells, datasets and arms on **one
machine type** (manager = worker). In the morning that type was n2-highmem-32 (256 GB fits every
dataset at W=1); by midday the user **reverted to n2-highmem-4** (4 vCPU, 32 GB) and asked for the
worker curve W = 1, 2, 4, 8, 16 now, with W = 32 and 48 kept in the planned table as a reminder.

**Why:** the N2 custom-training quota in us-east1 is 96 vCPUs (192 pending, 256 auto-denied). On -32
that is W <= 2 per cell; on -4 it is W <= 16 (68 vCPUs). The scaling curve matters more than a W=1
reference for the big sets.

**The "OOM above 70 % of RAM" line is mine, not the user's.** It came from one n2-highmem-2 failure
(SPO 1.6M x 512, formula said 12.6 GB of 16, killed in tree 1) and was never tested on 32 GB. When the
user asked where it came from (2026-10-24) they chose to **calibrate it**: Criteo aa_w4 (formula 25.7 GB
= 80 % of 32 GB, 2 trees, run 20261024T225254Z) is the probe. Survive -> OOM marks move one W step
left; killed -> 70 % stands with evidence. Never present that line as a user decision again.

**How to apply:** `run_vertex_barriers.py --machine n2-highmem-4` (manager defaults to the same);
threads per worker = YDF default (`determine_optimal_num_threads`, i.e. the node's vCPUs), never set
by hand. W=32 needs 132 vCPUs, W=48 needs 196 (a 200 request). Planned-runtime table: Overleaf
`tables/table_planned_runtime.tex` (generator in the session scratchpad `plan_table.py`; anchors:
Epsilon -4 grid 20261024T003657Z; AA on tall data runs 2-3x above its 89 ns/cell/tree anchor, recalibrate
from the finished cells). **Warn the user before any launch that would change the machine type.**
Related: [[trunk-min-256-columns]], [[no-dataset-slicing]], [[directives-go-in-repo-claude-md]].

**Update 2026-10-25 (user decision):** every distributed run now uses **n2-highmem-16** (16 vCPU, 128 GB),
manager = worker; quota 96 vCPUs ⇒ W ≤ 5 per cell. The -4 grids of 2026-10-24 are not mixed with -16 cells.
The 70 % RAM OOM heuristic and the Criteo aa_w4 probe were dropped by the user; a cell is OOM only when a
worker was actually killed. `run_vertex_barriers.py --machine` defaults to n2-highmem-16.
