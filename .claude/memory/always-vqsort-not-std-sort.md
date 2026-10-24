---
name: always-vqsort-not-std-sort
description: "User directive (2026-10-22) - in every sort on the hot path use Highway VQSort, never std::sort; applies to new kernels (e.g. distributed in-worker oblique finder) too"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 0d9290fc-c648-444c-9794-a538816b33e0
  modified: 2026-10-22T18:51:45.906Z
---

Always sort with Highway VQSort (`hwy::VQSort`, K32V32 packing as in `FindBestSplitFlatHighway`,
`splitter_scanner.h`), never `std::sort`, in any kernel on this fork - including new code such as
the distributed in-worker oblique finder (`distributed_decision_tree/oblique.cc`), which used
`std::sort` on (float, uint32) pairs when written on 2026-10-10.

**Why:** the single-machine exact finder already uses VQSort; it is several times faster than
`std::sort` on (key, value) pairs, and the user wants every arm measured on the same sort so
speedups compare the method, not the sort. Stated 2026-10-22 ("i always wanna use VQSort instead
of std::sort").

**How to apply:** when writing or touching any sort of projected values / (feature, label) pairs,
pack to `hwy::K32V32` (or K64V64) and call VQSort; grep for `std::sort` in the diff before
finishing and replace. Related: [[distributed-spo-gbt-in-node-parallelism]].
