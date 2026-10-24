---
name: test-split-last-10-percent
description: "Datasets without an explicit test set use the last 10 % of the file's rows as the test set (user directive 2026-10-24)"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 8bdbde36-6c81-454f-8256-46a8a978436e
  modified: 2026-10-24T00:27:23.364Z
---

User directive (2026-10-24): when a dataset ships without an explicit test split, the **last 10 % of
the file's rows** are the test set and the first 90 % are the training set. Datasets that do have a
split (CC18 train/test folders, Epsilon, ClimSim, YouTube-8M) keep theirs.

**Why:** one fixed rule across datasets, boxes and sessions; the rows are often chronological, so a
tail split is also a time-respecting hold-out. Stated when HIGGS and SUSY were found locally as single
unsplit files.

**How to apply:** cut by row count at the 90 % line (`head -n 1+N*0.9` / `tail -n N*0.1`, header on
both). This is a train/test split, not a "slice to fit a budget", so it does not conflict with
[[no-dataset-slicing]]. Written to the repo CLAUDE.md the same turn, per
[[directives-go-in-repo-claude-md]].
