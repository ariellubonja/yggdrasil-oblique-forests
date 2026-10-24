---
name: directives-go-in-repo-claude-md
description: "Every standing user directive must be written into the git-tracked project CLAUDE.md in the same turn, not only into agent memory"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 8bdbde36-6c81-454f-8256-46a8a978436e
  modified: 2026-10-22T19:26:09.064Z
---

User directive (2026-10-22): when the user states a standing rule ("always X", "never Y"), append it
to the repo's `CLAUDE.md` (dated, with the reason) in the same turn, under the matching section.
Memory files may mirror it, but the git-tracked file is the copy of record.

**Why:** the user works from many accounts and devices; Claude Code memory is per account and
machine and does not follow them ("i use many accounts and devices and its hard to synchronize
your memories").

**How to apply:** memory first is fine for recall, but never stop there; the turn is not done until
`CLAUDE.md` has the rule. Done so far for [[no-dataset-slicing]], [[trunk-min-256-columns]],
[[always-vqsort-not-std-sort]] (CLAUDE.md 2026-10-22).
