# report.md — `gbt-high-dim-speedup` vs `upstream/main`, m7i verdict (2026-09-07)

**Verdict: ★ real speedup on the upstream code path, trees bit-identical. Protocol is Exact vs Exact**
(upstream has no sparse-oblique histogramming as of `df16834d`, 2026-09-03). Full protocol on the verdict
machine (m7i, icx, GBT Exact, 300 trees, median of 3): 150k×40k **12.4×**, 15k×40k **73.8×**, HIGGS 1.02×,
1.5M×4096 1.08×; geometric mean 5.6×. One full-size run per arm of 15k×400k: 7513.7 s → 139.1 s
(**54.0×**). First m7i point of this protocol, so no replication statement.

Details, tables, provenance, commands and what was not done: `compare.md` in this directory.
Mac quick pass of 2026-09-05/06 and the port evidence: `../report.md`, `../ported_*`.
