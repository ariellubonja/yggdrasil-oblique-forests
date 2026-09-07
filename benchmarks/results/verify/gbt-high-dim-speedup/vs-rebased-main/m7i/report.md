# report.md — `gbt-high-dim-speedup` vs `rebased-main`, m7i verdict (2026-09-07)

**Verdict: ★ real speedup, trees bit-identical.** Full protocol on the verdict machine (m7i, icx, GBT
Dynamic Random Histogram, 300 trees, median of 3): 150k×40k **15.4×**, 15k×40k **83.7×**, HIGGS 1.03×,
1.5M×4096 1.05×; geometric mean 6.1×. Arm A replicates the 2026-07-19 m7i baseline within 2.1 % on every
shared cell. Quick pass (30 trees) additionally puts 15k×400k at 130× (745 s → 5.7 s).

Details, tables, provenance, commands and what was not done: `compare.md` in this directory.
Mac quick pass of 2026-09-05 (pipeline validation only): `../compare.md`.
