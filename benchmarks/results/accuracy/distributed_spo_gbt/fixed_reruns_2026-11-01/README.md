# KDD27 accuracy results rerun with the 2026-10-31/11-01 bug fixes

How to recreate everything here: `benchmarks/src/distributed_spo_gbt/REPLICABILITY.md` §G.
Comparison (prints every table below): `python benchmarks/src/distributed_spo_gbt/fixed_reruns/compare_fixed_reruns.py`
(writes `comparisons/`).

| Folder | Paper table | Code | Where it ran |
|---|---|---|---|
| `paper_refs/` | 2, 3 | the published runs, copied from branches `column-replication` and `cross-worker-col-comms` | Vertex |
| `table2/binfix/` | 2 | `fix-dgbt-spo-discretization` 7c0eebd3 (image `train-ydf:discfix-7c0eebd3`) | Vertex, run `colrep_discfix20261101` |
| `table2/fullfix/` | 2 | `dgbt-spo-full-fixes` 10509539 (`train-ydf:fullfix-10509539`) | Vertex, `colrep-full_fullfix20261101` |
| `table2/fullfix_splitbudget/` | 2 (parity) | same + `--split_budget` | Vertex, `colrep-fullparity_fullparity20261101` |
| `table3/binfix/g0..g3/` | 3 | `fix-disc-colcomms` fdeb67a5 (`train-ydf:colship-discfix-fdeb67a5`) | Vertex, `colship-g*_discfix20261101` |
| `table3/fullfix/g0..g3/` | 3 | `fix-disc-colcomms` 1ad2062f (`train-ydf:colship-fullfix-1ad2062f`) | Vertex, `colship-g*_fullfix20261101` |
| `table8/binfix{,_numeric}/` | 8 | 7c0eebd3 driver, paper flags / + numeric 0-1 typing | m7i (ariel-ydf-2), icx |
| `table8/fullfix{,_numeric,_rep}/` | 8 | 10509539 driver; `_rep` = repeat of `fullfix` | m7i |
| `table8/parity/` | 8 (parity) | 10509539, numeric typing, r = 1, split budget | m7i |
| `table8/sort_speed_ab_m7i.csv` | — | old vs deterministic sort, 3 reps (runtime only) | m7i |

Paper Table 8 cells: `benchmarks/results/runtime/{speedup_map_by_dataset,distributed_spo_gbt}/suite_results.csv`.
All cells rc = 0.

## Results (mean over datasets)

**Table 2** (34 CC18, W = 4, 10 trees, 5 seeds; single-machine SPO .8557)

| r | paper | bin fix | all fixes | all fixes + split budget |
|---|---|---|---|---|
| 0 | .8449 | .8559 | .8559 | .8536 |
| 0.5 | .8527 | .8586 | .8586 | .8538 |
| 1 | .8518 | .8574 | .8574 | .8540 |

Parity vs single machine at r = 1 (paired by dataset): paper −0.39 ± 0.35 pts; all fixes +0.17 ± 0.16 pts
(log loss −0.0065 ± 0.0018: P = F per worker is 4F per node); + split budget −0.17 ± 0.12 pts, log loss
+0.0000 ± 0.0008.

**Table 3** (34 CC18, W = 4, 300 trees, seed 1)

| Arm | paper | bin fix | all fixes | tree time paper → all fixes |
|---|---|---|---|---|
| in-worker | .8487 | .8797 | .8795 | 9.4 → 9.8 s |
| cross-network | .8474 | .8752 | .8752 | 15.5 → 15.8 s |

Internet-Advertisements .146 → .976 in both arms.

**Table 8** (33 TabArena/TabReD, 5 folds, 300 trees; AUC; single-machine SPO .8323)

| W | paper | all fixes | all fixes, numeric | parity (r = 1, split budget, numeric) |
|---|---|---|---|---|
| 1 | .8312 | .8312 | .8314 | — |
| 4 | .8305 | .8299 | .8298 | .8316 |
| 16 | .8272 | .8274 | .8276 | .8315 |
| 32 | — | .8274 | .8267 | .8314 |

`fullfix` vs `fullfix_rep`: 765/765 cells identical. Parity: ΔAUC vs single −0.0007 to −0.0009 at every W, the same
as W = 1 (−0.0009 ± 0.0007), so the distributed-specific gap is gone; the W-independent residual is an open question.
