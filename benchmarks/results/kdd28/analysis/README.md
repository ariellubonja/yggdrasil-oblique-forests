# analysis/: evidence with no paper float (moved here 2026-11-04)

Measurements that back claims in REPLICABILITY §G/§H and the journal. No float reads them.
`check_layout.py` requires every entry of this folder to be named here.

| Entry | What it is |
|---|---|
| `fixed_reruns_2026-11-01/` | README of the 2026-11-01 fixed reruns (folder map now in `fixed_reruns/layout.py`), `comparisons/` (`compare_fixed_reruns.py` output), `table8/{parity,seed_sweep}` (distributed-vs-single parity, seed sweep), sort A/B and unit-test log |
| `runtime_grid_old_vs_fixed/` | `compare_fix20261101.{csv,md}`: pre-fix vs fixed-image Vertex grid (`runtime_reruns/compare_runtime_grid.py`, reproduces it byte-identically) |
| `barrier_smoke_inprocess_20261022/` | first barrier measurement, 4 in-process workers on a 100k x 256 trunk (`run_barrier_cell.sh`), method note |

Known gap: rerunning `compare_fixed_reruns.py` blanks the `paper` column of `table8_paper_vs_fixed.csv`, because the
pre-fix distributed `suite_results.csv` it read was deleted on 2026-11-04 (a835d769, recoverable from 185b8b53). The
committed `comparisons/` files predate that deletion.
