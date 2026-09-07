## A/B runtime: A vs B

| dataset | algorithm | A median s | B median s | speedup A/B | time saved | verdict |
|---|---|---:|---:|---:|---:|---|
| trunk_15000_x_40000 | SPO-GBT_Exact | 77.22 ± n/a | 1.09 ± n/a | 70.84× | +98.6 % | ★ speedup |

Geometric-mean speedup over 1 dataset(s): **70.84×** (+98.6 % time saved). Gates: failed < 15 %, ★ ≥ 20 %.

### Provenance

| field | A | B |
|---|---|---|
| date_utc | 2026-09-07T00:57:04Z | 2026-09-07T00:58:27Z |
| git_sha | 9bd86619 | 368162ea |
| git_branch | upstream-main-benchmarks | upstream-main-benchmarks-gbt-high-dim-fix |
| machine | Intel(R) Xeon(R) Platinum 8488C (nproc=48) | Intel(R) Xeon(R) Platinum 8488C (nproc=48) |
| compiler | Intel(R) oneAPI DPC++/C++ Compiler 2025.2.1 (2025.2.0.20250806) | Intel(R) oneAPI DPC++/C++ Compiler 2025.2.1 (2025.2.0.20250806) |
| EXTRA_BAZEL_CONFIGS | <none> | --config=skip_dead_axis_jobs |
| EXTRA_TRAIN_ARGS | --ensemble_method Boosting --numerical_split_type "Exact" | --ensemble_method Boosting --numerical_split_type "Exact" |
| NUM_TREES | 30  NUM_RUNS: 1 | 30  NUM_RUNS: 1 |
| NUM_RUNS | 1 | 1 |
| CSV_DATASETS | <none> | <none> |
| TRUNK_DATASETS | 15000|40000 | 15000|40000 |
