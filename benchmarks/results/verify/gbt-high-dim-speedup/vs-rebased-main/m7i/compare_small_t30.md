## A/B runtime: A vs B

| dataset | algorithm | A median s | B median s | speedup A/B | time saved | verdict |
|---|---|---:|---:|---:|---:|---|
| trunk_15000_x_40000 | SPO-GBT_Dynamic_Random_Histogram | 76.12 ± n/a | 0.95 ± n/a | 79.99× | +98.7 % | ★ speedup |

Geometric-mean speedup over 1 dataset(s): **79.99×** (+98.7 % time saved). Gates: failed < 15 %, ★ ≥ 20 %.

### Provenance

| field | A | B |
|---|---|---|
| date_utc | 2026-09-07T00:58:38Z | 2026-09-07T01:00:01Z |
| git_sha | 5cf2e9a4 | 5cf2e9a4 |
| git_branch | gbt-high-dim-speedup | gbt-high-dim-speedup |
| machine | Intel(R) Xeon(R) Platinum 8488C (nproc=48) | Intel(R) Xeon(R) Platinum 8488C (nproc=48) |
| compiler | Intel(R) oneAPI DPC++/C++ Compiler 2025.2.1 (2025.2.0.20250806) | Intel(R) oneAPI DPC++/C++ Compiler 2025.2.1 (2025.2.0.20250806) |
| EXTRA_BAZEL_CONFIGS | <none> | --config=skip_dead_axis_jobs |
| EXTRA_TRAIN_ARGS | --ensemble_method Boosting --numerical_split_type "Dynamic Random Histogram" | --ensemble_method Boosting --numerical_split_type "Dynamic Random Histogram" |
| NUM_TREES | 30  NUM_RUNS: 1 | 30  NUM_RUNS: 1 |
| NUM_RUNS | 1 | 1 |
| CSV_DATASETS | <none> | <none> |
| TRUNK_DATASETS | 15000|40000 | 15000|40000 |
