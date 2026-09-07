## A/B runtime: rebased-main tip 5cf2e9a4, no config vs +--config=skip_dead_axis_jobs

| dataset | algorithm | A median s | B median s | speedup A/B | time saved | verdict |
|---|---|---:|---:|---:|---:|---|
| HIGGS_with_header | SPO-GBT_Dynamic_Random_Histogram | 140.38 ± n/a | 135.07 ± n/a | 1.04× | +3.8 % | no significant change |
| trunk_1500000_x_4096 | SPO-GBT_Dynamic_Random_Histogram | 33.07 ± n/a | 31.23 ± n/a | 1.06× | +5.6 % | no significant change |
| trunk_150000_x_40000 | SPO-GBT_Dynamic_Random_Histogram | 78.11 ± n/a | 4.89 ± n/a | 15.96× | +93.7 % | ★ speedup |
| trunk_15000_x_400000 | SPO-GBT_Dynamic_Random_Histogram | 745.00 ± n/a | 5.73 ± n/a | 130.03× | +99.2 % | ★ speedup |

Geometric-mean speedup over 4 dataset(s): **6.91×** (+85.5 % time saved). Gates: failed < 15 %, ★ ≥ 20 %.

### Provenance

| field | A | B |
|---|---|---|
| date_utc | 2026-09-06T23:49:38Z | 2026-09-07T00:11:44Z |
| git_sha | 5cf2e9a4 | 5cf2e9a4 |
| git_branch | gbt-high-dim-speedup | gbt-high-dim-speedup |
| machine | Intel(R) Xeon(R) Platinum 8488C (nproc=48) | Intel(R) Xeon(R) Platinum 8488C (nproc=48) |
| compiler | Intel(R) oneAPI DPC++/C++ Compiler 2025.2.1 (2025.2.0.20250806) | Intel(R) oneAPI DPC++/C++ Compiler 2025.2.1 (2025.2.0.20250806) |
| EXTRA_BAZEL_CONFIGS | <none> | --config=skip_dead_axis_jobs |
| EXTRA_TRAIN_ARGS | --ensemble_method Boosting --numerical_split_type "Dynamic Random Histogram" | --ensemble_method Boosting --numerical_split_type "Dynamic Random Histogram" |
| NUM_TREES | 30  NUM_RUNS: 1 | 30  NUM_RUNS: 1 |
| NUM_RUNS | 1 | 1 |
| CSV_DATASETS | benchmarks/data/HIGGS_with_header.csv|class | benchmarks/data/HIGGS_with_header.csv|class |
| TRUNK_DATASETS | 1500000|4096 150000|40000 15000|400000 | 1500000|4096 150000|40000 15000|400000 |
