## A/B runtime: rebased-main tip 5cf2e9a4, no config vs +--config=skip_dead_axis_jobs

| dataset | algorithm | A median s | B median s | speedup A/B | time saved | verdict |
|---|---|---:|---:|---:|---:|---|
| HIGGS_with_header | SPO-GBT_Dynamic_Random_Histogram | 1334.37 ± 5.618455 | 1294.00 ± 1.822096 | 1.03× | +3.0 % | no significant change |
| trunk_1500000_x_4096 | SPO-GBT_Dynamic_Random_Histogram | 319.01 ± 1.651306 | 304.71 ± 3.181777 | 1.05× | +4.5 % | no significant change |
| trunk_150000_x_40000 | SPO-GBT_Dynamic_Random_Histogram | 738.18 ± 14.852573 | 47.83 ± 0.623909 | 15.43× | +93.5 % | ★ speedup |
| trunk_15000_x_40000 | SPO-GBT_Dynamic_Random_Histogram | 732.23 ± 8.985084 | 8.74 ± 0.061610 | 83.74× | +98.8 % | ★ speedup |

Geometric-mean speedup over 4 dataset(s): **6.11×** (+83.6 % time saved). Gates: failed < 15 %, ★ ≥ 20 %.

### Provenance

| field | A | B |
|---|---|---|
| date_utc | 2026-09-07T02:56:47Z | 2026-09-07T01:09:53Z |
| git_sha | 5cf2e9a4 | 5cf2e9a4 |
| git_branch | gbt-high-dim-speedup | gbt-high-dim-speedup |
| machine | Intel(R) Xeon(R) Platinum 8488C (nproc=48) | Intel(R) Xeon(R) Platinum 8488C (nproc=48) |
| compiler | Intel(R) oneAPI DPC++/C++ Compiler 2025.2.1 (2025.2.0.20250806) | Intel(R) oneAPI DPC++/C++ Compiler 2025.2.1 (2025.2.0.20250806) |
| EXTRA_BAZEL_CONFIGS | <none> | --config=skip_dead_axis_jobs |
| EXTRA_TRAIN_ARGS | --ensemble_method Boosting --numerical_split_type "Dynamic Random Histogram" | --ensemble_method Boosting --numerical_split_type "Dynamic Random Histogram" |
| NUM_TREES | 300  NUM_RUNS: 3 | 300  NUM_RUNS: 3 |
| NUM_RUNS | 3 | 3 |
| CSV_DATASETS | benchmarks/data/HIGGS_with_header.csv|class | benchmarks/data/HIGGS_with_header.csv|class |
| TRUNK_DATASETS | 1500000|4096 150000|40000 15000|40000 | 1500000|4096 150000|40000 15000|40000 |
