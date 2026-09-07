## A/B runtime: upstream-main-benchmarks 9bd86619 vs upstream-main-benchmarks-gbt-high-dim-fix 368162ea +config

| dataset | algorithm | A median s | B median s | speedup A/B | time saved | verdict |
|---|---|---:|---:|---:|---:|---|
| HIGGS_with_header | SPO-GBT_Exact | 183.83 ± n/a | 176.40 ± n/a | 1.04× | +4.0 % | no significant change |
| trunk_1500000_x_4096 | SPO-GBT_Exact | 45.53 ± n/a | 41.42 ± n/a | 1.10× | +9.0 % | no significant change |
| trunk_150000_x_40000 | SPO-GBT_Exact | 73.42 ± n/a | 6.10 ± n/a | 12.04× | +91.7 % | ★ speedup |
| trunk_15000_x_400000 | SPO-GBT_Exact | 792.91 ± n/a | 14.45 ± n/a | 54.86× | +98.2 % | ★ speedup |

Geometric-mean speedup over 4 dataset(s): **5.24×** (+80.9 % time saved). Gates: failed < 15 %, ★ ≥ 20 %.

### Provenance

| field | A | B |
|---|---|---|
| date_utc | 2026-09-07T00:25:41Z | 2026-09-07T00:49:15Z |
| git_sha | 9bd86619 | 368162ea |
| git_branch | upstream-main-benchmarks | upstream-main-benchmarks-gbt-high-dim-fix |
| machine | Intel(R) Xeon(R) Platinum 8488C (nproc=48) | Intel(R) Xeon(R) Platinum 8488C (nproc=48) |
| compiler | Intel(R) oneAPI DPC++/C++ Compiler 2025.2.1 (2025.2.0.20250806) | Intel(R) oneAPI DPC++/C++ Compiler 2025.2.1 (2025.2.0.20250806) |
| EXTRA_BAZEL_CONFIGS | <none> | --config=skip_dead_axis_jobs |
| EXTRA_TRAIN_ARGS | --ensemble_method Boosting --numerical_split_type "Exact" | --ensemble_method Boosting --numerical_split_type "Exact" |
| NUM_TREES | 30  NUM_RUNS: 1 | 30  NUM_RUNS: 1 |
| NUM_RUNS | 1 | 1 |
| CSV_DATASETS | benchmarks/data/HIGGS_with_header.csv|class | benchmarks/data/HIGGS_with_header.csv|class |
| TRUNK_DATASETS | 1500000|4096 150000|40000 15000|400000 | 1500000|4096 150000|40000 15000|400000 |
