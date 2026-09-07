## A/B runtime: upstream-main-benchmarks 9bd86619 (upstream/main df16834d + tooling) vs upstream-main-benchmarks-gbt-high-dim-fix 368162ea +--config=skip_dead_axis_jobs

| dataset | algorithm | A median s | B median s | speedup A/B | time saved | verdict |
|---|---|---:|---:|---:|---:|---|
| HIGGS_with_header | SPO-GBT_Exact | 1759.77 ± 4.277152 | 1726.80 ± 0.907542 | 1.02× | +1.9 % | no significant change |
| trunk_1500000_x_4096 | SPO-GBT_Exact | 436.87 ± 1.160622 | 406.19 ± 17.096792 | 1.08× | +7.0 % | no significant change |
| trunk_150000_x_40000 | SPO-GBT_Exact | 737.11 ± 18.702124 | 59.34 ± 0.157366 | 12.42× | +91.9 % | ★ speedup |
| trunk_15000_x_40000 | SPO-GBT_Exact | 765.55 ± 1.109079 | 10.37 ± 0.018915 | 73.82× | +98.6 % | ★ speedup |

Geometric-mean speedup over 4 dataset(s): **5.63×** (+82.2 % time saved). Gates: failed < 15 %, ★ ≥ 20 %.

### Provenance

| field | A | B |
|---|---|---|
| date_utc | 2026-09-07T05:58:06Z | 2026-09-07T09:21:31Z |
| git_sha | 9bd86619 | 368162ea |
| git_branch | upstream-main-benchmarks | upstream-main-benchmarks-gbt-high-dim-fix |
| machine | Intel(R) Xeon(R) Platinum 8488C (nproc=48) | Intel(R) Xeon(R) Platinum 8488C (nproc=48) |
| compiler | Intel(R) oneAPI DPC++/C++ Compiler 2025.2.1 (2025.2.0.20250806) | Intel(R) oneAPI DPC++/C++ Compiler 2025.2.1 (2025.2.0.20250806) |
| EXTRA_BAZEL_CONFIGS | <none> | --config=skip_dead_axis_jobs |
| EXTRA_TRAIN_ARGS | --ensemble_method Boosting --numerical_split_type "Exact" | --ensemble_method Boosting --numerical_split_type "Exact" |
| NUM_TREES | 300  NUM_RUNS: 3 | 300  NUM_RUNS: 3 |
| NUM_RUNS | 3 | 3 |
| CSV_DATASETS | benchmarks/data/HIGGS_with_header.csv|class | benchmarks/data/HIGGS_with_header.csv|class |
| TRUNK_DATASETS | 1500000|4096 150000|40000 15000|40000 | 1500000|4096 150000|40000 15000|40000 |
