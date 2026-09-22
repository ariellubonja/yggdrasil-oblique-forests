# Largest row count that fits the default 240-tree run — large tabular datasets (2026-09-21)

Box: m7i.metal-24xl, 48 cores (SMT off), 377 GB RAM. Harness `examples/train_oblique_forest`
defaults: Bagging, 240 trees, 48 threads, Dynamic Random Histogram 64 bins (threshold 250),
unlimited depth, `min_examples=1` (trees grow to purity), seed 1. Runs are timing runs: no
`--test_csv`, so the harness exits right after the `Training block` timer (no YDF
post-processing). A watcher killed the harness when `MemAvailable` fell below 8 GB.
Protocol (user directive): run each dataset at full size, halve the row count (earliest-K
prefix of the chronological train CSV) until a run completes without a kill.

Peak RSS = `/usr/bin/time -v` maximum resident set size. Training block = the
`random_forest.cc Training block took:` timer.

| dataset (train CSV) | features | rows tried | share of full | peak RSS | training block | outcome |
|---|---|---|---|---|---|---|
| Numerai v5.0 `numerai/numerai_train.csv` | 2376 | 2,746,268 | 1 (full) | 116 GB | 561 s | **fits** |
| Airline 2018-2023 `airline/airline_train.csv` | 43 | 37,892,905 | 1 (full) | 342 GB | 1007 s | **fits** (35 GB headroom) |
| NYC taxi 2022-01..2024-06 `nyc_taxi/nyc_taxi_train.csv` | 18 | 98,298,417 | 1 (full) | 354 GB | killed at 23 min (~tree 50/240) | OOM |
| NYC taxi | 18 | 49,149,208 | 1/2 | 276 GB | 1070 s | **fits** |
| Criteo day 2015-02-15 `criteo/criteo_train.csv` | 39 | 195,841,983 | 1 (full) | 350 GB | killed at 19 min | OOM |
| Criteo | 39 | 97,920,991 | 1/2 | 331 GB | killed at 23 min | OOM |
| Criteo | 39 | 48,960,495 | 1/4 | 341 GB | killed at 25 min | OOM |
| Criteo | 39 | 24,480,247 | 1/8 | 183 GB | 747 s | **fits** |

Largest that fits: Numerai full, airline full, NYC taxi 1/2 (49.1M rows), Criteo 1/8 (24.5M rows;
the 1/4 run died late, so the true Criteo limit lies between 24.5M and 49M).

The exact row prefixes that fit are kept next to their datasets (gitignored, regenerate with
`head -n <rows+1> <train.csv>`): `nyc_taxi/nyc_taxi_train_49149208.csv`,
`criteo/criteo_train_24480247.csv`. Numerai and airline used the full train CSVs. The runs
themselves used identical `head` prefixes in the session scratchpad (`/tmp`, a tmpfs on this box):
failed prefixes were deleted after their run and the rest went with the scratchpad, hence the copies.
Note that a prefix on tmpfs occupies RAM during its own run (up to 18 GB for Criteo 1/2), so the
effective budget of the killed runs was that much lower than 377 GB.

Why RAM, not the data, is the limit: the forest holds ≈ one node per distinct bootstrap row per
tree × 240 trees (Bagging trains to purity), i.e. 5–25 kB of model per training row depending on
how separable the data is. Calibration: HIGGS 10.5M × 28 (1.2 GB of data) peaks at 267 GB
(training block 279 s, post-processing 356 s). Criteo is the most expensive per row (≈7 kB) —
its hashed-categorical ordinal codes are near-unique, so trees split all the way down.

Post-processing (YDF finalisation, single-threaded; only paid with `--test_csv`/`--model_out_dir`)
scales with features × nodes: Numerai full 4996 s vs 561 s of training (1M-row validation prefix:
accuracy 0.5950, AUC 0.6341, logloss 0.6665); YouTube-8M 810 s; HIGGS 356 s.

Dataset construction (targets, dropped columns, encodings) is documented in each
`benchmarks/data/download_<name>.py` docstring and the `<name>_meta.json` it writes; see also
`benchmarks/SETUP_FRESH_BOX.md` step 8.
