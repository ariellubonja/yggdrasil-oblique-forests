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

## Recreating the exact training inputs

Numerai and airline trained on the full train CSVs. Taxi and Criteo trained on earliest-K
prefixes of the chronological train CSV; these are the exact commands (K+1 lines = header + K rows):

```bash
cd benchmarks/data
head -n 49149209 nyc_taxi/nyc_taxi_train.csv > nyc_taxi/nyc_taxi_train_49149208.csv   # taxi 1/2, 3.2 GB
head -n 24480248 criteo/criteo_train.csv     > criteo/criteo_train_24480247.csv        # Criteo 1/8, 4.4 GB
```

Both files are gitignored (the whole `benchmarks/data/<name>/` folders are) and are also
produced by `benchmarks/setup_fresh_box.sh` (step "Large tabular"). The runs themselves used
byte-identical `head` prefixes in the session scratchpad (`/tmp`, a tmpfs on this box): failed
prefixes were deleted after their run and the rest went with the scratchpad, hence these copies.
Note that a prefix on tmpfs occupies RAM during its own run (up to 18 GB for Criteo 1/2), so the
effective budget of the killed runs was that much lower than 377 GB.

The run command per row (harness defaults; add `--test_csv=<name>/<name>_test.csv` only when
accuracy is wanted, which also switches on YDF's post-processing):

```bash
./bazel-bin/examples/train_oblique_forest --input_mode=csv --label_col=class \
    --train_csv=benchmarks/data/nyc_taxi/nyc_taxi_train_49149208.csv
```

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

## 2026-09-24 — Shifts weather, ClimSim, Jane Street

Same box, same protocol and harness defaults (plain `-c opt` build, no chrono; memwatch kill at
MemAvailable < 8 GB; peak RSS from `/usr/bin/time -v`). Dataset construction: `download_shifts_weather.py`,
`download_climsim.py`, `download_jane_street.py` (targets binarised at the train median; see each docstring
and `<name>_meta.json`).

| dataset (train CSV) | features | rows tried | share of full | peak RSS | training block | outcome |
|---|---|---|---|---|---|---|
| Shifts weather `shifts_weather/shifts_weather_train.csv` | 127 | 3,129,592 | 1 (full) | 17 GB | 72 s | **fits** |
| ClimSim subsampled low-res `climsim/climsim_train.csv` | 124 | 10,091,520 | 1 (full) | 39 GB | 204 s | **fits** |
| Jane Street 2024 `jane_street/jane_street_train.csv` | 82 | 40,852,762 | 1 (full) | 364 GB | killed at 12 min | OOM |
| Jane Street | 82 | 20,426,381 | 1/2 | 365 GB | killed at 14 min | OOM |
| Jane Street | 82 | 10,213,190 | 1/4 | 260 GB | 467 s | **fits** |

Weather and ClimSim are cheap because their labels (temperature > median; downward longwave flux > median)
are almost determined by a few features (forecast temperatures; surface-level state_t), so trees reach purity
early and the forests stay small. Jane Street's label (responder_6 > median) is near-noise, so trees grow to
full purity: ≈25 kB per training row, the HIGGS regime.

```bash
cd benchmarks/data
head -n 10213191 jane_street/jane_street_train.csv > jane_street/jane_street_train_10213190.csv   # Jane Street 1/4, 8.4 GB
```
