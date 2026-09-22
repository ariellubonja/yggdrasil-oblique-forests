# Fresh AWS box setup (Ubuntu, m7i / m7i.metal)

Environment from scratch: shell, toolchain, repo build, datasets. Originally copied from the
"AWS setup" Google Doc on 2026-09-21; turned into an unattended script the same day.
Companion docs: `oblique_context/build_measure.md` (bazel configs, harness flags),
`benchmarks/src/spo_vs_gbt/REPLICABILITY.md` §10 (regenerating the study inputs).

## Run it

The only manual step is getting the repo onto the box; everything else is
`benchmarks/setup_fresh_box.sh`, which runs without prompts:

```bash
git clone https://github.com/ariellubonja/yggdrasil-oblique-forests/
cd yggdrasil-oblique-forests
bash benchmarks/setup_fresh_box.sh 2>&1 | tee ~/setup_fresh_box.log
```

Then open a new shell (or `source ~/.bashrc`) to pick up oneAPI, uv and tmux.
The script is idempotent — re-running it skips whatever is already in place (a re-run on a
finished box takes ~10 s) — and takes these knobs: `SKIP_DATASETS=1` (toolchain + build only),
`SKIP_BUILD=1`, `SKIP_SMT=1`, `ONEAPI_URL=...`.

It ends with a status table, printed on every exit including failures (the failing step is
marked `FAILED`, later ones `not run`). A full first run on an m7i.metal-24xl (2026-09-21)
took ~35 min, dominated by the EPSILON merge and the oneAPI download:

```text
Step                                     | Status
---------------------------------------- | ----------------------------------------
tmux, apt, bazelisk                      | done (bazel 7.7.0)
Intel oneAPI (icx, VTune, Advisor)       | done (2025.2.1)
SMT off                                  | done (48 CPUs online, grub persisted)
bazel build harness                      | done (bazel-bin/examples/train_oblique_forest)
uv + .venv                               | done (Python 3.12.14)
HIGGS / SUSY                             | done (7.5G, 2.3G)
CC18                                     | done (34 tasks)
TabArena + all-numeric CSVs              | done (30/30 train.csv)
EPSILON                                  | done (15G)
YouTube-8M                               | done (56G, 16G; added 2026-09-21, ~5 min on m7i.metal, not in the 35 min)
Criteo / NYC taxi / Airline / Numerai    | done (37G+0.6G, 6.7G+1.5G, 6.5G+0.3G, 13G+20G; added 2026-09-21, ~25 min total, not in the 35 min)
TabReD                                   | skipped (no ~/.kaggle/kaggle.json: needs a Kaggle token + accepted competition rules)
```

## What it does, in order

1. **tmux** — `set -g mouse on` in `~/.tmux.conf`, auto-attach to session `main` on SSH
   logins (appended to `~/.bashrc`). tmux is configured, not started.
2. **apt** — `build-essential libc6-dev linux-libc-dev unzip wget curl git gh tmux htop`,
   non-interactive.
3. **bazelisk** → `/usr/local/bin/bazel` (bazel picks the version from `.bazelversion`).
4. **Intel oneAPI** (icx/icpx; `.bazelrc` pins `CC=icx` on Linux, gcc is 30-40 % slower on
   the hot path). Offline installer 2025.2.1.44, `--silent --eula accept`, components
   `dpcpp-cpp-compiler`, `vtune`, `advisor` only (the `--config=profiler` and roofline tooling
   use the latter two) instead of the full ~20 GB toolkit.
   `source /opt/intel/oneapi/setvars.sh` is appended to `~/.bashrc`; the repo's bench scripts
   source it themselves. If icx is ever not on PATH for bazel, pass it explicitly:
   `--repo_env=CC=/opt/intel/oneapi/compiler/latest/bin/icx --repo_env=CXX=.../icpx`.
5. **Hyperthreading off** (48 cores, not 96) — written to
   `/sys/devices/system/cpu/smt/control` immediately *and* persisted as `nosmt` in
   `/etc/default/grub.d/99-disable-smt.cfg`, so no reboot is needed and it survives one.
   Works on metal and non-metal instances alike (no console visit). `bazel shutdown` follows
   so the server re-reads the core count.
6. **Build** — `bazel build -c opt //examples:train_oblique_forest`.
7. **Python env** — uv, `.venv` on Python 3.12, `benchmarks/data/requirements.txt`
   + `certifi pyarrow` (for `download_epsilon.py`).
8. **Datasets** (all into `benchmarks/data/`):
   - HIGGS / SUSY from UCI (`higgs.zip` id 280, `susy.zip` id 279) → `HIGGS_with_header.csv`,
     `SUSY_with_header.csv` (header `class,feature1..featureN`). HIGGS is only ever used at
     full size; the 10.5M/500k and 4.5M/500k train/test splits used by the SPO-vs-GBT study
     are derived from these files as in REPLICABILITY.md §10 (note the label-token `sed` for
     the test split).
   - `download_cc18_datasets.py` → `cc18_binary_csv/` (pinned by `cc18_manifest.json`).
   - `download_tabarena_datasets.py` → `tabarena/<name>/data.parquet`, then
     `tabular_suite_prep.write_dataset` for every `status: ok` entry of
     `tabarena_binary_manifest.json` → `tabarena_binary_csv/<name>/train.csv` (the all-numeric
     CSVs the harness reads; preprocessing rule in the repo `CLAUDE.md`).
   - `download_epsilon.py` → `epsilon_normalized_train.csv`.
   - `download_youtube8m.py` → `youtube8m/youtube8m_video_{train,validate}.csv` (YouTube-8M 2018
     video-level features, https://research.google.com/youtube8m/download.html, CC BY 4.0):
     3,844 TFRecord shards per partition (18 GB train, 4 GB validate) pulled in parallel
     from `us.data.yt8m.org` with the official plan JSON's MD5 per shard, then decoded
     with a built-in protobuf reader (no TensorFlow) into `class,rgb_0..rgb_1023,
     audio_0..audio_127` (fp32 written `%.9g`, exact round-trip; train 3,888,919 rows × 1152 =
     56 GB, validate 1,112,356 × 1152 = 16 GB). Binary target per the majority-vs-rest rule
     (CLAUDE.md dataset policy): `class=1` iff entity id 0 — the most frequent label, "Game",
     20.3 % of videos in both partitions (entity 1 is next at 13.9 %) — is among the video's
     labels; the full multi-label list is kept in `youtube8m/youtube8m_video_<part>_labels.csv`
     (`id,labels`, same row order) so other targets can be derived without re-downloading
     (`--target_label N`). Everything lives under `benchmarks/data/youtube8m/` (gitignored);
     shards stay in `youtube8m/tfrecord_temp/` (22 GB; re-runs skip verified ones). On the m7i.metal: ~40 s download + ~100 s conversion per partition. `test` has no labels and is not downloaded. The vocabulary CSV linked on
     the download page returns AccessDenied (2026-09-21), so entity names are not resolved.
   - Large tabular sets added 2026-09-21 (each script is idempotent, writes only into
     `benchmarks/data/<name>/` (gitignored), keeps the raw download in a subfolder, applies the
     CLAUDE.md preprocessing rule — NaN → train column mean, strings → alphabetical ordinal codes,
     datetime → epoch seconds — and writes `<name>_train.csv`, `<name>_test.csv` (`class` first,
     all fp32-exact numeric, rows chronological so `head -n K` is an earliest-K prefix) plus
     `<name>_meta.json` with counts, positive rates, dropped columns, encodings and means):
     - `download_criteo.py` → `criteo/criteo_{train,test}.csv` — Criteo 1TB click logs from
       HF `criteo/CriteoClickLogs` (parquet, 24 days, 276 GB). Default: train = day 2015-02-15
       (195,841,983 × 39 features, 37 GB, 3.21 % clicks), test = first 4 parts of 2015-02-16
       (3,117,730 rows). 13 int features + 26 hashed categoricals (cardinalities up to 22M —
       codes above 2^24 collide in fp32, listed in the meta). ~10 min on the m7i.metal.
     - `download_nyc_taxi.py` → `nyc_taxi/nyc_taxi_{train,test}.csv` — TLC yellow-taxi monthly
       parquet (CloudFront). Default: train 2022-01..2024-06 (98,298,417 × 18, 6.7 GB), test
       2024-07..2024-12 (20,837,627). Regression target `trip_duration_s` binarised at the
       **train median (749 s)**: `class = 1` iff longer. Dropoff time dropped, pickup → epoch
       seconds (fp32 ⇒ 128 s quantisation). ~1 min once the parquet is down.
     - `download_airline.py` → `airline/airline_{train,test}.csv` — BTS Reporting Carrier
       On-Time Performance monthly zips (`transtats.bts.gov/PREZIP`, ~2.4 MB/s). Default: train
       2018-01..2023-12 (37,892,905 × 43, 6.5 GB, 18.3 % positive), test 2024-10..2024-12
       (1,765,279). `class = ArrDel15` (arrival delay ≥ 15 min); cancelled/diverted rows have
       no label and are dropped (2.6 %); arrival-side / diversion / delay-cause columns are
       dropped as leakage (67 of 110, listed in the meta); departure-side fields are kept.
     - `download_numerai.py` → `numerai/numerai_{train,test}.csv` — Numerai v5.0 "Atlas" via
       `numerapi` (public). train.parquet → train (2,746,268 × 2,376, 13 GB), validation.parquet
       → test (4,114,072 rows; 34,713 null-target rows dropped). `target` has 5 values
       {0,.25,.5,.75,1}; majority-vs-rest ⇒ `class = 1` iff `target == 0.5` (50.0 %). No NaNs
       in v5.0. ~6 min.
     - Fit limits for the default 240-tree run on the 377 GB m7i, and the exact `head` prefixes
       that fit (`nyc_taxi/nyc_taxi_train_49149208.csv`, `criteo/criteo_train_24480247.csv`,
       created by the setup script): `benchmarks/data/LARGE_DATASETS_240TREE_FIT.md`.
   - `download_tabred_datasets.py` → `tabred/` — **only if `~/.kaggle/kaggle.json` exists**
     (needs a Kaggle token and accepted competition rules; drop the token in and re-run the
     script for this step).
