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
   - `download_tabred_datasets.py` → `tabred/` — **only if `~/.kaggle/kaggle.json` exists**
     (needs a Kaggle token and accepted competition rules; drop the token in and re-run the
     script for this step).
