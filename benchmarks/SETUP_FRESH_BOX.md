# Fresh AWS box setup (Ubuntu, m7i / m7i.metal)

Environment from scratch: shell, toolchain, repo build, datasets. Copied from the
"AWS setup" Google Doc on 2026-09-21 and kept here as the source of truth.
Companion docs: `oblique_context/build_measure.md` (bazel configs, harness flags),
`benchmarks/src/spo_vs_gbt/REPLICABILITY.md` §10 (regenerating the study inputs).

## 1. tmux (auto-attach on SSH)

```bash
echo "set -g mouse on" > ~/.tmux.conf
tmux

cat >> ~/.bashrc <<'EOF'

# Auto-attach to tmux on interactive SSH logins
if command -v tmux &>/dev/null && [ -n "$PS1" ] && [ -z "$TMUX" ] && [ -n "$SSH_CONNECTION" ]; then
  tmux attach -t main 2>/dev/null || tmux new -s main
fi
EOF
```

## 2. Toolchain

```bash
sudo apt update
sudo apt-get install -y build-essential libc6-dev linux-libc-dev unzip

# bazelisk: bazel picks the version from .bazelversion
wget https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel
sudo chmod +x /usr/local/bin/bazel

# Intel oneAPI (icx/icpx). .bazelrc pins CC=icx on linux; gcc is 30-40 % slower on the hot path.
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/3b7a16b3-a7b0-460f-be16-de0d64fa6b1e/intel-oneapi-base-toolkit-2025.2.1.44_offline.sh
sudo sh ./intel-oneapi-base-toolkit-2025.2.1.44_offline.sh -a --cli --eula accept
```

## 3. Repo and build

```bash
git clone https://github.com/ariellubonja/yggdrasil-oblique-forests/
cd yggdrasil-oblique-forests
bazel build -c opt //examples:train_oblique_forest
```

If icx is not on PATH for bazel, pass it explicitly:
`--repo_env=CC=/opt/intel/oneapi/compiler/latest/bin/icx --repo_env=CXX=/opt/intel/oneapi/compiler/latest/bin/icpx`
(and the same two as `--action_env`).

## 4. Disable hyperthreading (48 cores, not 96)

- Non-metal instances: Stop instance → Instance Settings → CPU Options → Threads per core = 1.
- Metal instances:

```bash
sudo tee /etc/default/grub.d/99-disable-smt.cfg >/dev/null <<'EOF'
GRUB_CMDLINE_LINUX="$GRUB_CMDLINE_LINUX nosmt"
EOF
sudo update-grub
sudo reboot
```

Verify with `htop`, then `bazel shutdown` so the bazel server re-reads the core count.

## 5. Python env (uv)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env      # or restart the shell

cd ~/yggdrasil-oblique-forests
uv venv .venv
source .venv/bin/activate
uv pip install -r benchmarks/data/requirements.txt
uv pip install certifi pyarrow    # download_epsilon.py
```

## 6. Datasets

### 6a. HIGGS / SUSY (manual, UCI)

```bash
cd benchmarks/data
wget https://archive.ics.uci.edu/static/public/280/higgs.zip && unzip higgs.zip && gzip -d HIGGS.csv.gz
wget https://archive.ics.uci.edu/static/public/279/susy.zip  && unzip susy.zip  && gzip -d SUSY.csv.gz

# add headers
echo "class,$(printf 'feature%d,' {1..28} | sed 's/,$//')" > header.csv
cat header.csv HIGGS.csv > HIGGS_with_header.csv
echo "class,$(printf 'feature%d,' {1..18} | sed 's/,$//')" > header.csv
cat header.csv SUSY.csv > SUSY_with_header.csv
rm header.csv
```

HIGGS is only ever used at full size. The 10.5M/500k and 4.5M/500k train/test splits
used by the SPO-vs-GBT study are derived from these files as described in
`benchmarks/src/spo_vs_gbt/REPLICABILITY.md` §10 (note the label-token `sed` for the test split).

### 6b. Everything else: the download scripts under `benchmarks/data`

Run from the repo root with the venv active. Each script documents its output layout
in its docstring.

```bash
python3 benchmarks/data/download_cc18_datasets.py       # OpenML CC18 binary tasks → cc18_binary_csv/ (pinned by cc18_manifest.json)
python3 benchmarks/data/download_tabarena_datasets.py   # TabArena-v0.1 (51 OpenML tasks) → tabarena/<name>/data.parquet
python3 benchmarks/data/download_epsilon.py             # HF jxie/epsilon-normalized → epsilon_normalized_train.csv
python3 benchmarks/data/download_tabred_datasets.py     # TabReD → tabred/ ; needs ~/.kaggle/kaggle.json + accepted competition rules
```

After TabArena downloads, materialise the all-numeric CSVs the harness reads
(`tabarena_binary_csv/<name>/train.csv`) with `benchmarks/data/tabular_suite_prep.py`
(`write_dataset`), driven by `tabarena_binary_manifest.json`; see REPLICABILITY.md §10 for the
exact call. Preprocessing rule (categoricals → ordinal codes, NaNs → train-fold mean) is in
the repo `CLAUDE.md`.
