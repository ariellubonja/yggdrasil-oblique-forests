#!/usr/bin/env bash
# Fresh AWS box setup (Ubuntu, m7i / m7i.metal) — fully unattended.
#
# Runs every step of benchmarks/SETUP_FRESH_BOX.md without prompts. It lives in
# the repo, so it assumes the repo is already cloned and runs from any cwd:
#
#   bash benchmarks/setup_fresh_box.sh 2>&1 | tee ~/setup_fresh_box.log
#
# Every step is idempotent (re-running skips what is already done). Knobs:
#   SKIP_DATASETS=1   toolchain + build only, no dataset downloads
#   SKIP_BUILD=1      no bazel build
#   SKIP_SMT=1        leave hyperthreading alone
#   ONEAPI_URL=...    override the oneAPI offline installer URL
#
# Deliberate departures from the interactive doc:
#   - no `git clone` (this script is in the clone)
#   - tmux is configured, not started (starting it would block the script)
#   - oneAPI installs with `--silent` (the doc's `--cli` is an interactive wizard),
#     compiler + VTune + Advisor only (the profiler tooling needs the latter two),
#     not the full ~20 GB toolkit
#   - hyperthreading is turned off at runtime via sysfs *and* persisted in grub,
#     so no reboot is needed now and the setting survives one
#   - TabReD only downloads if ~/.kaggle/kaggle.json exists (needs a token +
#     accepted competition rules — nothing a script can do)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DL_DIR="${DL_DIR:-$HOME/.cache/fresh_box_downloads}"
ONEAPI_URL="${ONEAPI_URL:-https://registrationcenter-download.intel.com/akdlm/IRC_NAS/3b7a16b3-a7b0-460f-be16-de0d64fa6b1e/intel-oneapi-base-toolkit-2025.2.1.44_offline.sh}"
ONEAPI_ROOT=/opt/intel/oneapi
export DEBIAN_FRONTEND=noninteractive

log() { printf '\n[%s] === %s\n' "$(date +%H:%M:%S)" "$*"; }

# Status table, printed on exit (also on failure). mark <step> <status> [detail]
STEPS=("tmux, apt, bazelisk" "Intel oneAPI (icx, VTune, Advisor)" "SMT off" "bazel build harness"
       "uv + .venv" "HIGGS / SUSY" "CC18" "TabArena + all-numeric CSVs" "EPSILON" "TabReD")
declare -A STATUS
CURRENT_STEP=""
mark()  { STATUS[$1]=$2${3:+ ($3)}; }
begin() { CURRENT_STEP=$1; log "$1"; }
print_summary() {
  local rc=$?
  [[ $rc -ne 0 && -n "$CURRENT_STEP" ]] && STATUS[$CURRENT_STEP]="FAILED (exit $rc, see log above)"
  printf '\n%-40s | %s\n' "Step" "Status"
  printf '%s\n' "$(printf -- '-%.0s' {1..40}) | $(printf -- '-%.0s' {1..40})"
  for st in "${STEPS[@]}"; do printf '%-40s | %s\n' "$st" "${STATUS[$st]:-not run}"; done
  [[ $rc -eq 0 ]] && echo && echo "done — open a new shell (or: source ~/.bashrc) to pick up oneAPI, uv and tmux"
  exit $rc
}
trap print_summary EXIT
append_once() {  # append_once <file> <marker> <<'EOF' ... EOF
  local file=$1 marker=$2
  grep -qF "$marker" "$file" 2>/dev/null || cat >> "$file"
}
mkdir -p "$DL_DIR"

# ---------------------------------------------------------------- 1. tmux
begin "tmux, apt, bazelisk"
grep -qxF 'set -g mouse on' ~/.tmux.conf 2>/dev/null || echo 'set -g mouse on' >> ~/.tmux.conf
append_once ~/.bashrc '# Auto-attach to tmux on interactive SSH logins' <<'EOF'

# Auto-attach to tmux on interactive SSH logins
if command -v tmux &>/dev/null && [ -n "$PS1" ] && [ -z "$TMUX" ] && [ -n "$SSH_CONNECTION" ]; then
  tmux attach -t main 2>/dev/null || tmux new -s main
fi
EOF

# ---------------------------------------------------------------- 2. toolchain
sudo -E apt-get update -qq
sudo -E apt-get install -y -qq build-essential libc6-dev linux-libc-dev unzip wget curl git gh tmux htop

if [[ ! -x /usr/local/bin/bazel ]]; then
  wget -q -O "$DL_DIR/bazelisk-linux-amd64" \
    https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
  sudo install -m 0755 "$DL_DIR/bazelisk-linux-amd64" /usr/local/bin/bazel
fi
bazel --version
mark "tmux, apt, bazelisk" done "$(bazel --version)"

begin "Intel oneAPI (icx, VTune, Advisor)"
if [[ ! -x "$ONEAPI_ROOT/compiler/latest/bin/icx" ]]; then
  installer="$DL_DIR/$(basename "$ONEAPI_URL")"
  wget -q -c -O "$installer" "$ONEAPI_URL"
  sudo sh "$installer" -a --silent --eula accept \
    --components intel.oneapi.lin.dpcpp-cpp-compiler:intel.oneapi.lin.vtune:intel.oneapi.lin.advisor \
    --install-dir "$ONEAPI_ROOT"
fi
# .bazelrc pins CC=icx; the repo's bench scripts source setvars.sh themselves.
append_once ~/.bashrc '# Intel oneAPI (icx/icpx for bazel)' <<'EOF'

# Intel oneAPI (icx/icpx for bazel)
[ -r /opt/intel/oneapi/setvars.sh ] && source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
EOF
set +u; source "$ONEAPI_ROOT/setvars.sh" --force >/dev/null 2>&1; set -u
icx --version | head -1
mark "Intel oneAPI (icx, VTune, Advisor)" done "$(icx --version | head -1 | grep -oE "[0-9]+\.[0-9]+\.[0-9]+" | head -1)"

# ---------------------------------------------------------------- 3. hyperthreading off
if [[ "${SKIP_SMT:-0}" == 1 ]]; then
  mark "SMT off" skipped "SKIP_SMT=1"
else
  begin "SMT off"
  if [[ "$(cat /sys/devices/system/cpu/smt/control 2>/dev/null)" == on ]]; then
    echo off | sudo tee /sys/devices/system/cpu/smt/control >/dev/null
  fi
  if [[ ! -f /etc/default/grub.d/99-disable-smt.cfg ]]; then
    sudo mkdir -p /etc/default/grub.d
    printf 'GRUB_CMDLINE_LINUX="$GRUB_CMDLINE_LINUX nosmt"\n' | sudo tee /etc/default/grub.d/99-disable-smt.cfg >/dev/null
    sudo update-grub >/dev/null 2>&1 || echo "update-grub failed (non-grub image?); runtime SMT-off still applied"
  fi
  bazel shutdown >/dev/null 2>&1 || true   # bazel server re-reads the core count
  echo "online CPUs: $(nproc)"
  mark "SMT off" done "$(nproc) CPUs online, grub persisted"
fi

# ---------------------------------------------------------------- 4. build
if [[ "${SKIP_BUILD:-0}" == 1 ]]; then
  mark "bazel build harness" skipped "SKIP_BUILD=1"
else
  begin "bazel build harness"
  cd "$REPO_ROOT"
  bazel build -c opt //examples:train_oblique_forest
  ls -l bazel-bin/examples/train_oblique_forest
  mark "bazel build harness" done "bazel-bin/examples/train_oblique_forest"
fi

# ---------------------------------------------------------------- 5. python env (uv)
begin "uv + .venv"
if [[ ! -x "$HOME/.local/bin/uv" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh -s -- -q
fi
export PATH="$HOME/.local/bin:$PATH"
cd "$REPO_ROOT"
[[ -x .venv/bin/python ]] || uv venv -q --python 3.12 .venv
uv pip install -q --python .venv/bin/python -r benchmarks/data/requirements.txt certifi pyarrow
VENV_PY="$REPO_ROOT/.venv/bin/python"
mark "uv + .venv" done "$("$VENV_PY" --version)"

# ---------------------------------------------------------------- 6. datasets
if [[ "${SKIP_DATASETS:-0}" == 1 ]]; then
  for st in "HIGGS / SUSY" "CC18" "TabArena + all-numeric CSVs" "EPSILON" "TabReD"; do mark "$st" skipped "SKIP_DATASETS=1"; done
  exit 0
fi

begin "HIGGS / SUSY"
cd "$REPO_ROOT/benchmarks/data"
uci_dataset() {  # uci_dataset <NAME> <uci id> <n features>
  local name=$1 id=$2 nfeat=$3
  [[ -s "${name}_with_header.csv" ]] && { echo "${name}_with_header.csv present"; return; }
  local lower; lower=$(echo "$name" | tr '[:upper:]' '[:lower:]')
  [[ -s "$name.csv" ]] || {
    wget -q -c -O "$lower.zip" "https://archive.ics.uci.edu/static/public/$id/$lower.zip"
    unzip -o -q "$lower.zip"
    gzip -d -f "$name.csv.gz"
  }
  { echo "class,$(seq -s, -f 'feature%g' 1 "$nfeat")"; cat "$name.csv"; } > "${name}_with_header.csv"
  rm -f "$lower.zip" "$name.csv"
}
uci_dataset HIGGS 280 28
uci_dataset SUSY  279 18
mark "HIGGS / SUSY" done "$(du -sh HIGGS_with_header.csv | cut -f1), $(du -sh SUSY_with_header.csv | cut -f1)"

cd "$REPO_ROOT"
begin "CC18"
if compgen -G "benchmarks/data/cc18_binary_csv/task_*/*_train.csv" >/dev/null; then
  echo "cc18_binary_csv present"
else
  "$VENV_PY" benchmarks/data/download_cc18_datasets.py
fi
mark "CC18" done "$(ls -d benchmarks/data/cc18_binary_csv/task_* | wc -l) tasks"

begin "TabArena + all-numeric CSVs"
"$VENV_PY" benchmarks/data/download_tabarena_datasets.py

"$VENV_PY" - <<'PY'
import json, os, sys
import pandas as pd
sys.path.insert(0, "benchmarks/data")
from tabular_suite_prep import write_dataset
src, out = "benchmarks/data/tabarena", "benchmarks/data/tabarena_binary_csv"
for e in json.load(open("benchmarks/data/tabarena_binary_manifest.json"))["entries"]:
    if e["status"] != "ok":
        continue
    name, target = e["dataset_name"], e["target_name"]
    if os.path.exists(os.path.join(out, name, "train.csv")):
        continue
    df = pd.read_parquet(os.path.join(src, name, "data.parquet"))
    meta = write_dataset(out, name, df.drop(columns=[target]), df[target])
    print(f"{name}: {meta['rows']} rows x {meta['features']} features")
PY
mark "TabArena + all-numeric CSVs" done "$(ls benchmarks/data/tabarena_binary_csv/*/train.csv | wc -l)/$("$VENV_PY" -c 'import json;print(sum(e["status"]=="ok" for e in json.load(open("benchmarks/data/tabarena_binary_manifest.json"))["entries"]))') train.csv"

begin "EPSILON"
if [[ -s benchmarks/data/epsilon_normalized_train.csv ]]; then
  echo "epsilon_normalized_train.csv present"
else
  "$VENV_PY" benchmarks/data/download_epsilon.py
fi
mark "EPSILON" done "$(du -sh benchmarks/data/epsilon_normalized_train.csv | cut -f1)"

begin "TabReD"
if [[ -r "$HOME/.kaggle/kaggle.json" ]]; then
  uv pip install -q --python .venv/bin/python kaggle
  "$VENV_PY" benchmarks/data/download_tabred_datasets.py
  mark "TabReD" done
else
  mark "TabReD" skipped "no ~/.kaggle/kaggle.json: needs a Kaggle token + accepted competition rules"
fi
