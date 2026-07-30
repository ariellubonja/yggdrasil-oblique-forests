#!/usr/bin/env bash
# Shared plumbing for benchmarks/evaluation/*.sh: icx pinning, e-core toggling,
# bazel_build + EXTRA_BAZEL_CONFIGS / EXTRA_TRAIN_ARGS, provenance headers,
# repeated-run timing with median/stddev, and CSV/log helpers.
#
# Usage (after `set -euo pipefail`, from repo root):
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "$SCRIPT_DIR/../utils/bench_common.sh"
#   BENCH_LOGFILE="$logfile"          # required before the log/timing helpers
#   BENCH_ECORE_TOGGLE=false          # optional; default true (runtime-style)
#
# Knobs read from the environment (set by the caller of the benchmark script):
#   EXTRA_BAZEL_CONFIGS  space-separated bazel flags added to every bazel_build
#   EXTRA_TRAIN_ARGS     extra train_oblique_forest flags (callers append these)
#   ONEAPI_SETVARS       oneAPI setvars.sh path (default /opt/intel/oneapi/...)

[[ -n "${BENCH_COMMON_SOURCED:-}" ]] && return 0
BENCH_COMMON_SOURCED=1

BENCH_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SET_CPU_E_FEATURES="$BENCH_REPO_ROOT/benchmarks/utils/set_cpu_e_features.sh"

# Optional space-separated extra build configs/flags injected into every
# bazel_build (e.g. EXTRA_BAZEL_CONFIGS="--config=symmetric_optimized").
# shellcheck disable=SC2206
EXTRA_BAZEL_CONFIGS_ARR=(${EXTRA_BAZEL_CONFIGS:-})
# Optional extra train_oblique_forest flags injected into every binary command
# (e.g. EXTRA_TRAIN_ARGS="--dataset_layout=flat_column").
EXTRA_TRAIN_ARGS=${EXTRA_TRAIN_ARGS:-}

# Log sink for bench_log/banner/bench_repeat_cmd; unset => stdout only.
BENCH_LOGFILE="${BENCH_LOGFILE:-}"
# true: bazel_build enables e-cores for the build, disables them for the runs.
# false: leave the CPU configuration untouched (accuracy-style runs).
BENCH_ECORE_TOGGLE="${BENCH_ECORE_TOGGLE:-true}"
# Set by bench_repeat_cmd: last command was OOM-killed / errored.
BENCH_OOM=0
BENCH_ERR=0

# ---------- logging ----------

bench_log() {
  if [[ -n "$BENCH_LOGFILE" ]]; then
    printf '%s\n' "$*" | tee -a "$BENCH_LOGFILE"
  else
    printf '%s\n' "$*"
  fi
}

banner() {
  if [[ -n "$BENCH_LOGFILE" ]]; then
    echo -e "\n\n======== $* ========\n" | tee -a "$BENCH_LOGFILE"
  else
    echo -e "\n\n======== $* ========\n"
  fi
}

# ---------- host / toolchain ----------

bench_nproc() {
  nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1
}

# .bazelrc pins --repo_env=CC=icx. If icx is not on PATH the build either
# hard-fails or (worse, across bazel/cc_configure cache states) silently falls
# back to gcc -- a gcc-built binary quietly corrupts every timing number.
ensure_icx() {
  # .bazelrc only pins CC=icx under build:linux; macOS has no such pin (Intel
  # never shipped icx for Apple Silicon), so there is nothing to ensure there.
  if [[ "$(uname -s)" != "Linux" ]]; then
    return 0
  fi
  if ! command -v icx >/dev/null 2>&1; then
    local setvars="${ONEAPI_SETVARS:-/opt/intel/oneapi/setvars.sh}"
    if [[ -r "$setvars" ]]; then
      set +u +e                       # setvars.sh is not -e/-u clean
      # shellcheck disable=SC1090
      source "$setvars" >/dev/null 2>&1
      set -u -e
    fi
  fi
  if ! command -v icx >/dev/null 2>&1 || ! command -v icpx >/dev/null 2>&1; then
    echo "ERROR: icx/icpx not on PATH and could not source oneAPI." >&2
    echo "       .bazelrc pins CC=icx; building with gcc would silently" >&2
    echo "       corrupt timings. Run 'source /opt/intel/oneapi/setvars.sh'" >&2
    echo "       (or set ONEAPI_SETVARS=/path/to/setvars.sh) and retry." >&2
    exit 1
  fi
  echo "Compiler: $(command -v icx)"
}

# $1 = --enable | --disable; $2 = "force" to ignore BENCH_ECORE_TOGGLE.
# No-op unless BENCH_ECORE_TOGGLE=true. The helper itself no-ops on any CPU
# other than the 185H, so this is safe everywhere.
bench_ecores() {
  [[ "${2:-}" == "force" || "$BENCH_ECORE_TOGGLE" == "true" ]] || return 0
  [[ -x "$SET_CPU_E_FEATURES" ]] || return 0
  sudo "$SET_CPU_E_FEATURES" "$1"
}

# Re-enable e-cores when the script exits (call once, early).
bench_restore_ecores_on_exit() {
  trap 'bench_ecores --enable' EXIT
}

# Build with EXTRA_BAZEL_CONFIGS appended, e-cores on for the build only.
bazel_build() {
  ensure_icx
  bench_ecores --enable
  bazel build "$@" "${EXTRA_BAZEL_CONFIGS_ARR[@]}"
  bench_ecores --disable
}

# ---------- output files ----------

# Refuse to clobber an existing result file.
bench_require_absent() {
  local f
  for f in "$@"; do
    if [[ -e "$f" ]]; then
      echo "ERROR: $f already exists. Use a different suffix or remove it." >&2
      exit 1
    fi
  done
}

# Like bench_require_absent, but asks before clobbering a .csv (logs are always
# recreated). Reads /dev/tty so the prompt survives a piped stdin; with no tty
# it keeps the fail-safe behavior.
confirm_overwrite() {
  local f="$1"
  [[ -e "$f" ]] || return 0
  if [[ "$f" != *.csv ]]; then
    rm -f "$f"
    return 0
  fi
  if [[ ! -t 0 && ! -e /dev/tty ]]; then
    echo "ERROR: $f already exists (no terminal to confirm overwrite). Use a different suffix or remove it." >&2
    exit 1
  fi
  local reply
  read -r -p "$f already exists. Overwrite? [y/N] " reply </dev/tty
  if [[ "$reply" =~ ^[Yy]([Ee][Ss])?$ ]]; then
    rm -f "$f"
  else
    echo "Aborting; not overwriting $f. Use a different suffix or remove it." >&2
    exit 1
  fi
}

csv_escape() {
  local value="$1"
  value="${value//\"/\"\"}"
  printf '"%s"' "$value"
}

# ---------- provenance ----------

# Emit the provenance block on stdout; each argument becomes an extra line.
# Without this a result CSV cannot be traced back to the build that produced it
# (bazel configs are invisible in the binary command line).
#   bench_provenance_block "NUM_TREES: $NUM_TREES" | tee -a "$logfile" "$metafile"
bench_provenance_block() {
  # Hardware serial for traceability. dmidecode needs root and is Linux-only;
  # keep the error text in the field rather than aborting -- best effort.
  local machine_serial
  machine_serial=$(sudo dmidecode -s system-serial-number 2>&1) \
    || machine_serial="dmidecode failed: $machine_serial"
  machine_serial="${machine_serial//$'\n'/ ; }"   # keep the field on one line
  local model=""
  if [[ "$(uname -s)" == "Linux" ]]; then
    model=$(lscpu 2>/dev/null | grep 'Model name' | sed 's/Model name:[[:space:]]*//') || true
  else
    model=$(sysctl -n machdep.cpu.brand_string 2>/dev/null) || true
  fi
  [[ -n "$model" ]] || model="unknown ($(uname -m))"
  echo "==== PROVENANCE ===="
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_sha: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo '-dirty')"
  echo "git_branch: $(git branch --show-current 2>/dev/null || echo unknown)"
  echo "machine: $model (nproc=$(bench_nproc))"
  echo "machine_serial: $machine_serial"
  echo "EXTRA_BAZEL_CONFIGS: ${EXTRA_BAZEL_CONFIGS:-<none>}"
  echo "EXTRA_TRAIN_ARGS: ${EXTRA_TRAIN_ARGS:-<none>}"
  local line
  for line in "$@"; do
    echo "$line"
  done
  echo "===================="
}

# Prepend $2 (a provenance file) to the CSV $1, then delete it.
bench_prepend_provenance() {
  local csvfile="$1" metafile="$2" tmp
  [[ -f "$metafile" ]] || return 0
  tmp=$(mktemp)
  cat "$metafile" "$csvfile" >"$tmp" && mv "$tmp" "$csvfile"
  rm -f "$metafile"
}

# ---------- timing ----------

# bench_median_stddev <samples_array> <median_var> <stddev_var>
# Median + sample stddev (n-1); both "N/A" when undefined.
bench_median_stddev() {
  local -n __bench_samples=$1
  local -n __bench_median=$2
  local -n __bench_stddev=$3
  local n=${#__bench_samples[@]}
  __bench_median="N/A"
  __bench_stddev="N/A"
  (( n == 0 )) && return 0
  local sorted mid
  mapfile -t sorted < <(printf '%s\n' "${__bench_samples[@]}" | sort -g)
  mid=$(( n / 2 ))
  if (( n % 2 == 1 )); then
    __bench_median="${sorted[$mid]}"
  else
    __bench_median=$(awk -v a="${sorted[$((mid-1))]}" -v b="${sorted[$mid]}" \
      'BEGIN{printf "%.6f", (a+b)/2.0}')
  fi
  if (( n >= 2 )); then
    __bench_stddev=$(printf '%s\n' "${__bench_samples[@]}" | awk '
      {x[NR]=$1; s+=$1}
      END{m=s/NR; ss=0; for(i=1;i<=NR;i++){d=x[i]-m; ss+=d*d}
          printf "%.6f", sqrt(ss/(NR-1))}')
  fi
}

# bench_repeat_cmd <cmd> <times_array_var> [reps]
# Runs <cmd> `reps` (default $NUM_RUNS) times, tees each run to the log, and
# appends the parsed "Training block took: X s" seconds to <times_array_var>.
# Sets BENCH_OOM / BENCH_ERR. An OOM (137) is deterministic for a given
# (dataset, config, memory), so it stops the remaining reps instead of burning
# them on the same kill.
bench_repeat_cmd() {
  local cmd="$1"
  local -n __bench_times=$2
  local reps="${3:-$NUM_RUNS}"
  __bench_times=()
  BENCH_OOM=0
  BENCH_ERR=0
  local i out t rc
  for ((i=1; i<=reps; i++)); do
    bench_log "----- Run $i/$reps -----"
    rc=0
    out=$(bash -c "$cmd" 2>&1) || rc=$?
    bench_log "$out"
    if (( rc != 0 )); then
      bench_log "WARNING: command exited with status $rc on run $i (continuing)"
      if (( rc == 137 )); then
        BENCH_OOM=1
        bench_log "WARNING: OOM-killed (137) on run $i; skipping remaining runs of this dataset"
        break
      else
        BENCH_ERR=1
      fi
    fi
    # A missing timing line means the binary died before training; `|| true`
    # keeps `set -euo pipefail` from aborting so the remaining datasets run.
    t=$(echo "$out" | grep -oE 'Training block took:[[:space:]]*[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?' \
        | grep -oE '[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?' | tail -1) || true
    if [[ -n "$t" ]]; then
      __bench_times+=("$t")
    else
      bench_log "WARNING: Could not parse 'Training block took' from run $i"
      if (( rc == 0 )); then BENCH_ERR=1; fi  # clean exit but no timing => real error
    fi
  done
}

is_dynamic_method() {
  [[ "$1" == Dynamic* ]]
}
