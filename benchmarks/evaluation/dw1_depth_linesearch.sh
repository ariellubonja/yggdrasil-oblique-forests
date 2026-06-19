#!/usr/bin/env bash
set -euo pipefail

# Depthwise-1-pass depth line-search.
#
# Sweeps the YDF_DW1_MIN_DEPTH runtime knob (see GrowTreeLocalBFS in
# learner/decision_tree/training.cc) across tree depth in increments of 4:
# levels shallower than the threshold run plain per-node BFS, levels at or
# below it run the fused depthwise_1pass Apply. d=0 is depthwise everywhere;
# d >= the tree's deepest level is BFS everywhere. The goal is to find the
# crossover depth where switching to depthwise_1pass starts paying off.
#
# The binary is built ONCE with --config=depthwise_1_pass (the macro is a
# build-time #ifdef); YDF_DW1_MIN_DEPTH is read once per process, so every
# depth point is a fresh binary launch.
#
# Unlike runtime.sh this writes its own CSV (dataset, dw1_min_depth, median,
# stddev, ...) because the shared parser has no depth column. Methodology
# (median over --runs, e-core disable, icx pinning, provenance) is reused.
#
# Usage:  $0 [--runs=N] <suffix>     (default --runs=3)
#   e.g. '$0 m7i' -> benchmarks/results/depth_linesearch_3runs_m7i.csv

NUM_RUNS=1
SUFFIX=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs=*)
      NUM_RUNS="${1#*=}"
      if ! [[ "$NUM_RUNS" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: --runs must be a positive integer, got '$NUM_RUNS'" >&2
        exit 2
      fi
      shift ;;
    -h|--help)
      echo "Usage: $0 [--runs=N] <suffix>" >&2
      exit 0 ;;
    --*)
      echo "ERROR: unknown flag '$1'" >&2
      echo "Usage: $0 [--runs=N] <suffix>" >&2
      exit 2 ;;
    *)
      if [[ -n "$SUFFIX" ]]; then
        echo "ERROR: unexpected positional argument '$1' (suffix already set to '$SUFFIX')" >&2
        exit 2
      fi
      SUFFIX="${1,,}"; shift ;;
  esac
done
if [[ -z "$SUFFIX" ]]; then
  echo "Usage: $0 [--runs=N] <suffix>" >&2
  echo "  e.g. '$0 m7i' -> benchmarks/results/depth_linesearch_3runs_m7i.csv" >&2
  exit 2
fi

###### Fixed parameters (not configurable here)

DEPTH_STEP=4                       # line-search granularity
NUM_THREADS=-1                     # train_oblique_forest.cc default is 1, so passed
# Other arguments will use the defaults from train_oblique_forest.cc

# Datasets: the uncommented entries in runtime.sh, each tagged with its
# approximate deepest tree level (the line-search upper bound).
#   csv|<path>|<label_col>|<max_depth>
#   trunk|<rows>|<cols>|<max_depth>
# max_depth values: HIGGS=60 and 1.5M x 4096=25 are measured; the two smaller
# trunk datasets are GUESSES (fewer rows -> shallower trees) -- adjust once you
# have real numbers.
DATASETS=(
  "csv|benchmarks/data/HIGGS_with_header.csv|class|60"
  "trunk|1500000|4096|25"
  "trunk|150000|40000|20"   # GUESSed depth
  "trunk|15000|400000|15"   # GUESSed depth
)

# =========================
BUILD_TARGET="//examples:train_oblique_forest"
BAZEL_FLAGS=(-c opt --cxxopt="-O3" --cxxopt="-march=native" --repo_env=CC=icx --repo_env=CXX=icpx)
DW1_CONFIG="--config=depthwise_1_pass"
VEC_CONFIG="--config=enable_std_upper_bound_avx2"   # vectorized_avx2 split-finding

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SET_CPU_E_FEATURES="$(cd "$SCRIPT_DIR/../.." && pwd)/benchmarks/utils/set_cpu_e_features.sh"
trap 'sudo "$SET_CPU_E_FEATURES" --enable' EXIT

# .bazelrc pins CC=icx; a silent gcc fallback corrupts every timing number.
# Source oneAPI here and abort loudly if icx still can't be found.
ensure_icx() {
  if ! command -v icx >/dev/null 2>&1; then
    local setvars="${ONEAPI_SETVARS:-/opt/intel/oneapi/setvars.sh}"
    if [[ -r "$setvars" ]]; then
      set +u +e
      # shellcheck disable=SC1090
      source "$setvars" >/dev/null 2>&1
      set -u -e
    fi
  fi
  if ! command -v icx >/dev/null 2>&1 || ! command -v icpx >/dev/null 2>&1; then
    echo "ERROR: icx/icpx not on PATH and could not source oneAPI." >&2
    echo "       Run 'source /opt/intel/oneapi/setvars.sh' and retry." >&2
    exit 1
  fi
  echo "Compiler: $(command -v icx)"
}

bazel_build() {
  ensure_icx
  sudo "$SET_CPU_E_FEATURES" --enable
  bazel build "$@"
  sudo "$SET_CPU_E_FEATURES" --disable
}

logdir="benchmarks/results"
mkdir -p "$logdir"
logfile="${logdir}/depth_linesearch_${NUM_RUNS}runs_${SUFFIX}.log"
csvfile="${logdir}/depth_linesearch_${NUM_RUNS}runs_${SUFFIX}.csv"
# If an output file already exists, ask before clobbering it instead of
# aborting outright. Reads from the terminal (/dev/tty) so the prompt works
# even when stdin is piped; a non-interactive run (no tty) keeps the old
# fail-safe behavior rather than silently overwriting.
confirm_overwrite() {
  local f="$1"
  [[ -e "$f" ]] || return 0
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
for f in "$logfile" "$csvfile"; do
  confirm_overwrite "$f"
done

# Build the depthwise_1pass + AVX2-vectorized binary once.
bazel_build "${BAZEL_FLAGS[@]}" "$DW1_CONFIG" "$VEC_CONFIG" "$BUILD_TARGET"
BINARY="./bazel-bin/examples/train_oblique_forest"

# E-cores now disabled. Compute NUM_TREES from runtime nproc (P-cores only).
NUM_TREES=$(( $(nproc) * 5 ))   # 5x cores to reduce scheduling skew
# Only flags that DIFFER from train_oblique_forest.cc defaults are passed.
BASE_ARGS="--num_trees=$NUM_TREES --num_threads=$NUM_THREADS"

# Provenance: prepended to the top of the CSV so the result is self-describing.
# The leading lines are intentionally not well-formed CSV. Also tee'd to the log.
# Hardware serial for traceability. dmidecode needs root and is Linux-only; if
# it fails (no sudo, not installed, non-Linux), keep the error text in the field
# rather than aborting -- provenance is best-effort.
machine_serial=$(sudo dmidecode -s system-serial-number 2>&1) \
  || machine_serial="dmidecode failed: $machine_serial"
{
  echo "==== PROVENANCE ===="
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_sha: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo '-dirty')"
  echo "git_branch: $(git branch --show-current 2>/dev/null || echo unknown)"
  echo "machine: $(lscpu | grep 'Model name' | sed 's/Model name:[[:space:]]*//') (nproc=$(nproc))"
  echo "machine_serial: $machine_serial"
  echo "build_config: $DW1_CONFIG $VEC_CONFIG"
  echo "NUM_TREES: $NUM_TREES  NUM_RUNS: $NUM_RUNS  NUM_THREADS: $NUM_THREADS  DEPTH_STEP: $DEPTH_STEP"
  echo "===================="
} | tee -a "$logfile" > "$csvfile"

# CSV header (appended below the provenance block).
echo "dataset,dw1_min_depth,median_s,stddev_s,n_ok,samples" >> "$csvfile"

# Run one (dataset, depth) point: NUM_RUNS reps, median + sample stddev, append
# a CSV row. Mirrors runtime.sh's run_cmd timing/median math.
run_point() {
  local dataset_label="$1" depth="$2" cmd="$3"
  echo "YDF_DW1_MIN_DEPTH=$depth $cmd" | tee -a "$logfile"
  local times=() i out t rc
  for ((i=1; i<=NUM_RUNS; i++)); do
    echo "----- Run $i/$NUM_RUNS (depth=$depth) -----" | tee -a "$logfile"
    rc=0
    out=$(YDF_DW1_MIN_DEPTH="$depth" bash -c "$cmd" 2>&1) || rc=$?
    echo "$out" | tee -a "$logfile"
    if (( rc != 0 )); then
      echo "WARNING: command exited with status $rc on run $i (continuing)" | tee -a "$logfile"
    fi
    t=$(echo "$out" | grep -oE 'Training block took:[[:space:]]*[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?' \
        | grep -oE '[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?' | tail -1)
    if [[ -n "$t" ]]; then
      times+=("$t")
    else
      echo "WARNING: Could not parse 'Training block took' from run $i" | tee -a "$logfile"
    fi
  done

  local median stddev n_ok="${#times[@]}"
  if (( n_ok > 0 )); then
    local sorted; mapfile -t sorted < <(printf '%s\n' "${times[@]}" | sort -g)
    local n=${#sorted[@]} mid=$(( n_ok / 2 ))
    if (( n % 2 == 1 )); then
      median="${sorted[$mid]}"
    else
      median=$(awk -v a="${sorted[$((mid-1))]}" -v b="${sorted[$mid]}" 'BEGIN{printf "%.6f", (a+b)/2.0}')
    fi
    if (( n >= 2 )); then
      stddev=$(printf '%s\n' "${times[@]}" | awk '
        {x[NR]=$1; s+=$1}
        END{m=s/NR; ss=0; for(i=1;i<=NR;i++){d=x[i]-m; ss+=d*d}
            printf "%.6f", sqrt(ss/(NR-1))}')
    else
      stddev="N/A"
    fi
    echo "MEDIAN of ${n_ok}/${NUM_RUNS} runs: ${median} s  STDDEV: ${stddev} s  (samples: ${times[*]})" | tee -a "$logfile"
  else
    median="N/A"; stddev="N/A"
    echo "MEDIAN of 0/${NUM_RUNS} runs: N/A" | tee -a "$logfile"
  fi
  echo "${dataset_label},${depth},${median},${stddev},${n_ok},\"${times[*]:-}\"" >> "$csvfile"
}

banner() { echo -e "\n\n======== $* ========\n" | tee -a "$logfile"; }

# -------------------------
# Line-search: for each dataset, sweep depth 0, STEP, 2*STEP, ... up to and
# including max_depth (last point ~ pure BFS).
# -------------------------
for entry in "${DATASETS[@]}"; do
  IFS='|' read -r kind a b maxdepth <<<"$entry"

  if [[ "$kind" == "csv" ]]; then
    dataset_label="$(basename "${a%.csv}")"
    cmd_base="$BINARY --input_mode csv --train_csv \"$a\" --label_col \"$b\" $BASE_ARGS"
  else
    dataset_label="trunk_${a}_x_${b}"
    cmd_base="$BINARY --input_mode trunk --rows $a --cols $b $BASE_ARGS"
  fi

  # Depth grid: 0..maxdepth by DEPTH_STEP, with maxdepth appended if the step
  # didn't already land on it (guarantees a pure-BFS endpoint).
  depths=()
  for ((d=0; d<=maxdepth; d+=DEPTH_STEP)); do depths+=("$d"); done
  if (( depths[${#depths[@]}-1] != maxdepth )); then depths+=("$maxdepth"); fi

  banner "DATASET $dataset_label (max_depth=$maxdepth) — depths: ${depths[*]}"
  for d in "${depths[@]}"; do
    run_point "$dataset_label" "$d" "$cmd_base"
  done
done

echo "CSV: $csvfile  (provenance prepended at top)"
# Log kept (no separate parse step). CPU features re-enabled by EXIT trap.
