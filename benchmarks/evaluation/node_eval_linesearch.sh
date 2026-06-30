#!/usr/bin/env bash
set -euo pipefail

# Multi-mode training-runtime line-search.
#
# One script, three sweeps selected by --mode:
#
#   --mode=dw1        (default) Sweep YDF_DW1_MIN_DEPTH across tree depth in
#                     increments of --step (default 4). Levels shallower than the
#                     threshold run plain per-node BFS; levels at/below it run the
#                     fused depthwise_1pass Apply. d=0 is depthwise everywhere;
#                     d >= the tree's deepest level is BFS everywhere. Finds the
#                     crossover depth where depthwise_1pass starts paying off.
#                     Built once with --config=depthwise_1_pass.
#
#   --mode=symmetric  Sweep YDF_SYMMETRIC_MAX_DEPTH across tree depth in
#                     increments of --step (default 5): symmetric to depth 5, 10,
#                     15, ... then DFS (GrowTreeLocal) for deeper levels. The tree
#                     root is depth 1, so YDF_SYMMETRIC_MAX_DEPTH=5 keeps depths
#                     1..5 symmetric and switches to DFS from depth 6. The
#                     all-symmetric endpoint (threshold >= maxdepth) is skipped --
#                     it is already measured by a plain --config=symmetric_*
#                     runtime.sh run. (YDF_SYMMETRIC_MAX_DEPTH=0 would be the
#                     all-DFS endpoint; not swept by default.) Built once with
#                     --config=symmetric_depthwise_ap.
#
#   --mode=layout     Sweep --dataset_layout over --layouts (default "column row")
#                     to compare row- vs column-major numerical storage. NOT a
#                     depth sweep: each layout is one point. Non-column layouts
#                     need a matching Bazel config, so the binary is rebuilt per
#                     config group (column=plain, row/dynamic_row_col_major=
#                     row_major_dataset_layout).
#
# Each sweep point is a fresh binary launch: the *_MIN/MAX_DEPTH knobs are read
# once per process, and --dataset_layout is a per-launch flag.
#
# Unlike runtime.sh this writes its own CSV (dataset, <sweep_col>, median,
# stddev, ...) because the shared parser has no sweep column. Methodology
# (median over --runs, e-core disable, icx pinning, provenance, OOM/ERROR
# tolerance, dataset/config overrides) is imported from runtime.sh.
#
# Usage:  $0 [--runs=N] [--mode=dw1|symmetric|layout] [--step=N]
#            [--layouts="column row ..."] <suffix>     (default --runs=1)
#   e.g. '$0 --mode=symmetric m7i'
#         -> benchmarks/results/symmetric_depth_linesearch_1runs_m7i.csv

NUM_RUNS=1
MODE="dw1"
STEP_OVERRIDE=""
LAYOUTS_OVERRIDE=""
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
    --mode=*)
      MODE="${1#*=}"
      case "$MODE" in
        dw1|symmetric|layout) ;;
        *) echo "ERROR: --mode must be dw1, symmetric, or layout, got '$MODE'" >&2
           exit 2 ;;
      esac
      shift ;;
    --step=*)
      STEP_OVERRIDE="${1#*=}"
      if ! [[ "$STEP_OVERRIDE" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: --step must be a positive integer, got '$STEP_OVERRIDE'" >&2
        exit 2
      fi
      shift ;;
    --layouts=*)
      LAYOUTS_OVERRIDE="${1#*=}"; shift ;;
    -h|--help)
      echo "Usage: $0 [--runs=N] [--mode=dw1|symmetric|layout] [--step=N] [--layouts=\"column row ...\"] <suffix>" >&2
      exit 0 ;;
    --*)
      echo "ERROR: unknown flag '$1'" >&2
      echo "Usage: $0 [--runs=N] [--mode=dw1|symmetric|layout] [--step=N] [--layouts=\"...\"] <suffix>" >&2
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
  echo "Usage: $0 [--runs=N] [--mode=dw1|symmetric|layout] [--step=N] [--layouts=\"...\"] <suffix>" >&2
  echo "  e.g. '$0 --mode=symmetric m7i' -> benchmarks/results/symmetric_depth_linesearch_1runs_m7i.csv" >&2
  exit 2
fi

###### Per-mode parameters

# Depth-sweep modes share the line-search machinery; only the build config, the
# env knob, the CSV column name and the default step differ. Layout mode sweeps a
# discrete flag and rebuilds per config group, so it has its own driver below.
case "$MODE" in
  dw1)
    MODE_CONFIG="--config=depthwise_1_pass"
    SWEEP_ENV="YDF_DW1_MIN_DEPTH"
    SWEEP_COL="dw1_min_depth"
    DEPTH_STEP="${STEP_OVERRIDE:-4}"
    OUT_PREFIX="dw1_depth_linesearch"
    ;;
  symmetric)
    MODE_CONFIG="--config=symmetric_depthwise_ap"
    SWEEP_ENV="YDF_SYMMETRIC_MAX_DEPTH"
    SWEEP_COL="symmetric_max_depth"
    DEPTH_STEP="${STEP_OVERRIDE:-5}"
    OUT_PREFIX="symmetric_depth_linesearch"
    ;;
  layout)
    MODE_CONFIG=""          # per-layout config groups, set in the layout driver
    SWEEP_ENV=""            # layout is a train flag, not an env knob
    SWEEP_COL="dataset_layout"
    OUT_PREFIX="layout_linesearch"
    LAYOUTS=(column row)
    if [[ -n "$LAYOUTS_OVERRIDE" ]]; then
      # shellcheck disable=SC2206
      LAYOUTS=($LAYOUTS_OVERRIDE)
    fi
    ;;
esac

NUM_THREADS=-1                     # train_oblique_forest.cc default is 1, so passed

# Datasets: the uncommented entries in runtime.sh, each tagged with its
# approximate deepest tree level (the depth-sweep upper bound; ignored by
# --mode=layout).
#   csv|<path>|<label_col>|<max_depth>
#   trunk|<rows>|<cols>|<max_depth>
# max_depth values: HIGGS=60 and 1.5M x 4096=25 are measured; the two smaller
# trunk datasets are GUESSES (fewer rows -> shallower trees) -- adjust once you
# have real numbers.
DATASETS=(
  "csv|benchmarks/data/HIGGS_with_header.csv|class|60" # Necessary, though timing-expensive. Trunk 11m x 28 is only depth 38
  "trunk|1500000|4096|40"
  "trunk|150000|40000|40"
  "trunk|15000|400000|40"
)
# Optional override (QoL from runtime.sh): isolate ONE dataset per run so an OOM
# costs only that dataset. Same 'kind|a|b|maxdepth' format, ';'-separated.
# Empty string selects ZERO datasets; unset => the defaults above.
if [[ "${DATASETS_OVERRIDE+x}" == "x" ]]; then
  DATASETS=()
  [[ -n "$DATASETS_OVERRIDE" ]] && IFS=';' read -r -a DATASETS <<<"$DATASETS_OVERRIDE"
fi

# =========================
BUILD_TARGET="//examples:train_oblique_forest"
BAZEL_FLAGS=(-c opt --cxxopt="-O3" --cxxopt="-march=native" --repo_env=CC=icx --repo_env=CXX=icpx)
VEC_CONFIG="--config=enable_std_upper_bound_avx2"   # vectorized_avx2 split-finding
# Optional extra build configs/flags injected into every bazel_build (QoL from
# runtime.sh), e.g. EXTRA_BAZEL_CONFIGS="--config=cache_projection_evaluator".
# shellcheck disable=SC2206
EXTRA_BAZEL_CONFIGS_ARR=(${EXTRA_BAZEL_CONFIGS:-})
# Optional extra train_oblique_forest flags injected into every binary command,
# e.g. EXTRA_TRAIN_ARGS="--num_trees=120". Do NOT put --dataset_layout here in
# layout mode (the sweep sets it).
EXTRA_TRAIN_ARGS=${EXTRA_TRAIN_ARGS:-}

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
  bazel build "$@" "${EXTRA_BAZEL_CONFIGS_ARR[@]}"
  sudo "$SET_CPU_E_FEATURES" --disable
}

logdir="benchmarks/results"
mkdir -p "$logdir"
logfile="${logdir}/${OUT_PREFIX}_${NUM_RUNS}runs_${SUFFIX}.log"
csvfile="${logdir}/${OUT_PREFIX}_${NUM_RUNS}runs_${SUFFIX}.csv"
# Set to 1 by run_point when any point OOM'd or errored; surfaces a warning at the
# end and keeps the log for inspection (QoL from runtime.sh).
DEGRADED=0

# If an output file already exists, ask before clobbering it (CSV only; the log is
# silently replaced). Reads from /dev/tty so the prompt works under a pipe; a
# non-interactive run with no tty keeps the fail-safe.
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
for f in "$logfile" "$csvfile"; do
  confirm_overwrite "$f"
done

# Provenance: prepended to the top of the CSV so the result is self-describing.
# The leading lines are intentionally not well-formed CSV. Also tee'd to the log.
# Hardware serial for traceability. dmidecode needs root and is Linux-only; if it
# fails, keep the error text in the field rather than aborting.
write_provenance() {
  local build_desc="$1"
  machine_serial=$(sudo dmidecode -s system-serial-number 2>&1) \
    || machine_serial="dmidecode failed: $machine_serial"
  {
    echo "==== PROVENANCE ===="
    echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_sha: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo '-dirty')"
    echo "git_branch: $(git branch --show-current 2>/dev/null || echo unknown)"
    echo "machine: $(lscpu | grep 'Model name' | sed 's/Model name:[[:space:]]*//') (nproc=$(nproc))"
    echo "machine_serial: $machine_serial"
    echo "mode: $MODE  sweep_var: ${SWEEP_ENV:-$SWEEP_COL}"
    echo "build_config: $build_desc"
    echo "EXTRA_BAZEL_CONFIGS: ${EXTRA_BAZEL_CONFIGS:-<none>}"
    echo "EXTRA_TRAIN_ARGS: ${EXTRA_TRAIN_ARGS:-<none>}"
    echo "NUM_TREES: $NUM_TREES  NUM_RUNS: $NUM_RUNS  NUM_THREADS: $NUM_THREADS  DEPTH_STEP: ${DEPTH_STEP:-n/a}"
    echo "===================="
  } | tee -a "$logfile" > "$csvfile"
  echo "dataset,${SWEEP_COL},median_s,stddev_s,n_ok,samples" >> "$csvfile"
}

# Run one sweep point: NUM_RUNS reps, median + sample stddev, append a CSV row.
# Mirrors runtime.sh's run_cmd timing/median math + OOM/ERROR tolerance.
#   run_point <dataset_label> <point_value> <env_prefix> <cmd>
# env_prefix is e.g. "YDF_SYMMETRIC_MAX_DEPTH=5" (depth modes) or "" (layout).
run_point() {
  local dataset_label="$1" point="$2" env_prefix="$3" cmd="$4"
  echo "${env_prefix:+$env_prefix }$cmd" | tee -a "$logfile"
  local times=() i out t rc oom=0 err=0
  for ((i=1; i<=NUM_RUNS; i++)); do
    echo "----- Run $i/$NUM_RUNS (${SWEEP_COL}=$point) -----" | tee -a "$logfile"
    rc=0
    out=$(bash -c "${env_prefix:+export $env_prefix; }$cmd" 2>&1) || rc=$?
    echo "$out" | tee -a "$logfile"
    if (( rc != 0 )); then
      echo "WARNING: command exited with status $rc on run $i (continuing)" | tee -a "$logfile"
      # 137 = 128 + SIGKILL(9): OOM-killed. Deterministic for a given
      # (dataset, config, memory), so stop after the first instead of wasting
      # NUM_RUNS-1 more OOMs; record OOM below and move on.
      if (( rc == 137 )); then
        oom=1
        echo "WARNING: OOM-killed (137) on run $i; skipping remaining runs of this point" | tee -a "$logfile"
        break
      else
        err=1
      fi
    fi
    t=$(echo "$out" | grep -oE 'Training block took:[[:space:]]*[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?' \
        | grep -oE '[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?' | tail -1) || true
    if [[ -n "$t" ]]; then
      times+=("$t")
    else
      echo "WARNING: Could not parse 'Training block took' from run $i" | tee -a "$logfile"
      if (( rc == 0 )); then err=1; fi  # clean exit but no timing => a real error
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
    # No run produced a timing. Label OOM (process killed) or ERROR (anything
    # else) so the point surfaces as a cell instead of crashing the sweep.
    median="ERROR"; (( oom == 1 )) && median="OOM"
    stddev="N/A"
    echo "MEDIAN of 0/${NUM_RUNS} runs: ${median}" | tee -a "$logfile"
    DEGRADED=1
  fi
  echo "${dataset_label},${point},${median},${stddev},${n_ok},\"${times[*]:-}\"" >> "$csvfile"
}

banner() { echo -e "\n\n======== $* ========\n" | tee -a "$logfile"; }

# Build the base argument list for a dataset entry. Sets `dataset_label` and
# `cmd_base` (binary + input flags + BASE_ARGS, no sweep-specific flag).
build_cmd_base() {
  local kind="$1" a="$2" b="$3"
  if [[ "$kind" == "csv" ]]; then
    dataset_label="$(basename "${a%.csv}")"
    cmd_base="$BINARY --input_mode csv --train_csv \"$a\" --label_col \"$b\" $BASE_ARGS $EXTRA_TRAIN_ARGS"
  else
    dataset_label="trunk_${a}_x_${b}"
    cmd_base="$BINARY --input_mode trunk --rows $a --cols $b $BASE_ARGS $EXTRA_TRAIN_ARGS"
  fi
}

# =========================
# DEPTH-SWEEP DRIVER (dw1, symmetric): build once, sweep the env knob.
# =========================
if [[ "$MODE" == "dw1" || "$MODE" == "symmetric" ]]; then
  bazel_build "${BAZEL_FLAGS[@]}" "$MODE_CONFIG" "$VEC_CONFIG" "$BUILD_TARGET"
  BINARY="./bazel-bin/examples/train_oblique_forest"

  # E-cores now disabled; compute NUM_TREES from runtime nproc (P-cores only).
  NUM_TREES=$(( $(nproc) * 5 ))   # 5x cores to reduce scheduling skew
  BASE_ARGS="--num_trees=$NUM_TREES --num_threads=$NUM_THREADS"
  write_provenance "$MODE_CONFIG $VEC_CONFIG"

  for entry in "${DATASETS[@]}"; do
    IFS='|' read -r kind a b maxdepth <<<"$entry"
    build_cmd_base "$kind" "$a" "$b"

    # Depth grid.
    depths=()
    if [[ "$MODE" == "dw1" ]]; then
      # 0..maxdepth by step, with maxdepth appended (pure-BFS endpoint).
      for ((d=0; d<=maxdepth; d+=DEPTH_STEP)); do depths+=("$d"); done
      if (( ${#depths[@]} == 0 || depths[${#depths[@]}-1] != maxdepth )); then
        depths+=("$maxdepth")
      fi
    else
      # symmetric: step, 2*step, ... strictly below maxdepth (skip all-symmetric).
      for ((d=DEPTH_STEP; d<maxdepth; d+=DEPTH_STEP)); do depths+=("$d"); done
    fi

    if (( ${#depths[@]} == 0 )); then
      banner "DATASET $dataset_label (max_depth=$maxdepth) — no sweep points (maxdepth<=step); skipping"
      continue
    fi
    banner "DATASET $dataset_label (max_depth=$maxdepth) — ${SWEEP_COL}: ${depths[*]}"
    for d in "${depths[@]}"; do
      run_point "$dataset_label" "$d" "$SWEEP_ENV=$d" "$cmd_base"
    done
  done

  echo "CSV: $csvfile  (provenance prepended at top)"
  (( DEGRADED == 1 )) && echo "WARNING: one or more points OOM'd/ERROR'd — see log $logfile" >&2
  exit 0
fi

# =========================
# LAYOUT-SWEEP DRIVER: rebuild per config group, sweep --dataset_layout.
# =========================

# Map a layout to (build_key, bazel config flags). Layouts sharing a key share a
# binary, so each config is built exactly once.
layout_build_spec() {
  case "$1" in
    column)                 build_key="col_base";  layout_cfg="" ;;
    row|dynamic_row_col_major)
                            build_key="rowmajor";  layout_cfg="--config=row_major_dataset_layout" ;;
    *) echo "ERROR: unknown layout '$1' (use column, row, dynamic_row_col_major)" >&2
       exit 2 ;;
  esac
}

# Group layouts by build key (preserving first-seen order) so we rebuild minimally.
declare -A KEY_CFG KEY_LAYOUTS
ORDERED_KEYS=()
ALL_CFGS=""
for layout in "${LAYOUTS[@]}"; do
  layout_build_spec "$layout"
  if [[ -z "${KEY_LAYOUTS[$build_key]+x}" ]]; then
    ORDERED_KEYS+=("$build_key")
    KEY_CFG[$build_key]="$layout_cfg"
    KEY_LAYOUTS[$build_key]="$layout"
    ALL_CFGS+="${layout_cfg:+$layout_cfg }"
  else
    KEY_LAYOUTS[$build_key]+=" $layout"
  fi
done

# Provenance needs NUM_TREES, which needs e-cores disabled. The first bazel_build
# in the loop disables them; do an explicit disable here so NUM_TREES (computed
# before any build for the provenance header) reflects P-core topology.
sudo "$SET_CPU_E_FEATURES" --disable
NUM_TREES=$(( $(nproc) * 5 ))
BASE_ARGS="--num_trees=$NUM_TREES --num_threads=$NUM_THREADS"
write_provenance "layout sweep [${LAYOUTS[*]}] configs: ${ALL_CFGS:-<plain>} $VEC_CONFIG"

BINARY="./bazel-bin/examples/train_oblique_forest"
for key in "${ORDERED_KEYS[@]}"; do
  cfg="${KEY_CFG[$key]}"
  # shellcheck disable=SC2086
  bazel_build "${BAZEL_FLAGS[@]}" $cfg "$VEC_CONFIG" "$BUILD_TARGET"

  for layout in ${KEY_LAYOUTS[$key]}; do
    banner "LAYOUT $layout (config: ${cfg:-<plain>})"
    for entry in "${DATASETS[@]}"; do
      IFS='|' read -r kind a b maxdepth <<<"$entry"
      build_cmd_base "$kind" "$a" "$b"
      run_point "$dataset_label" "$layout" "" "$cmd_base --dataset_layout=$layout"
    done
  done
done

echo "CSV: $csvfile  (provenance prepended at top)"
(( DEGRADED == 1 )) && echo "WARNING: one or more points OOM'd/ERROR'd — see log $logfile" >&2
# CPU features re-enabled by the EXIT trap.
