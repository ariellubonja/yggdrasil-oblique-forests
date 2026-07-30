#!/usr/bin/env bash
set -euo pipefail

# Multi-mode training-runtime line-search.
#
# One script, three sweeps selected by --mode:
#
#   --mode=dw1        (default) Sweep DW1_MIN_DEPTH across tree depth in
#                     increments of --step (default 4). Levels shallower than the
#                     threshold run plain per-node BFS; levels at/below it run the
#                     fused depthwise_1pass Apply. d<=2 is depthwise everywhere
#                     below the root (depth 1 is never depthwise);
#                     d >= the tree's deepest level is BFS everywhere. Finds the
#                     crossover depth where depthwise_1pass starts paying off.
#                     Built once with --config=depthwise_1_pass.
#
#   --mode=symmetric  Sweep SYMMETRIC_MAX_DEPTH across tree depth in
#                     increments of --step (default 5): symmetric to depth 5, 10,
#                     15, ... then DFS (GrowTreeLocal) for deeper levels. The tree
#                     root is depth 1 (never symmetric), so SYMMETRIC_MAX_DEPTH=5
#                     keeps depths 2..5 symmetric and switches to DFS from 6. The
#                     all-symmetric endpoint (threshold >= maxdepth) is skipped --
#                     it is already measured by a plain --config=symmetric_*
#                     runtime.sh run. (SYMMETRIC_MAX_DEPTH=0 would be the
#                     all-DFS endpoint; not swept by default.) Built once with
#                     --config=symmetric_optimized.
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
    SWEEP_ENV="DW1_MIN_DEPTH"
    SWEEP_COL="dw1_min_depth"
    DEPTH_STEP="${STEP_OVERRIDE:-4}"
    OUT_PREFIX="dw1_depth_linesearch"
    ;;
  symmetric)
    MODE_CONFIG="--config=symmetric_optimized"
    SWEEP_ENV="SYMMETRIC_MAX_DEPTH"
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

# Shared plumbing: ensure_icx, bazel_build (+EXTRA_BAZEL_CONFIGS), e-core
# toggling, EXTRA_TRAIN_ARGS, provenance, confirm_overwrite, timing helpers.
# EXTRA_TRAIN_ARGS is injected into every binary command; do NOT put
# --dataset_layout there in layout mode (the sweep sets it).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/bench_common.sh"
bench_restore_ecores_on_exit

logdir="benchmarks/results"
mkdir -p "$logdir"
logfile="${logdir}/${OUT_PREFIX}_${NUM_RUNS}runs_${SUFFIX}.log"
csvfile="${logdir}/${OUT_PREFIX}_${NUM_RUNS}runs_${SUFFIX}.csv"
BENCH_LOGFILE="$logfile"
# Set to 1 by run_point when any point OOM'd or errored; surfaces a warning at the
# end and keeps the log for inspection (QoL from runtime.sh).
DEGRADED=0

confirm_overwrite "$logfile" "$csvfile"

# Provenance + CSV header, both prepended before the first appended row.
write_provenance() {
  local build_desc="$1"
  bench_provenance_to_csv "$csvfile" \
    "dataset,${SWEEP_COL},median_s,stddev_s,n_ok,samples" \
    "mode: $MODE  sweep_var: ${SWEEP_ENV:-$SWEEP_COL}" \
    "build_config: $build_desc" \
    "NUM_TREES: $NUM_TREES  NUM_RUNS: $NUM_RUNS  DEPTH_STEP: ${DEPTH_STEP:-n/a}"
}

# Run one sweep point: NUM_RUNS reps, median + sample stddev, append a CSV row.
#   run_point <dataset_label> <point_value> <env_prefix> <cmd>
# env_prefix is e.g. "SYMMETRIC_MAX_DEPTH=5" (depth modes) or "" (layout).
run_point() {
  local dataset_label="$1" point="$2" env_prefix="$3" cmd="$4"
  bench_log "===== ${SWEEP_COL}=$point ${dataset_label} ====="
  bench_log "${env_prefix:+$env_prefix }$cmd"
  local times=()
  bench_repeat_cmd "${env_prefix:+export $env_prefix; }$cmd" times
  local median stddev n_ok="${#times[@]}"
  if (( n_ok > 0 )); then
    bench_median_stddev times median stddev
    bench_log "MEDIAN of ${n_ok}/${NUM_RUNS} runs: ${median} s  STDDEV: ${stddev} s  (samples: ${times[*]})"
  else
    # No run produced a timing. Label OOM (process killed) or ERROR (anything
    # else) so the point surfaces as a cell instead of crashing the sweep.
    median="ERROR"; (( BENCH_OOM == 1 )) && median="OOM"
    stddev="N/A"
    bench_log "MEDIAN of 0/${NUM_RUNS} runs: ${median}"
    DEGRADED=1
  fi
  echo "${dataset_label},${point},${median},${stddev},${n_ok},\"${times[*]:-}\"" >> "$csvfile"
}

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
  bazel_build "${BAZEL_FLAGS[@]}" "$MODE_CONFIG" "$BUILD_TARGET"
  BINARY="./bazel-bin/examples/train_oblique_forest"

  # E-cores now disabled, so nproc reflects the P-core count.
  # 5x cores to prevent skewness, or a fixed count under Boosting.
  NUM_TREES="$(bench_num_trees)"
  BASE_ARGS="--num_trees=$NUM_TREES"
  write_provenance "$MODE_CONFIG"

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
bench_ecores --disable
NUM_TREES="$(bench_num_trees)"
BASE_ARGS="--num_trees=$NUM_TREES"
write_provenance "layout sweep [${LAYOUTS[*]}] configs: ${ALL_CFGS:-<plain>}"

BINARY="./bazel-bin/examples/train_oblique_forest"
for key in "${ORDERED_KEYS[@]}"; do
  cfg="${KEY_CFG[$key]}"
  # shellcheck disable=SC2086  # $cfg intentionally word-split (empty for column)
  bazel_build "${BAZEL_FLAGS[@]}" $cfg "$BUILD_TARGET"

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
