#!/usr/bin/env bash
set -euo pipefail

# Find the (N, D) breakeven where FORCE_PRESORTED becomes faster than IN_NODE
# for AA + Exact splits.
#
# Why only AA + Exact: PRESORT only takes effect for axis-aligned splits
# (train_oblique_forest.cc:436) and only helps when the per-node sort is the
# bottleneck (i.e. Exact, not Random/Histogram which sort once into bins).
#
# Theory: in_node = K * mtry * N * log^2 N. presort = D * N * log N once
# (global, not per-tree -- see random_forest.cc:522), plus mtry * N per node.
# Breakeven N grows with D (more features to presort up front) and shrinks
# with deeper trees (more nodes amortize the presort).
#
# Grid: powers of two -- N in {2^10..2^24}, D in {2^2..2^12}, stepping the
# exponent by 2. Per cell we skip (N,D) if N*D*4B > MEM_CAP_BYTES (default
# 50% of MemTotal), since the float dataset alone would exceed the cap; this
# also keeps the PRESORT index buffer (same size, N*D*4B) within the other
# half of RAM. Tight -- if you OOM, drop MEM_FRAC.
#
# Usage:  $0 [--smoke|--full] <suffix>
#   --smoke           : 3 N x 1 D x 2 strategies x 1 run -- confirms direction.
#   --full (default)  : 2D power-of-2 grid x NUM_RUNS reps, median reported.

MODE="full"
SUFFIX=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) MODE="smoke"; shift ;;
    --full)  MODE="full";  shift ;;
    -h|--help)
      echo "Usage: $0 [--smoke|--full] <suffix>" >&2
      exit 0 ;;
    --*)
      echo "ERROR: unknown flag '$1'" >&2; exit 2 ;;
    *)
      if [[ -n "$SUFFIX" ]]; then
        echo "ERROR: unexpected positional '$1'" >&2; exit 2
      fi
      SUFFIX="${1,,}"; shift ;;
  esac
done
[[ -z "$SUFFIX" ]] && { echo "Usage: $0 [--smoke|--full] <suffix>" >&2; exit 2; }

###### Parameters

STRATEGIES=("in_node" "force_presorted")
NUM_THREADS=-1
COMPUTE_OOB_PERFORMANCES=false

# Memory cap: dataset (float N*D*4B) must fit in MEM_FRAC of MemTotal so the
# equal-sized PRESORT index buffer can fit in the rest. User instruction:
# 50% for the dataset.
MEM_FRAC="0.50"
MEM_TOTAL_KB=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
MEM_CAP_BYTES=$(awk -v kb="$MEM_TOTAL_KB" -v f="$MEM_FRAC" 'BEGIN{printf "%d", kb*1024*f}')
echo "MemTotal: $((MEM_TOTAL_KB/1024)) MiB; dataset cap (${MEM_FRAC}): $((MEM_CAP_BYTES/1024/1024)) MiB"

# Build the grid. Powers of two, exponent stepped by 2.
build_grid() {
  GRID=()
  if [[ "$MODE" == "smoke" ]]; then
    NUM_RUNS=1
    GRID+=("10000|64" "1000000|64" "10000000|64")
    return
  fi
  NUM_RUNS=3   # full grid is large; 3-run median to keep total budget sane.
  local nexp dexp N D bytes
  for nexp in 10 12 14 16 18 20 22 24; do
    N=$((1 << nexp))
    for dexp in 4 6 8 10 12 14; do
      D=$((1 << dexp))
      bytes=$(( N * D * 4 ))
      if (( bytes > MEM_CAP_BYTES )); then
        echo "  skip N=2^${nexp}=$N D=2^${dexp}=$D ($((bytes/1024/1024)) MiB > cap)"
        continue
      fi
      GRID+=("${N}|${D}")
    done
  done
}
build_grid
echo "Grid cells: ${#GRID[@]}  (x ${#STRATEGIES[@]} strategies x $NUM_RUNS runs = $(( ${#GRID[@]} * ${#STRATEGIES[@]} * NUM_RUNS )) total runs)"

# Build target and base flags (kept identical to runtime.sh)
BUILD_TARGET="//examples:train_oblique_forest"
BAZEL_FLAGS=(-c opt --cxxopt="-O3" --cxxopt="-march=native" --repo_env=CC=icx --repo_env=CXX=icpx)

# Shared plumbing: ensure_icx, bazel_build (+EXTRA_BAZEL_CONFIGS), e-core gating
# (mirrors runtime.sh; memory: feedback_cpu_isolation), provenance,
# confirm_overwrite, timing helpers.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/bench_common.sh"
bench_restore_ecores_on_exit

logdir="benchmarks/results"
mkdir -p "$logdir"
logfile="${logdir}/presort_breakeven_${MODE}_${SUFFIX}.log"
csvfile="${logdir}/presort_breakeven_${MODE}_${SUFFIX}.csv"
BENCH_LOGFILE="$logfile"

confirm_overwrite "$logfile" "$csvfile"

bazel_build "${BAZEL_FLAGS[@]}" "$BUILD_TARGET"
BINARY="./bazel-bin/examples/train_oblique_forest"

# After --disable, nproc reflects P-cores only.
NUM_TREES=$(( $(bench_nproc) * 5 ))
BASE_ARGS="--num_trees=$NUM_TREES --num_threads=$NUM_THREADS --compute_oob_performances=$COMPUTE_OOB_PERFORMANCES"

# Provenance + header, both before the first appended row.
bench_provenance_to_csv "$csvfile" \
  "rows,cols,strategy,num_trees,median_s,stddev_s,n_samples,all_samples_s" \
  "NUM_TREES: $NUM_TREES  NUM_RUNS: $NUM_RUNS  NUM_THREADS: $NUM_THREADS  MODE: $MODE" \
  "STRATEGIES: ${STRATEGIES[*]}" \
  "MEM_FRAC: $MEM_FRAC  grid_cells: ${#GRID[@]}"

run_one() {
  local rows=$1 cols=$2 strategy=$3
  local cmd="$BINARY --input_mode trunk --rows $rows --cols $cols \
    --feature_split_type \"Axis Aligned\" --numerical_split_type \"Exact\" \
    --aa_sorting_strategy=$strategy $BASE_ARGS"
  bench_log "=== rows=$rows cols=$cols strategy=$strategy ==="
  bench_log "$cmd"

  local times=()
  bench_repeat_cmd "$cmd" times

  local median stddev n=${#times[@]} samples=""
  bench_median_stddev times median stddev
  if (( n > 0 )); then
    samples=$(IFS=';'; echo "${times[*]}")
  fi
  bench_log "RESULT rows=$rows cols=$cols strategy=$strategy median=${median}s stddev=${stddev}s n=$n"
  echo "${rows},${cols},${strategy},${NUM_TREES},${median},${stddev},${n},${samples}" >> "$csvfile"
}

for entry in "${GRID[@]}"; do
  IFS='|' read -r rows cols <<<"$entry"
  for strategy in "${STRATEGIES[@]}"; do
    run_one "$rows" "$cols" "$strategy"
  done
done

echo
echo "Done. CSV: $csvfile"
echo "Log: $logfile  (kept; not auto-parsed)"
