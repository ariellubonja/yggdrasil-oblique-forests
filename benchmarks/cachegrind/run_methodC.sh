#!/usr/bin/env bash
# Method C: callgrind per-depth cache-sim of the DW1 oblique gather (col[sel_ptr[i]]).
#
# Build first with build_methodC.sh. Outputs go to a PERSISTENT dir (default
# $HOME/dw1_cachegrind_C) -- NOT /tmp, which some cloud boxes wipe on reboot.
#
# Key flags:
#  --instr-atstart=no  : do NOT cache-simulate the ~8GB CSV load / non-kernel code.
#                        training.cc calls CALLGRIND_START/STOP_INSTRUMENTATION around
#                        the kernel, so only the gather is simulated -> load is ~30x
#                        faster (minutes, not hours).
#  --num_trees=1 / --num_threads=1 : Valgrind serializes threads, so one tree single-
#                        threaded is both faster and gives clean per-depth dumps.
#
# Override via env: METHODC_OUT, HIGGS_CSV, LABEL_COL, MAX_PROJ, TREE_DEPTH.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${METHODC_OUT:-$HOME/dw1_cachegrind_C}"
CGDIR="$OUT/cgout"
BIN="$REPO/bazel-bin/examples/train_oblique_forest"
HIGGS="${HIGGS_CSV:-$REPO/benchmarks/data/HIGGS_with_header.csv}"
LABEL="${LABEL_COL:-class}"
MAXPROJ="${MAX_PROJ:-200}"
DEPTH="${TREE_DEPTH:--1}"

command -v valgrind >/dev/null || { echo "valgrind not found"; exit 1; }
[ -x "$BIN" ] || { echo "binary missing: $BIN (run build_methodC.sh)"; exit 1; }
[ -f "$HIGGS" ] || { echo "CSV missing: $HIGGS (set HIGGS_CSV)"; exit 1; }

mkdir -p "$CGDIR"
cd "$REPO" || exit 1
echo "run started $(date -u +%FT%TZ) out=$OUT" | tee "$OUT/run.log"

valgrind --tool=callgrind --cache-sim=yes --instr-atstart=no \
  --callgrind-out-file="$CGDIR/callgrind.out.%p" \
  "$BIN" \
    --input_mode=csv --train_csv="$HIGGS" --label_col="$LABEL" \
    --num_trees=1 --tree_depth="$DEPTH" --num_threads=1 \
    --feature_split_type=Oblique --max_num_projections="$MAXPROJ" \
    --growing_strategy=Local \
  >> "$OUT/run.log" 2>&1
echo "run exit=$? $(date -u +%FT%TZ)" | tee -a "$OUT/run.log"
echo "Dumps: $CGDIR ; parse with: python3 $REPO/benchmarks/cachegrind/parseC.py $CGDIR"
