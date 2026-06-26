#!/usr/bin/env bash
# Build the DW1 oblique trainer with Method C (callgrind per-depth) instrumentation.
#
# The hooks live in learner/decision_tree/training.cc behind #ifdef YDF_CALLGRIND_DEPTH
# (no-op unless this define is set), so we enable them on training.cc ONLY via
# --per_file_copt. -march=skylake caps the ISA at AVX2: Valgrind 3.22 cannot decode
# the AVX-512 that -march=native emits. The gather access pattern is SIMD-width
# independent, so AVX2 is representative for cache behaviour.
#
# Requirements: bazel, Intel oneAPI (icx/icpx). Adjust ONEAPI if your install path
# differs. Run from anywhere; paths are resolved relative to this script.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ONEAPI="${ONEAPI:-/opt/intel/oneapi/compiler/latest/bin}"
cd "$REPO"
export PATH="$ONEAPI:$PATH"

bazel build -c opt --config=profiler --config=depthwise_1_pass \
  --copt=-march=skylake \
  --repo_env=CC="$ONEAPI/icx" \
  --repo_env=CXX="$ONEAPI/icpx" \
  --per_file_copt='decision_tree/training\.cc@-DYDF_CALLGRIND_DEPTH' \
  //examples:train_oblique_forest

echo "Built (instrumented): $REPO/bazel-bin/examples/train_oblique_forest"
echo "NOTE: this binary is the Method C build. Rebuild WITHOUT --per_file_copt before timing benchmarks."
