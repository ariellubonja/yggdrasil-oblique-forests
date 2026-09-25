#!/usr/bin/env bash
# Build the three harness variants needed by the SPO-vs-GBT / speedup-map study
# and copy each to a stable path (bazel-bin is overwritten by every config).
set -euo pipefail
cd /home/ubuntu/yggdrasil-oblique-forests
set +u +e
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
set -u -e
which icx
BASE=(-c opt --cxxopt="-O3" --cxxopt="-march=native" --repo_env=CC=icx --repo_env=CXX=icpx)
OUT=/home/ubuntu/spo_vs_gbt/bin
build() { # $1 = name, rest = extra bazel flags
  local name="$1"; shift
  echo "=== $(date -u +%FT%TZ) build $name $*"
  bazel build "${BASE[@]}" "$@" //examples:train_oblique_forest
  cp -f bazel-bin/examples/train_oblique_forest "$OUT/$name"
  chmod u+w "$OUT/$name"
  n=$(strings "$OUT/$name" | grep -c -i -E "intel|svml" || true)
  echo "    toolchain check: $n intel/svml strings (icx expected 100+)"
  git rev-parse --short HEAD > "$OUT/$name.gitsha"
}
build default
build exact_std_sort --config=exact_std_sort
build scalar --config=disable_std_upper_bound_vectorization
# leave bazel-bin at the default build
build default
echo "BUILD_ALL_DONE $(date -u +%FT%TZ)"
