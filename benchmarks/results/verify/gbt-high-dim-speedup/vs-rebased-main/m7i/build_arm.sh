#!/usr/bin/env bash
# build_arm.sh <out_binary_path>   (EXTRA_BAZEL_CONFIGS from env, same flags as runtime.sh)
set -euo pipefail
cd /home/ubuntu/yggdrasil-oblique-forests
source benchmarks/utils/bench_common.sh
bazel_build -c opt --cxxopt="-O3" --cxxopt="-march=native" //examples:train_oblique_forest
cp -f bazel-bin/examples/train_oblique_forest "$1"
sha256sum "$1"
