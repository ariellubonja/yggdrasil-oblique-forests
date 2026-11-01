#!/bin/bash
# m7i: tab:dist-runtime (in-process MULTI_THREAD, 50 trees, depth 6) on the bug-fixed driver (main + these scripts), same flags as
# the 2026-10-12 grid (P = ceil(sqrt F) per worker, 6 threads/worker), plus same-boot single-machine anchors (paper binary).
source /opt/intel/oneapi/setvars.sh > /dev/null  # before set -u
set -u
O=~/runtime_fix/familyC; mkdir -p $O/bin $O/logs
cd ~/yggdrasil-oblique-forests && git fetch -q origin && (git worktree add --detach ~/wt_runtime origin/kdd27-runtime-reruns 2>/dev/null || true)
cd ~/wt_runtime && git checkout -q --detach origin/kdd27-runtime-reruns && git log -1 --oneline > $O/bin/PROVENANCE.txt
bazel build //examples:train_distributed_gbt > $O/build.log 2>&1 || { echo BUILD_FAILED; exit 1; }
cp bazel-bin/examples/train_distributed_gbt $O/bin/
S=~/wt_runtime/benchmarks/src/distributed_spo_gbt
export BIN=$O/bin/train_distributed_gbt OUT=$O RESULTS=$O/results.csv WORK=$O/work THREADS=6 GITSHA=$(git rev-parse --short HEAD)
for ds in HIGGS SUSY EPSILON YOUTUBE8M; do for W in 1 4; do for ax in SPARSE_OBLIQUE AXIS_ALIGNED; do
  echo "### $(date -u +%FT%TZ) START $ds W=$W $ax"; bash $S/run_dgbt.sh $ds $W $ax 50; echo "### END rc=$?"
done; done; rm -rf $O/work/$ds; done
unset BIN THREADS GITSHA; export RESULTS=$O/anchors.csv
for ds in HIGGS SUSY EPSILON YOUTUBE8M; do for sp in Oblique AxisAligned; do
  echo "### $(date -u +%FT%TZ) START single $ds $sp"; bash $S/run_single_anchor.sh $ds 50 $sp; echo "### END rc=$?"
done; done
echo "### ALL DONE $(date -u +%FT%TZ)" > $O/DONE
