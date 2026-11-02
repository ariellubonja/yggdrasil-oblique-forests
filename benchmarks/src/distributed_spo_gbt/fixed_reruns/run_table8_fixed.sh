#!/bin/bash
# KDD28 Table 8 distributed-SPO arms with every fix (branch dgbt-spo-full-fixes), in-process workers, on one Linux box.
# Variants (each: 33 datasets x their folds x W in 1,4,8,16,32 -> 765 cells, or 612 for parity without W=1):
#   fullfix          paper flags (0/1 columns typed BOOLEAN by the driver, r=0, local projection count)
#   fullfix_numeric  + --force_numerical_features (0/1 columns NUMERICAL, as single-machine)
#   fullfix_rep      fullfix again: must be identical cell for cell (determinism)
#   parity           numeric typing + column_replication 1 + split budget (cluster total P = F): matches single-machine
# Usage: run_table8_fixed.sh <out_root> <train_distributed_gbt> <folds_dir> [variants...]
# Folds: rebuild_table8_data.py. Binary: bazel build //examples:train_distributed_gbt on dgbt-spo-full-fixes.
set -u
OUT=${1:?out_root}; BIN=${2:?train_distributed_gbt}; FOLDS=${3:?folds_dir}; shift 3
VARIANTS=${*:-fullfix fullfix_numeric fullfix_rep parity}
HERE=$(cd "$(dirname "$0")" && pwd)
# Threads per worker = NCPU / W change the model (block-wise root statistics): the recorded run used NCPU=16.
NCPU=${NCPU:-16}
A="1:SPO 4:SPO 8:SPO 16:SPO 32:SPO"
for v in $VARIANTS; do
  mkdir -p $OUT/$v/bin; ln -sfn $FOLDS $OUT/$v/folds; cp $BIN $OUT/$v/bin/train_distributed_gbt
  case $v in
    fullfix|fullfix_rep) X="" ; ARMS=$A ;;
    fullfix_numeric)     X="--force_numerical_features=true" ; ARMS=$A ;;
    parity)              X="--force_numerical_features=true --column_replication 1 --sparse_oblique_split_budget --sparse_oblique_local_projection_count=false"
                         ARMS="4:SPO 8:SPO 16:SPO 32:SPO" ;;
    *) echo "unknown variant $v"; exit 1 ;;
  esac
  # Variants run concurrently (4 x 16 threads oversubscribes a 48-core box; only runtime suffers).
  NCPU=$NCPU ARMS="$ARMS" DIST_EXTRA="$X" \
    $HERE/../run_table8_local.sh $OUT/$v > $OUT/$v/run.out 2>&1 &
done
wait
echo "### DONE $(date -u +%FT%TZ)"
