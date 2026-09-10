#!/usr/bin/env bash
# Roofline collection: 3 arms x shapes. Usage: [CACHESIM=0|1] [INTERVAL=20|100] run_roofline.sh <shape> [arms...]
# No --stacks (not needed for a roofline). No cache simulation by default (75-140x slowdown). INTERVAL=100 for the
# May-2025 binary on big shapes: its recursive tree grower makes Advisor's survey stack file huge at 20 ms and the
# single-threaded finalization takes hours (see benchmarks/results/roofline/README.md).
S=${ROOFLINE_OUT:-$PWD/roofline_out}
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
set -uo pipefail
FORK=/home/ubuntu/yggdrasil-oblique-forests
MAY=/home/ubuntu/ydf-may-2025
SHAPE=$1; shift
CACHESIM=${CACHESIM:-0}; INTERVAL=${INTERVAL:-20}; if [[ $CACHESIM == 1 ]]; then CACHESIM_FLAG=--enable-cache-simulation; SUFFIX=""; MEMLVL=--memory-level=L1_L2_L3_DRAM; else CACHESIM_FLAG=""; SUFFIX=_nocs; MEMLVL=""; fi
ARMS=("$@"); [[ ${#ARMS[@]} -eq 0 ]] && ARMS=(dyn hwy_exact may2025_exact)
case $SHAPE in
  trunk100k) DATA="--input_mode trunk --rows 100000 --cols 4096" ;;
  higgs)     DATA="--input_mode csv --train_csv $FORK/benchmarks/data/HIGGS_with_header.csv --label_col class" ;;
  epsilon)   DATA="--input_mode csv --train_csv $FORK/benchmarks/data/epsilon_normalized_train.csv --label_col label" ;;
  trunk1500k) DATA="--input_mode trunk --rows 1500000 --cols 4096" ;;
  *) echo "bad shape"; exit 2 ;;
esac
COMMON="--num_trees=48 --num_threads=48 --seed=1"
cd $S
for arm in "${ARMS[@]}"; do
  case $arm in
    dyn)           BIN=${FORK_BIN:-$S/tof_fork_g};    ARGS="--numerical_split_type 'Dynamic Random Histogram' --histogram_num_bins=64 --dynamic_split_threshold=250" ;;
    hwy_exact)     BIN=${FORK_BIN:-$S/tof_fork_g};    ARGS="--numerical_split_type Exact" ;;
    may2025_exact) BIN=${MAY_BIN:-$S/tof_may2025_g}; ARGS="--numerical_split_type Exact" ;;
  esac
  P=$S/adv_${arm}_${SHAPE}${SUFFIX}
  rm -rf "$P"
  echo "=== $(date -Is) START $arm $SHAPE" | tee -a $S/run.log
  eval advisor --collect=roofline ${CACHESIM_FLAG} --interval=${INTERVAL} --no-stack-stitching \
     --search-dir src:r=$FORK --search-dir src:r=$MAY --project-dir=$P \
     -- $BIN $DATA $COMMON $ARGS > $P.log 2>&1
  rc=$?
  grep -E "Training block took|Loading/Init|Elapsed|CPU Time|GFLOPS|GINTOPS" $P.log | tee -a $S/run.log
  echo "=== $(date -Is) END $arm $SHAPE rc=$rc" | tee -a $S/run.log
  timeout 3600 advisor --report=survey --project-dir=$P --format=csv --show-all-columns --report-output=$P.survey.csv >/dev/null 2>&1
  timeout 3600 advisor --report=survey --show-functions --project-dir=$P --format=csv --show-all-columns --report-output=$P.functions.csv >/dev/null 2>&1
  timeout 3600 advisor --report=top-down --project-dir=$P --format=csv --show-all-columns --report-output=$P.topdown.csv >/dev/null 2>&1
  timeout 600 advisor --report=roofs --project-dir=$P --format=csv --report-output=$P.roofs.csv >/dev/null 2>&1
  timeout 1800 advisor --report=roofline --project-dir=$P --data-type=mixed $MEMLVL --report-output=$P.roofline_mixed.html >/dev/null 2>&1
  timeout 1800 advisor --report=roofline --project-dir=$P --data-type=float $MEMLVL --report-output=$P.roofline_float.html >/dev/null 2>&1
done
