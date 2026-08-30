#!/bin/bash
# usage: run_seeds.sh <binary> <compiler> <arm> <out.csv> [parallel]
set -u
BIN=$1; COMP=$2; ARM=$3; OUT=$4; PAR=${5:-8}
S=$(dirname "$0")
WT=/home/ariel/prog/ydf/wt-lazy-shuffle
export TEST_TMPDIR=$S/tmp; mkdir -p $TEST_TMPDIR $S/logs_${COMP}_${ARM}
FILTER='RandomForestOnAbalone.SparseOblique:RandomForestOnAdult.NoWinnerTakeAllWithWeights:RandomForestOnSimPTE.LowerBound'
one() {
  seed=$1
  log=$S/logs_${COMP}_${ARM}/$seed.log
  cd $WT
  if [ "$seed" = default ]; then env -u YDF_TEST_SEED $BIN --gtest_filter=$FILTER > $log 2>&1
  else YDF_TEST_SEED=$seed $BIN --gtest_filter=$FILTER > $log 2>&1; fi
  grep '^SEED_STUDY' $log | awk -v c=$COMP -v a=$ARM '{split($3,s,"="); split($4,m,"="); print c","a","$2","s[2]","m[2]}'
}
export -f one; export S WT BIN COMP ARM FILTER
{ echo default; seq 3001 3099; } | xargs -P $PAR -I{} bash -c 'one {}' > $OUT.tmp
echo "compiler,arm,test,seed,metric" > $OUT
sort -t, -k3,3 -k4,4 $OUT.tmp >> $OUT; rm $OUT.tmp
echo "rows=$(($(wc -l < $OUT)-1)) -> $OUT"
