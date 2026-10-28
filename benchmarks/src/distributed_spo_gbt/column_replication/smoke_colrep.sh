#!/usr/bin/env bash
# Smoke / bit-identity checks of the column-replication build against the baseline binary.
#   smoke_colrep.sh BASELINE_BIN COLREP_BIN DATA_DIR OUT_DIR
# DATA_DIR = prep_cc18_fold0.py output. Prints PASS/FAIL per check; models are saved under OUT_DIR.
set -uo pipefail
BASE=$1; NEW=$2; DATA=$3; OUT=$4
mkdir -p "$OUT"
COMMON=(--alsologtostderr --label label --split_axis SPARSE_OBLIQUE
  --sparse_oblique_num_projections_exponent 1 --sparse_oblique_max_num_projections 2147483647
  --sparse_oblique_local_projection_count=false --sparse_oblique_projection_density_factor 1.5
  --sparse_oblique_weights BINARY --max_unique_values_for_discretized_numerical=1 --worker_logs=true
  --num_trees 20 --max_depth 6)
fails=0
run() {  # run NAME BIN DATASET W T extra...  (new-binary-only flags are added when BIN is the new one and a seed is given)
  local name=$1 bin=$2 ds=$3 w=$4 t=$5; shift 5
  local extra=(); [ "$bin" = "$NEW" ] && [ $# -gt 0 ] && extra=(--force_numerical_features)
  local d=$OUT/$name; rm -rf "$d"; mkdir -p "$d"
  local t0=$(date +%s.%N)
  "$bin" "${COMMON[@]}" --train_csv "$DATA/$ds/train.csv" --test_csv "$DATA/$ds/test.csv" \
    --work_dir "$d/work" --num_workers "$w" --num_threads "$t" --model_out_dir "$d/model" "${extra[@]}" "$@" > "$d/log" 2>&1
  local rc=$? wall=$(python3 -c "print(round($(date +%s.%N)-$t0,1))")
  local acc=$(grep -m1 -E "^Accuracy:" "$d/log" | awk '{print $2}')
  local cond=$(grep -o "Conditions: .*" "$d/log" | tail -1)
  local restarts=$(grep -c "restart" "$d/log")
  local held=$(grep -A8 "Column replication" "$d/log" | grep -o "worker [0-9]*: [0-9]* owned + [0-9]* replicated" | tr '\n' ';')
  echo "  $name: rc=$rc wall=${wall}s acc=$acc $cond restarts=$restarts held=[$held]"
  [ $rc -ne 0 ] && { echo "  FAIL $name rc=$rc"; tail -5 "$d/log"; fails=$((fails+1)); }
  [ "$restarts" != "0" ] && { echo "  FAIL $name: iteration restarts logged"; fails=$((fails+1)); }
  rm -rf "$d/work"
}
# Trees + GBT header carry the learned model; header.pb holds run metadata (date, uid).
trees_equal() { cmp -s "$OUT/$1/model/nodes-00000-of-00001" "$OUT/$2/model/nodes-00000-of-00001" && cmp -s "$OUT/$1/model/gradient_boosted_trees_header.pb" "$OUT/$2/model/gradient_boosted_trees_header.pb"; }
same_model() {  # same_model A B -> PASS if the trees are byte-identical
  if trees_equal "$1" "$2"; then echo "  PASS identical trees: $1 == $2 (data_spec $(cmp -s "$OUT/$1/model/data_spec.pb" "$OUT/$2/model/data_spec.pb" && echo same || echo DIFFERS))"; else echo "  FAIL trees differ: $1 vs $2"; fails=$((fails+1)); fi
}
diff_model() {
  if trees_equal "$1" "$2"; then echo "  FAIL trees unexpectedly identical: $1 == $2"; fails=$((fails+1)); else echo "  PASS trees differ as expected: $1 != $2"; fi
}
for DS in task_10093_banknote-authentication task_43_spambase task_9946_wdbc; do
  echo "== $DS: replication 0 must reproduce the baseline binary bit for bit (W=4, T=1, default seed; no new flags)"
  run ${DS}_base_r0 "$BASE" $DS 4 1
  run ${DS}_new_r0 "$NEW" $DS 4 1 --column_replication 0
  same_model ${DS}_base_r0 ${DS}_new_r0
done
DS=task_43_spambase
echo "== $DS: replication 0.5 / 1.0 train end to end (W=4)"
run ${DS}_new_r05_t1 "$NEW" $DS 4 1 --seed 1 --column_replication 0.5
run ${DS}_new_r10_t1 "$NEW" $DS 4 1 --seed 1 --column_replication 1.0
echo "== thread count must not change the model (r=0.5, T=1 vs T=4)"
run ${DS}_new_r05_t4 "$NEW" $DS 4 4 --seed 1 --column_replication 0.5
same_model ${DS}_new_r05_t1 ${DS}_new_r05_t4
echo "== seed changes the model and the replica draw (r=0.5, seed 1 vs 2)"
run ${DS}_new_r05_s2 "$NEW" $DS 4 1 --seed 2 --column_replication 0.5
diff_model ${DS}_new_r05_t1 ${DS}_new_r05_s2
echo "== replication levels must give different models (r=0 vs 0.5 vs 1, seed 1)"
run ${DS}_new_r0_s1 "$NEW" $DS 4 1 --seed 1 --column_replication 0
diff_model ${DS}_new_r0_s1 ${DS}_new_r05_t1
diff_model ${DS}_new_r05_t1 ${DS}_new_r10_t1
echo "== W=1: replication is a no-op (no foreign columns), r=1 == r=0"
run ${DS}_w1_r0 "$NEW" $DS 1 1 --seed 1 --column_replication 0
run ${DS}_w1_r1 "$NEW" $DS 1 1 --seed 1 --column_replication 1
same_model ${DS}_w1_r0 ${DS}_w1_r1
echo "== run-to-run determinism at r=0.5 and r=1 (same flags twice)"
run ${DS}_new_r05_t1_again "$NEW" $DS 4 1 --seed 1 --column_replication 0.5
same_model ${DS}_new_r05_t1 ${DS}_new_r05_t1_again
run ${DS}_new_r10_t1_again "$NEW" $DS 4 1 --seed 1 --column_replication 1.0
same_model ${DS}_new_r10_t1 ${DS}_new_r10_t1_again
echo "== bitmaps from a different holder: DGBT_OBLIQUE_EVALUATOR_LAST=1 must give the same model (r=0.5, r=1)"
DGBT_OBLIQUE_EVALUATOR_LAST=1 run ${DS}_new_r05_lastholder "$NEW" $DS 4 1 --seed 1 --column_replication 0.5
same_model ${DS}_new_r05_t1 ${DS}_new_r05_lastholder
DGBT_OBLIQUE_EVALUATOR_LAST=1 run ${DS}_new_r10_lastholder "$NEW" $DS 4 1 --seed 1 --column_replication 1.0
same_model ${DS}_new_r10_t1 ${DS}_new_r10_lastholder
echo "== wide dataset at r=1 (madelon F=500, W=4): every worker holds all 500"
run task_9976_madelon_r10 "$NEW" task_9976_madelon 4 2 --seed 1 --column_replication 1.0
echo "== mixed-origin dataset (adult, encoded categoricals) r=0.3"
run task_7592_adult_r03 "$NEW" task_7592_adult 4 2 --seed 3 --column_replication 0.3
echo "FAILS=$fails"
