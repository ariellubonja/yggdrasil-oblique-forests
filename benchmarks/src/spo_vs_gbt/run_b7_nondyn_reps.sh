#!/usr/bin/env bash
# 2026-09-17: reps 2..7 (7 total with rep 1) of the six NON-dynamic B7 map arms
# on the cells_b7_dynscalar.json grid (19 datasets x depths {6,10,16,24,full}),
# so the depth table can carry mean +- std over 7 reps. Two groups, run in
# sequence so the fast arms' reps land first (user preference: slow scalar
# results last): "fast" = rand_vec, rand256_vec, exact_hwy; "slow" =
# rand_scalar, rand256_scalar, exact_stdsort. Within a group: rep-major, then
# depth (cheap depths first), then dataset-major cells. Each rep goes to its
# own file results/speedup_map_rep<N>.csv (run_speedup_map.py has no rep
# column; its skip-existing is per file and keyed on arm, so a rep is
# resumable and both groups append to the same rep file). Rep 1 stays in
# speedup_map.csv. Holds run_all.lock throughout.
set -uo pipefail
WORK_DIR=/home/ubuntu/spo_vs_gbt; REPO_DIR=/home/ubuntu/yggdrasil-oblique-forests
SCRIPT_DIR="$REPO_DIR/benchmarks/src/spo_vs_gbt"; PY=/home/ubuntu/gbt_venv/bin/python
LOG_DIR="$WORK_DIR/logs"; RES_DIR="$REPO_DIR/benchmarks/results/runtime/speedup_map_by_dataset"
L="$LOG_DIR/run_b7_nondyn_reps.log"
exec 9>"$WORK_DIR/run_all.lock"
echo "[nondyn] $(date -u +%FT%TZ) waiting for run_all.lock" | tee -a "$L"
flock 9
echo "[nondyn] $(date -u +%FT%TZ) start" | tee -a "$L"
for group in fast slow; do
  for rep in 2 3 4 5 6 7; do
    OUT="$WORK_DIR/results/speedup_map_rep${rep}.csv"
    for tag in d6 d10 d16 d24 full; do
      echo "[nondyn] $(date -u +%FT%TZ) $group rep $rep depth $tag start" | tee -a "$L"
      "$PY" "$SCRIPT_DIR/run_speedup_map.py" --cells-file "$SCRIPT_DIR/cells_b7_nondyn_${group}_${tag}.json" \
        --threads 48 --timeout 14400 --out "$OUT" > "$LOG_DIR/b7_nondyn_${group}_rep${rep}_${tag}.log" 2>&1
      rc=$?
      cp -f "$OUT" "$RES_DIR/"
      echo "[nondyn] $(date -u +%FT%TZ) $group rep $rep depth $tag DONE exit $rc" | tee -a "$L"
      touch "$WORK_DIR/b7_nondyn_${group}_rep${rep}_${tag}_DONE"
    done
    touch "$WORK_DIR/b7_nondyn_${group}_rep${rep}_DONE"
  done
  touch "$WORK_DIR/b7_nondyn_${group}_DONE"
done
touch "$WORK_DIR/b7_nondyn_reps_DONE"
echo "[nondyn] $(date -u +%FT%TZ) ALL DONE" | tee -a "$L"
