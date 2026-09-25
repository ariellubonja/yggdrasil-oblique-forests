#!/usr/bin/env bash
# 2026-09-25: the arm missing from the depth-table rows of Shifts weather, ClimSim and Jane Street —
# spo_rf_dyn256_vec (Dynamic Random Histogram, 256 bins, AVX-512 binner, --dynamic_split_threshold 1000)
# x depths {6,10,16,24,purity}, 240 trees, min_examples 1, 48 threads, seed 1, 1 run, dataset-outer
# -> speedup_map.csv. Holds run_all.lock; commits + pushes at the end (union-merges on conflict).
set -uo pipefail
WORK_DIR=/home/ubuntu/spo_vs_gbt; REPO_DIR=/home/ubuntu/yggdrasil-oblique-forests
SCRIPT_DIR="$REPO_DIR/benchmarks/src/spo_vs_gbt"; PY=${PY:-$REPO_DIR/.venv/bin/python}
LOG_DIR="$WORK_DIR/logs"; RES_DIR="$REPO_DIR/benchmarks/results/runtime/speedup_map_by_dataset"
OUT="$WORK_DIR/results/speedup_map.csv"; L="$LOG_DIR/run_b7_new3_dyn256.log"
exec 9>"$WORK_DIR/run_all.lock"
echo "[new3-dyn256] $(date -u +%FT%TZ) waiting for run_all.lock" | tee -a "$L"
flock 9
echo "[new3-dyn256] $(date -u +%FT%TZ) start" | tee -a "$L"
"$PY" "$SCRIPT_DIR/run_speedup_map.py" --cells-file "$SCRIPT_DIR/cells_b7_new3_dyn256.json" \
  --bin-dir "$WORK_DIR/bin" --threads 48 --timeout 14400 --out "$OUT" > "$LOG_DIR/b7_new3_dyn256.log" 2>&1
rc=$?
echo "[new3-dyn256] $(date -u +%FT%TZ) DONE exit $rc" | tee -a "$L"
cd "$REPO_DIR" && git pull -q --rebase origin rebased-main
"$PY" - "$OUT" "$RES_DIR/speedup_map.csv" <<'PYEOF'
import sys, pandas as pd
w, r = sys.argv[1:]; m = pd.concat([pd.read_csv(r), pd.read_csv(w)]).drop_duplicates(); m.to_csv(r, index=False); print("merged rows", len(m))
PYEOF
git add "$RES_DIR/speedup_map.csv" "$SCRIPT_DIR/cells_b7_new3_dyn256.json" "$SCRIPT_DIR/run_b7_new3_dyn256.sh" \
  && git commit -q -m "[results] Shifts weather, ClimSim, Jane Street: spo_rf_dyn256_vec (AVX-512 256-bin dynamic, threshold 1000), depths 6/10/16/24/purity

                                                       " \
  && (git push -q origin rebased-main && echo "[new3-dyn256] push ok" || echo "[new3-dyn256] PUSH FAILED; resolve by hand") | tee -a "$L"
echo "[new3-dyn256] $(date -u +%FT%TZ) ALL DONE" | tee -a "$L"
