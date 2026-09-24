#!/usr/bin/env bash
# Run the overall-depth-table suite (paper table `table_overall_depth.tex`) on ONE new CSV dataset.
#
#   bash benchmarks/src/spo_vs_gbt/run_b7_dataset.sh <NAME> <train.csv> <label_col>
#
# NAME is the `dataset` key in speedup_map.csv (e.g. YOUTUBE8M) — add a row
# ("<NAME>", "<display> R$\times$C") to DATASETS in make_overall_depth_table.py afterwards.
# The CSV must follow the dataset policy in CLAUDE.md: fully numeric, NaN-free, binary 0/1 label.
#
# What it runs (identical to the YouTube-8M run of 2026-09-24, cells_b7_youtube8m_*.json):
#   6 SPO-RF arms x depths {6,10,16,24,purity(-1)}, 240 trees, min_examples 1, 48 threads, seed 1,
#   1 run -> $WORK_DIR/results/speedup_map.csv. Stage "fast" = exact_hwy, rand_vec, rand256_vec,
#   dyn_vec (dataset-outer, depth inner); stage "slow" = rand_scalar, rand256_scalar. Each cell is
#   one train_oblique_forest invocation timed by its "Training block took" line
#   (run_speedup_map.py); resumable (cells already OK in the out file are skipped).
# Rules: one training at a time per box (holds $WORK_DIR/run_all.lock, flock fd 9); run inside
#   tmux (`tmux new-session -d -s b7_<name> 'bash ... run_b7_dataset.sh ...'`); binaries in
#   $WORK_DIR/bin/{default,scalar} (benchmarks/SETUP_FRESH_BOX.md); mask the apt timers first
#   (`sudo systemctl mask --now apt-daily.timer apt-daily-upgrade.timer`).
# After each stage: copies the CSV into the repo results dir and, with COMMIT=1, commits + pushes.
set -uo pipefail
[ $# -eq 3 ] || { echo "usage: $0 <NAME> <train.csv> <label_col>"; exit 2; }
NAME=$1; CSV=$(readlink -f "$2"); LABEL=$3
WORK_DIR=${WORK_DIR:-/home/ubuntu/spo_vs_gbt}
REPO_DIR=${REPO_DIR:-/home/ubuntu/yggdrasil-oblique-forests}
PY=${PY:-/home/ubuntu/gbt_venv/bin/python}
SCRIPT_DIR="$REPO_DIR/benchmarks/src/spo_vs_gbt"
LOG_DIR="$WORK_DIR/logs"; mkdir -p "$LOG_DIR" "$WORK_DIR/results"
RES_DIR="$REPO_DIR/benchmarks/results/runtime/speedup_map_by_dataset"
OUT=${OUT:-$WORK_DIR/results/speedup_map.csv}
L="$LOG_DIR/run_b7_${NAME}.log"
[ -s "$CSV" ] || { echo "missing $CSV"; exit 1; }
ROWS=$(( $(wc -l < "$CSV") - 1 ))
lc=$(echo "$NAME" | tr 'A-Z' 'a-z')
"$PY" - "$SCRIPT_DIR" "$lc" "$NAME" "$CSV" "$LABEL" "$ROWS" <<'PYEOF'
import json, sys
d, lc, name, csv, label, rows = sys.argv[1:]
base = {"rows": int(rows), "csv": csv, "label_col": label, "dataset": name, "min_examples": 1}
groups = {"fast": ["spo_rf_exact_hwy", "spo_rf_rand_vec", "spo_rf_rand256_vec", "spo_rf_dyn_vec"],
          "slow": ["spo_rf_rand_scalar", "spo_rf_rand256_scalar"]}
for g, arms in groups.items():
    cells = [dict(base, depth=dep, arm=a) for dep in (6, 10, 16, 24, -1) for a in arms]
    json.dump(cells, open(f"{d}/cells_b7_{lc}_{g}.json", "w"), indent=1)
PYEOF
exec 9>"$WORK_DIR/run_all.lock"
echo "[$NAME] $(date -u +%FT%TZ) rows=$ROWS waiting for run_all.lock" | tee -a "$L"
flock 9
for stage in fast slow; do
  echo "[$NAME] $(date -u +%FT%TZ) stage $stage start" | tee -a "$L"
  "$PY" "$SCRIPT_DIR/run_speedup_map.py" --cells-file "$SCRIPT_DIR/cells_b7_${lc}_${stage}.json" \
    --bin-dir "$WORK_DIR/bin" --threads 48 --timeout 14400 --out "$OUT" > "$LOG_DIR/b7_${lc}_${stage}.log" 2>&1
  rc=$?; cp -f "$OUT" "$RES_DIR/"
  echo "[$NAME] $(date -u +%FT%TZ) stage $stage DONE exit $rc" | tee -a "$L"
  touch "$WORK_DIR/b7_${lc}_${stage}_DONE"
  if [ "${COMMIT:-0}" = 1 ]; then
    cd "$REPO_DIR" && git add "$RES_DIR/$(basename "$OUT")" "$SCRIPT_DIR/cells_b7_${lc}_fast.json" "$SCRIPT_DIR/cells_b7_${lc}_slow.json" \
      && git commit -q -m "[results] $NAME on the depth table, $stage arms, depths 6/10/16/24/purity" \
      && (git push -q origin rebased-main || (git stash -q; git fetch -q origin && git rebase -q origin/rebased-main && git push -q origin rebased-main; git stash pop -q))
    echo "[$NAME] $(date -u +%FT%TZ) git push exit $?" | tee -a "$L"
  fi
done
touch "$WORK_DIR/b7_${lc}_DONE"
echo "[$NAME] $(date -u +%FT%TZ) ALL DONE" | tee -a "$L"
