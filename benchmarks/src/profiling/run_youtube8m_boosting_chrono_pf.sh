#!/usr/bin/env bash
set -uo pipefail

# Re-run the YouTube-8M SPO-GBT per-function chrono (Figure 1, SPO-GBT panel) with
# P = F projections. The 2026-09-24 runs (commit 83d2d460: 7 Boosting arms, 1 tree,
# 1 thread, GBT default depth 6) predate the P = F switch and passed no projection
# flags, so they measured P = ceil(sqrt(F)) = 34 on F = 1152. Same seven invocations
# (recorded in each CSV's "Helper invocation" cell) plus
#   --num_projections_exponent 1      explicit P = F
#   --max_num_projections 2147483647  F = 1152 exceeds the harness cap of 1000
# Results overwrite the same raw CSVs (git history keeps the sqrt(F) versions; the
# raw .log files are timestamped, so both stay on disk). The full-depth set
# (commit f2c873ea, --tree_depth 1000) is not re-run here.
#
# Waits for the P = F accuracy re-run (run_hve_gbt_pf.sh) to finish, then takes the
# box lock so the 1-thread chrono runs alone.

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
WAIT_FOR=/home/ubuntu/spo_vs_gbt/hve_gbt_pf_DONE
LOCK=/home/ubuntu/spo_vs_gbt/run_all.lock
DONE_MARK=/home/ubuntu/spo_vs_gbt/youtube8m_boosting_chrono_pf_DONE
OUT=ariel_notes/results/2026-10-25_youtube8m_boosting_chrono_pf
mkdir -p "$OUT"
LOG=$OUT/run.log
log() { echo "[yt8m-chrono-pf] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

log "waiting for $WAIT_FOR"
until [[ -f "$WAIT_FOR" ]]; do sleep 300; done
log "accuracy re-run done; acquiring lock"

COMMON=(--input_mode csv --save_log --train_csv benchmarks/data/youtube8m/youtube8m_video_train.csv
        --label_col class --ensemble_method Boosting
        --num_projections_exponent 1 --max_num_projections 2147483647)

run() { # $1 = experiment name, rest = arm flags
  local name=$1; shift
  log "RUN $name"
  if flock "$LOCK" python3 benchmarks/src/profiling/parallel_chrono.py "${COMMON[@]}" "$@" \
       --experiment_name "$name" >>"$OUT/$name.log" 2>&1; then
    log "OK $name"
  else
    log "FAILED $name (see $OUT/$name.log)"
  fi
}

run exact                  --numerical_split_type Exact --vectorized avx2
run vectorized-64          --numerical_split_type Random --vectorized avx2 --histogram_num_bins 64
run vectorized-256         --numerical_split_type Random --vectorized avx512 --histogram_num_bins 256
run scalar-64              --numerical_split_type Random --vectorized None --histogram_num_bins 64
run scalar-256             --numerical_split_type Random --vectorized None --histogram_num_bins 256
run vectorized-64-thr250   --numerical_split_type 'Dynamic Random Histogram' --vectorized avx2 --histogram_num_bins 64 --dynamic_split_threshold 250
run vectorized-256-thr1000 --numerical_split_type 'Dynamic Random Histogram' --vectorized avx512 --histogram_num_bins 256 --dynamic_split_threshold 1000

log "all done"
touch "$DONE_MARK"
