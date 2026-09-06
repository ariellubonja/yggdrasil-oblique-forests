#!/usr/bin/env bash
# Driver 3 (SPEC.md v2 addendum A8 / PROTOCOL.md v2): the serialized
# SPO-vs-GBT study chain, end to end.
#
# Usage:  tmux new-session -d -s spo_run 'bash run_all.sh'
#         STAGES=smoke,suite bash run_all.sh   # rerun only these stages
#
# Stages, strictly serialized (never two training binaries at once):
#   0. smoke  — run_suite.py --smoke, diabetes only, all 15 arms (sanity check;
#               writes into the SAME suite_results.csv stage 1 continues, so
#               its rows are not repeated -- skip-existing is on by default).
#   1. suite  — run_suite.py full CV suite: tabarena_binary_csv (5-fold) +
#               tabred_binary_csv (--chrono-holdout, all 3 datasets), 15 arms.
#   2. large  — run_suite.py --mode large: HIGGS, SUSY, Epsilon, 15 arms,
#               seed 1 (default), rep 0 (default) i.e. PROTOCOL.md's "rep 1",
#               --timeout-s 10800.
#   3. b1/b2/b4/b5 — run_speedup_map.py, one stage (and DONE sentinel) per
#               cells file, all writing into the same speedup_map.csv
#               (disjoint cell keys; each independently resumable). Kept as
#               4 separate sentinels rather than one "stage 3" sentinel so a
#               single cells file can be rerun without redoing the others
#               (documented deviation from the addendum's numbering, not from
#               its content -- see oblique_context notes / hand-off doc).
#   4. rep2   — run_suite.py --mode large --rep 2, PROTOCOL.md D12(a): a 2nd
#               timing rep for the 11 headline arms only.
#   5. seed2  — run_suite.py --mode large --seed 2, PROTOCOL.md D12(b): a 2nd
#               YDF seed for the 7 YDF RF-family arms only (accuracy variance).
#
# Each stage logs to /home/ubuntu/spo_vs_gbt/logs/<stage>.log and, on success,
# touches /home/ubuntu/spo_vs_gbt/<STAGE>_DONE; a stage is skipped if its DONE
# file already exists, so the chain is resumable end-to-end. Set STAGES to a
# comma list of stage names (smoke,suite,large,b1,b2,b4,b5,rep2,seed2) to
# additionally restrict which stages run this invocation (still subject to
# the DONE-file skip above -- rm the DONE file first to force a rerun).
# `set -uo pipefail` WITHOUT -e: a failed stage is logged and the chain moves
# on to the next one rather than aborting.
#
# Background loggers for the whole chain, killed on any exit from this
# script: MemAvailable every 60s -> logs/mem.log, and
# `sudo turbostat --quiet --interval 60 --show Avg_MHz,Busy%,Bzy_MHz,PkgTmp`
# -> logs/turbostat.log (sudo is passwordless on this box; falls back to
# logging the mean of /proc/cpuinfo "cpu MHz" every 60s if turbostat refuses
# to run here).
set -uo pipefail

WORK_DIR=/home/ubuntu/spo_vs_gbt
REPO_DIR=/home/ubuntu/yggdrasil-oblique-forests
SCRIPT_DIR="$REPO_DIR/benchmarks/evaluation/spo_vs_gbt"
PY=/home/ubuntu/gbt_venv/bin/python
LOG_DIR="$WORK_DIR/logs"
RESULTS_DIR="$WORK_DIR/results"
DEST_DIR="$REPO_DIR/benchmarks/results/spo_vs_gbt"

mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$WORK_DIR/folds" "$LOG_DIR/runs"

# --- single-instance guard: refuse to start a second concurrent chain ---
# (this project's own history: two overlapping --num_threads=-1 binaries
# inflate timings ~1.7x -- see oblique_context notes). Held for the whole
# script via fd 9 on a lock file; released automatically on any exit.
LOCK_FILE="$WORK_DIR/run_all.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[run_all] $(date -u +%FT%TZ) another run_all.sh is already running " \
       "(lock held on $LOCK_FILE); refusing to start a second, overlapping " \
       "chain -- exiting" >&2
  exit 1
fi

TABARENA_DIR="$REPO_DIR/benchmarks/data/tabarena_binary_csv"
TABRED_DIR="$REPO_DIR/benchmarks/data/tabred_binary_csv"
TABRED_CHRONO_NAMES="ecom-offers,homecredit-default,homesite-insurance"

SUITE_OUT="$RESULTS_DIR/suite_results.csv"
LARGE_OUT="$RESULTS_DIR/large_results.csv"
SPEEDUP_OUT="$RESULTS_DIR/speedup_map.csv"

# PROTOCOL.md v2 A8 large-tier dataset specs (NAME=TRAIN:TEST:LABEL).
HIGGS_TRAIN="$REPO_DIR/benchmarks/data/HIGGS_train_10500k.csv"
HIGGS_TEST="$REPO_DIR/benchmarks/data/HIGGS_test_500k.csv"
SUSY_TRAIN="$REPO_DIR/benchmarks/data/SUSY_train_4500k.csv"
SUSY_TEST="$REPO_DIR/benchmarks/data/SUSY_test_500k.csv"
EPSILON_TRAIN="$REPO_DIR/benchmarks/data/epsilon_normalized_train.csv"
EPSILON_TEST="$WORK_DIR/data/epsilon_test_100k.csv"
LARGE_DATASETS=(
  "HIGGS=${HIGGS_TRAIN}:${HIGGS_TEST}:class"
  "SUSY=${SUSY_TRAIN}:${SUSY_TEST}:class"
  "EPSILON=${EPSILON_TRAIN}:${EPSILON_TEST}:label"
)
LARGE_TIMEOUT_S=10800

# PROTOCOL.md D12(a): 2nd timing rep, headline arms only (11 of 15 -- the
# ydf_fork/library pair the paper actually compares; xgboost_rf/lightgbm_rf
# and the *_scalar/*_rand_* SPO ablations are excluded from the repeat).
HEADLINE_ARMS="spo_rf_exact_hwy,spo_rf_dyn_vec,aa_rf_exact,xgboost_rf,lightgbm_rf,spo_gbt_exact_hwy,spo_gbt_dyn_vec,aa_gbt_exact,xgboost,lightgbm,catboost"
# PROTOCOL.md D12(b): 2nd YDF seed, the 7 YDF RF-family arms only.
YDF_RF_ARMS="spo_rf_exact_stdsort,spo_rf_exact_hwy,spo_rf_rand_scalar,spo_rf_rand_vec,spo_rf_dyn_scalar,spo_rf_dyn_vec,aa_rf_exact"

# --- STAGES filter (env var): comma list of stage names to allow this run.
# Empty/unset = no extra filtering (every non-DONE stage runs). Entries are
# whitespace-trimmed so `STAGES="smoke, suite"` (a space after the comma)
# does not silently drop "suite" -- it split as literal " suite" otherwise,
# which never equals the bare stage name "suite" in stage_allowed()'s ==.
IFS=',' read -r -a STAGES_FILTER <<< "${STAGES:-}"
for _i in "${!STAGES_FILTER[@]}"; do
  _s="${STAGES_FILTER[$_i]}"
  _s="${_s#"${_s%%[![:space:]]*}"}"   # trim leading whitespace
  _s="${_s%"${_s##*[![:space:]]}"}"   # trim trailing whitespace
  STAGES_FILTER[$_i]="$_s"
done
unset _i _s
stage_allowed() {
  local name="$1"
  [[ ${#STAGES_FILTER[@]} -eq 0 || -z "${STAGES_FILTER[0]}" ]] && return 0
  local s
  for s in "${STAGES_FILTER[@]}"; do
    [[ "$s" == "$name" ]] && return 0
  done
  return 1
}

# --- background loggers; both killed on any exit from this script ---
MEM_LOG="$LOG_DIR/mem.log"
(
  while true; do
    ts=$(date -u +%FT%TZ)
    avail=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    echo "$ts MemAvailable_kB=$avail" >> "$MEM_LOG"
    sleep 60
  done
) &
MEM_PID=$!

TURBOSTAT_LOG="$LOG_DIR/turbostat.log"
if sudo -n turbostat --quiet --num_iterations 1 --interval 1 --show Avg_MHz \
     >/dev/null 2>>"$TURBOSTAT_LOG"; then
  sudo turbostat --quiet --interval 60 \
    --show Avg_MHz,Busy%,Bzy_MHz,PkgTmp >> "$TURBOSTAT_LOG" 2>&1 &
  TURBOSTAT_PID=$!
  TURBOSTAT_MODE="turbostat"
else
  echo "[run_all] turbostat refused (sudo -n probe failed); falling back to " \
       "/proc/cpuinfo MHz mean every 60s" | tee -a "$TURBOSTAT_LOG"
  (
    while true; do
      ts=$(date -u +%FT%TZ)
      mhz=$(awk -F: '/cpu MHz/ {gsub(/ /,"",$2); sum+=$2; n++} \
                     END {if (n>0) printf "%.1f", sum/n; else print "NA"}' \
                /proc/cpuinfo)
      echo "$ts cpuinfo_MHz_mean=$mhz" >> "$TURBOSTAT_LOG"
      sleep 60
    done
  ) &
  TURBOSTAT_PID=$!
  TURBOSTAT_MODE="cpuinfo_fallback"
fi

cleanup() {
  # pkill -P first: each logger is a `while ...; sleep 60; done` loop, so the
  # in-flight `sleep` is a *child* of MEM_PID/TURBOSTAT_PID, not killed by
  # killing the loop PID alone -- an un-killed sleep would hold the script's
  # stdout fd open (e.g. under a pipe) for up to 60s after exit.
  pkill -P "$MEM_PID" >/dev/null 2>&1 || true
  kill "$MEM_PID" >/dev/null 2>&1 || true
  pkill -P "$TURBOSTAT_PID" >/dev/null 2>&1 || true
  kill "$TURBOSTAT_PID" >/dev/null 2>&1 || true
  sudo pkill -x turbostat >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[run_all] $(date -u +%FT%TZ) started, mem logger pid=$MEM_PID, " \
     "$TURBOSTAT_MODE logger pid=$TURBOSTAT_PID" | tee -a "$LOG_DIR/run_all.log"

# run_stage NAME  cmd...   — logs to logs/<NAME>.log, touches <NAME>_DONE on
# success, skips if <NAME>_DONE already exists OR NAME is excluded by the
# STAGES filter. Never aborts the chain.
run_stage() {
  local name="$1"; shift
  local done_file="$WORK_DIR/${name}_DONE"
  local log_file="$LOG_DIR/${name}.log"

  if ! stage_allowed "$name"; then
    echo "[run_all] $(date -u +%FT%TZ) $name: SKIP (excluded by STAGES=${STAGES:-})" \
      | tee -a "$LOG_DIR/run_all.log"
    return 0
  fi
  if [[ -f "$done_file" ]]; then
    echo "[run_all] $(date -u +%FT%TZ) $name: SKIP (found $done_file)" | tee -a "$LOG_DIR/run_all.log"
    return 0
  fi

  echo "[run_all] $(date -u +%FT%TZ) $name: START" | tee -a "$LOG_DIR/run_all.log"
  {
    echo "CMD: $*"
    echo "START: $(date -u +%FT%TZ)"
  } > "$log_file"

  "$@" >> "$log_file" 2>&1
  local rc=$?

  # run_suite.py's run_jobs() (used by the smoke/suite/large/rep2/seed2
  # stages) always returns 0, even when it recorded ERROR/OOM/TIMEOUT rows
  # this session -- unlike run_speedup_map.py (b1/b2/b4/b5), whose exit code
  # already reflects cell failures by design (see its own comment on why:
  # "makes a failed cell self-heal on the next invocation instead of being
  # silently frozen in as done"). Without this check, a stage with a real
  # per-row failure would still get its DONE sentinel touched below and
  # would never be retried. Grep the stage's own progress lines (printed as
  # "... status=<STATUS>") for a non-OK, non-SLOWPATH status as a
  # compensating check -- SLOWPATH is a recorded-but-successful row (A2) and
  # must not trip this. run_suite.py's DONE_STATUSES ({OK, SLOWPATH}) is the
  # authority this mirrors.
  if [[ $rc -eq 0 ]] && grep -qE ' status=(ERROR|OOM|TIMEOUT)$' "$log_file"; then
    rc=1
    echo "[run_all] non-OK row status found in $log_file despite exit 0 " \
         "(run_suite.py does not propagate per-row failures into its exit " \
         "code); treating $name as FAILED so it retries next invocation" \
         >> "$log_file"
  fi

  echo "END: $(date -u +%FT%TZ)  EXIT: $rc" >> "$log_file"
  if [[ $rc -eq 0 ]]; then
    touch "$done_file"
    echo "[run_all] $(date -u +%FT%TZ) $name: OK" | tee -a "$LOG_DIR/run_all.log"
  else
    echo "[run_all] $(date -u +%FT%TZ) $name: FAILED (exit $rc, see $log_file)" | tee -a "$LOG_DIR/run_all.log" >&2
  fi
  return 0
}

# ---------------------------------------------------------------------------
# Stage 0: smoke — diabetes only (--only), fold 0 (--smoke), all 15 arms.
# Writes into the SAME suite_results.csv that stage 1 continues, so its rows
# are not repeated (skip-existing is on by default in run_suite.py).
# run_suite.py has no per-run --trees override (unlike run_speedup_map.py),
# so this runs at each arm's normal tree count (240 RF / 300 GBT); diabetes
# is 768 rows x 8 features, still well under a minute for all 15 arms.
# ---------------------------------------------------------------------------
run_stage smoke "$PY" "$SCRIPT_DIR/run_suite.py" \
  --mode suite \
  --datasets-dir "$TABARENA_DIR" \
  --only diabetes \
  --folds 5 --fold-seed 0 \
  --work-dir "$WORK_DIR/folds" \
  --out "$SUITE_OUT" \
  --threads 48 \
  --smoke

# ---------------------------------------------------------------------------
# Stage 1: full CV suite — tabarena_binary_csv (5-fold StratifiedKFold) +
# tabred_binary_csv (D2/A3: one chronological holdout fold, all 3 datasets).
# ---------------------------------------------------------------------------
run_stage suite "$PY" "$SCRIPT_DIR/run_suite.py" \
  --mode suite \
  --datasets-dir "$TABARENA_DIR" \
  --datasets-dir "$TABRED_DIR" \
  --folds 5 --fold-seed 0 \
  --chrono-holdout "$TABRED_CHRONO_NAMES" \
  --work-dir "$WORK_DIR/folds" \
  --out "$SUITE_OUT" \
  --threads 48

# ---------------------------------------------------------------------------
# Stage 2: large single-split, seed 1 (default) / rep 0 (default) — HIGGS,
# SUSY, Epsilon, all 15 arms, 3h timeout per run (A8).
# ---------------------------------------------------------------------------
run_stage large "$PY" "$SCRIPT_DIR/run_suite.py" \
  --mode large \
  --dataset "${LARGE_DATASETS[0]}" \
  --dataset "${LARGE_DATASETS[1]}" \
  --dataset "${LARGE_DATASETS[2]}" \
  --seed 1 \
  --timeout-s "$LARGE_TIMEOUT_S" \
  --out "$LARGE_OUT" \
  --threads 48

# ---------------------------------------------------------------------------
# Stage 3: speedup map B1, B2, B4, B5 -- one sentinel per cells file, all
# appending to the same speedup_map.csv (disjoint cell keys; see header
# comment). No --trees override: each cell runs at its own arm's tree count
# from arms.py (b1/b2/b5 are RF-only = 240; b4 is GBT-only = 300 -- passing
# a single --trees for all four would silently wreck b4).
# ---------------------------------------------------------------------------
run_stage b1 "$PY" "$SCRIPT_DIR/run_speedup_map.py" \
  --cells-file "$SCRIPT_DIR/cells_b1.json" \
  --threads 48 \
  --out "$SPEEDUP_OUT"

run_stage b2 "$PY" "$SCRIPT_DIR/run_speedup_map.py" \
  --cells-file "$SCRIPT_DIR/cells_b2.json" \
  --threads 48 \
  --out "$SPEEDUP_OUT"

run_stage b4 "$PY" "$SCRIPT_DIR/run_speedup_map.py" \
  --cells-file "$SCRIPT_DIR/cells_b4.json" \
  --threads 48 \
  --out "$SPEEDUP_OUT"

run_stage b5 "$PY" "$SCRIPT_DIR/run_speedup_map.py" \
  --cells-file "$SCRIPT_DIR/cells_b5.json" \
  --threads 48 \
  --out "$SPEEDUP_OUT"

# ---------------------------------------------------------------------------
# Stage 4: large tier, rep 2 — PROTOCOL.md D12(a), headline arms only.
# ---------------------------------------------------------------------------
run_stage rep2 "$PY" "$SCRIPT_DIR/run_suite.py" \
  --mode large \
  --dataset "${LARGE_DATASETS[0]}" \
  --dataset "${LARGE_DATASETS[1]}" \
  --dataset "${LARGE_DATASETS[2]}" \
  --arms "$HEADLINE_ARMS" \
  --seed 1 --rep 2 \
  --timeout-s "$LARGE_TIMEOUT_S" \
  --out "$LARGE_OUT" \
  --threads 48

# ---------------------------------------------------------------------------
# Stage 5: large tier, seed 2 — PROTOCOL.md D12(b), the 7 YDF RF arms only.
# ---------------------------------------------------------------------------
run_stage seed2 "$PY" "$SCRIPT_DIR/run_suite.py" \
  --mode large \
  --dataset "${LARGE_DATASETS[0]}" \
  --dataset "${LARGE_DATASETS[1]}" \
  --dataset "${LARGE_DATASETS[2]}" \
  --arms "$YDF_RF_ARMS" \
  --seed 2 \
  --timeout-s "$LARGE_TIMEOUT_S" \
  --out "$LARGE_OUT" \
  --threads 48

# ---------------------------------------------------------------------------
# Final: copy results into the repo (never overwrite an in-repo CSV with an
# empty/missing one) and drop the ALL_DONE sentinel. Runs every invocation
# (not gated by STAGES/DONE) so a partial chain still publishes what exists.
# ---------------------------------------------------------------------------
mkdir -p "$DEST_DIR"
# run_suite.py writes a <out>.provenance.txt sidecar for SUITE_OUT/LARGE_OUT
# (run_speedup_map.py writes none for SPEEDUP_OUT); copy those alongside their
# CSVs so the published results carry their provenance record.
for f in "$SUITE_OUT" "$LARGE_OUT" "$SPEEDUP_OUT" \
         "${SUITE_OUT}.provenance.txt" "${LARGE_OUT}.provenance.txt"; do
  if [[ -s "$f" ]]; then
    cp -f "$f" "$DEST_DIR/"
  fi
done

# ALL_DONE means the *entire* chain (all stages, across however many
# invocations it took) finished successfully -- so gate it on every stage's
# own DONE sentinel actually being present, rather than touching it
# unconditionally on every invocation. Without this, a run limited by
# STAGES=, or one where an earlier stage failed (run_stage never aborts the
# chain -- see its own comment), would still leave ALL_DONE behind, which
# is indistinguishable from a genuine full success to anything (a human, a
# cron job) that only checks for that file's existence.
ALL_STAGE_NAMES=(smoke suite large b1 b2 b4 b5 rep2 seed2)
missing_stages=()
for s in "${ALL_STAGE_NAMES[@]}"; do
  [[ -f "$WORK_DIR/${s}_DONE" ]] || missing_stages+=("$s")
done
if [[ ${#missing_stages[@]} -eq 0 ]]; then
  touch "$WORK_DIR/ALL_DONE"
  echo "[run_all] $(date -u +%FT%TZ) ALL_DONE" | tee -a "$LOG_DIR/run_all.log"
else
  echo "[run_all] $(date -u +%FT%TZ) not all stages complete yet " \
       "(missing: ${missing_stages[*]}); ALL_DONE not touched" \
       | tee -a "$LOG_DIR/run_all.log"
fi
