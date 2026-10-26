#!/usr/bin/env bash
set -uo pipefail

# Re-run the SPO-GBT half of the split-finder accuracy study (Table 4,
# tab:hist-vs-exact-summary) with P = F projections per node.
#
# Why: the 2026-08-11 study (run_hve_m7i_all.sh) passed no projection flags, and
# the harness default then was num_projections_exponent = 0.5 for both learners,
# so every accuracy_hve_gbt_* CSV and the GBT rows of accuracy_hve_physics.csv
# measured P = ceil(sqrt(F)). The paper's GBT arms use P = F (directive
# 2026-10-24; Table 9 *_fullproj arms). The RF half (P = sqrt(F)) is unchanged.
# The sqrt(F) GBT files were moved to benchmarks/results/accuracy/sqrtF_gbt_2026-08/.
#
# Same protocol as the original chain, GBT only, Table-4 inputs first:
#   A  CC18 {exact,rand,dyn}, seed 1              (accuracy.sh sweeps)
#   B  HIGGS + SUSY held-out, {exact,rand,dyn}    -> accuracy_hve_physics_gbt.csv
#   E  CC18 {exact,rand,dyn}, seeds 2-3           (seed-noise yardstick)
#   C  bin-count ablation, Random @ 16/32/128/256 bins, seed 1
#   D  scalar-binner build, Random 64 bins, seed 1 (bit-identity check)
#   S  statistics -> hist_vs_exact_accuracy/ (same two views as phase4)
# Resumable: finished CSVs are skipped. 'set -uo pipefail' without -e: one
# failed sweep reports and the chain continues.
#
# Non-default flags, same as the Table 9 *_fullproj arms: --num_projections_exponent 1
# (explicit, so the result does not depend on the harness default) and
# --max_num_projections 2147483647 (Bioresponse F=1776 and Internet-Advertisements
# F=1558 exceed the default cap of 1000). Everything else is the library default.

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
RESULTS=benchmarks/results/accuracy
LOG=$RESULTS/hve_gbt_pf.log
DONE_MARK=/home/ubuntu/spo_vs_gbt/hve_gbt_pf_DONE
BINARY=./bazel-bin/examples/train_oblique_forest
PF='--num_projections_exponent 1 --max_num_projections 2147483647'
ENS='--ensemble_method Boosting'

log() { echo "[hve-gbt-pf] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

split_args() {
  case "$1" in
    exact) echo '--numerical_split_type "Exact"' ;;
    rand)  echo '--numerical_split_type "Random" --histogram_num_bins=64' ;;
    dyn)   echo '--numerical_split_type "Dynamic Random Histogram" --histogram_num_bins=64 --dynamic_split_threshold=250' ;;
  esac
}

# $1 = accuracy.sh suffix, $2 = EXTRA_TRAIN_ARGS, $3 = EXTRA_BAZEL_CONFIGS (optional)
sweep() {
  if [[ -f "$RESULTS/accuracy_$1.csv" ]]; then log "SKIP $1 (CSV exists)"; return 0; fi
  log "RUN $1"
  if EXTRA_BAZEL_CONFIGS="${3:-}" EXTRA_TRAIN_ARGS="$2" bash benchmarks/src/accuracy.sh "$1" >>"$LOG" 2>&1; then
    log "OK $1"
  else
    log "FAILED $1 (see $LOG)"
  fi
}

# Default (vectorized) binary, same flags accuracy.sh uses.
build_default() {
  bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    bazel build -c opt --cxxopt="-O3" --cxxopt="-march=native" --enable_platform_specific_config \
      --repo_env=CC=icx --repo_env=CXX=icpx //examples:train_oblique_forest' >>"$LOG" 2>&1 \
    && log "default binary built" || log "FAILED default build"
}

phaseA() { for arm in exact rand dyn; do sweep "hve_gbt_${arm}_s1" "$(split_args $arm) $ENS $PF --seed=1"; done; }

phaseB() {
  local csv=$RESULTS/accuracy_hve_physics_gbt.csv logf=$RESULTS/accuracy_hve_physics_gbt.log
  if [[ -f "$csv" ]]; then log "SKIP physics_gbt (CSV exists)"; return 0; fi
  build_default
  : >"$logf"
  local ds train test arm cmd
  for ds in HIGGS SUSY; do
    case "$ds" in
      HIGGS) train=benchmarks/data/HIGGS_train_10500k.csv; test=benchmarks/data/HIGGS_test_500k.csv ;;
      SUSY)  train=benchmarks/data/SUSY_train_4500k.csv;   test=benchmarks/data/SUSY_test_500k.csv ;;
    esac
    for arm in exact rand dyn; do
      log "RUN physics $ds gbt $arm"
      cmd="$BINARY --input_mode csv --train_csv \"$train\" --test_csv \"$test\" --label_col class --num_trees=300 $ENS $(split_args $arm) $PF --seed=1"
      echo "$cmd" | tee -a "$logf" >/dev/null
      echo "----- Run 1/1 (fold=0) -----" >>"$logf"
      bash -c "$cmd" >>"$logf" 2>&1 || log "WARNING: $ds gbt $arm exited nonzero"
    done
  done
  python3 benchmarks/src/utils/parse_log_to_csv.py "$logf" "$csv" accuracy \
    && log "physics_gbt CSV: $csv (log kept: $logf)" || log "FAILED physics_gbt parse"
}

phaseE() { for seed in 2 3; do for arm in exact rand dyn; do sweep "hve_gbt_${arm}_s${seed}" "$(split_args $arm) $ENS $PF --seed=$seed"; done; done; }

phaseC() { for bins in 16 32 128 256; do sweep "hve_gbt_rand_b${bins}_s1" "--numerical_split_type \"Random\" --histogram_num_bins=$bins $ENS $PF --seed=1"; done; }

phaseD() {
  sweep "hve_gbt_rand_scalar_s1" "$(split_args rand) $ENS $PF --seed=1" "--config=disable_std_upper_bound_vectorization"
  local f_vec=$RESULTS/accuracy_hve_gbt_rand_s1.csv f_sca=$RESULTS/accuracy_hve_gbt_rand_scalar_s1.csv
  if [[ -f "$f_vec" && -f "$f_sca" ]]; then
    if diff <(grep -v '^[^,]*:' "$f_vec" | grep -v '^====') <(grep -v '^[^,]*:' "$f_sca" | grep -v '^====') >/dev/null; then
      log "BIT-IDENTICAL: gbt scalar == vectorized"
    else
      log "MISMATCH: gbt scalar != vectorized (expected under icx fast-math, see README)"
    fi
  fi
  build_default
}

phaseS() {
  python3 benchmarks/src/utils/accuracy_stats.py --out_dir "$RESULTS/hist_vs_exact_accuracy" \
    "$RESULTS"/accuracy_hve_*.csv >>"$LOG" 2>&1 && log "stats: all seeds OK" || log "FAILED stats (all seeds)"
  python3 benchmarks/src/utils/accuracy_stats.py --out_dir "$RESULTS/hist_vs_exact_accuracy/seed1_only" \
    "$RESULTS"/accuracy_hve_*_s1.csv "$RESULTS"/accuracy_hve_*_s1_auc.csv \
    "$RESULTS"/accuracy_hve_*_s1_logloss.csv "$RESULTS"/accuracy_hve_physics*.csv >>"$LOG" 2>&1 \
    && log "stats: seed1_only OK" || log "FAILED stats (seed1_only)"
}

log "start (P = F SPO-GBT re-run)"
for ph in phaseA phaseB phaseE phaseC phaseD phaseS; do
  log "===== $ph start"
  if $ph; then log "===== $ph OK"; else log "===== $ph FAILED (continuing)"; fi
done
log "all done"
touch "$DONE_MARK"
