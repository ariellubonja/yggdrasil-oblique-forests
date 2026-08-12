#!/usr/bin/env bash
set -uo pipefail

# Split-finder accuracy study, m7i edition — runs the WHOLE study from
# scratch in seed-1-first order:
#   phase0   CC18 main sweep, seed 1 only ({rf,gbt} x {exact,rand,dyn})
#   phase1   HIGGS + SUSY held-out (RF 240 trees, GBT 300; all 3 arms)
#   phase2   bin-count ablation (Random @ 16/32/128/256 bins, seed 1)
#   phase3   scalar-vs-vectorized bit-identity (CC18 rand arms, scalar build)
#   phase3b  CC18 main sweep, seeds 2-3 (seed-noise yardstick; LAST on purpose)
#   phase4   statistics -> benchmarks/results/hist_vs_exact_accuracy/
#
# Every arm is resumable: finished CSVs are skipped on rerun.
# 'set -uo pipefail' without -e: one failed phase reports and the chain
# continues.

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
RESULTS=benchmarks/results
BINARY=./bazel-bin/examples/train_oblique_forest

phase0() { SEEDS_OVERRIDE="1" bash benchmarks/evaluation/run_hist_vs_exact_accuracy.sh; }

phase1() {
  local logfile="$RESULTS/accuracy_hve_physics.log"
  local csvfile="$RESULTS/accuracy_hve_physics.csv"
  if [[ -f "$csvfile" ]]; then echo "[phase1] SKIP (csv exists)"; return 0; fi
  : > "$logfile"
  local ds train test learner arm ens trees split_args cmd
  for ds in HIGGS SUSY; do
    case "$ds" in
      HIGGS) train=benchmarks/data/HIGGS_train_10500k.csv
             test=benchmarks/data/HIGGS_test_500k.csv ;;
      SUSY)  train=benchmarks/data/SUSY_train_4500k.csv
             test=benchmarks/data/SUSY_test_500k.csv ;;
    esac
    for learner in rf gbt; do
      # RF 240 trees = the runtime section's config; a fully-grown 10.5M-row
      # tree costs ~1 GB so this needs a large-memory machine (fine on m7i
      # 384G; it OOM'd a 62G laptop). GBT is depth-6, tiny.
      ens=""; trees=240
      [[ "$learner" == gbt ]] && { ens='--ensemble_method Boosting'; trees=300; }
      for arm in exact rand dyn; do
        case "$arm" in
          exact) split_args='--numerical_split_type "Exact"' ;;
          rand)  split_args='--numerical_split_type "Random" --histogram_num_bins=64' ;;
          dyn)   split_args='--numerical_split_type "Dynamic Random Histogram" --histogram_num_bins=64 --dynamic_split_threshold=250' ;;
        esac
        cmd="$BINARY --input_mode csv --train_csv \"$train\" --test_csv \"$test\" --label_col class --num_trees=$trees $ens $split_args --seed=1"
        echo "$cmd" | tee -a "$logfile"
        echo "----- Run 1/1 (fold=0) -----" | tee -a "$logfile"
        bash -c "$cmd" >>"$logfile" 2>&1 \
          || echo "WARNING: $ds $learner $arm exited nonzero" | tee -a "$logfile"
      done
    done
  done
  python3 benchmarks/utils/parse_log_to_csv.py "$logfile" "$csvfile" accuracy \
    && echo "[phase1] CSV: $csvfile (log kept: $logfile)"
}

phase2() {
  local seed=1 bins learner ens suffix
  for bins in 16 32 128 256; do
    for learner in rf gbt; do
      ens=""
      [[ "$learner" == gbt ]] && ens="--ensemble_method Boosting"
      suffix="hve_${learner}_rand_b${bins}_s${seed}"
      if [[ -f "$RESULTS/accuracy_${suffix}.csv" ]]; then
        echo "[phase2] SKIP $suffix"; continue
      fi
      echo "[phase2] RUN $suffix"
      EXTRA_TRAIN_ARGS="--numerical_split_type \"Random\" --histogram_num_bins=$bins $ens --seed=$seed" \
        bash benchmarks/evaluation/accuracy.sh "$suffix"
    done
  done
}

phase3() {
  local seed=1 learner ens suffix
  for learner in rf gbt; do
    ens=""
    [[ "$learner" == gbt ]] && ens="--ensemble_method Boosting"
    suffix="hve_${learner}_rand_scalar_s${seed}"
    if [[ -f "$RESULTS/accuracy_${suffix}.csv" ]]; then
      echo "[phase3] SKIP $suffix"; continue
    fi
    echo "[phase3] RUN $suffix (scalar build)"
    EXTRA_BAZEL_CONFIGS="--config=disable_std_upper_bound_vectorization" \
    EXTRA_TRAIN_ARGS="--numerical_split_type \"Random\" --histogram_num_bins=64 $ens --seed=$seed" \
      bash benchmarks/evaluation/accuracy.sh "$suffix"
  done
  # Bit-identity: fold cells of scalar vs vectorized must match exactly
  # (provenance lines are stripped: they legitimately differ in configs).
  local ok=1 f_vec f_sca
  for learner in rf gbt; do
    f_vec="$RESULTS/accuracy_hve_${learner}_rand_s1.csv"
    f_sca="$RESULTS/accuracy_hve_${learner}_rand_scalar_s1.csv"
    if [[ -f "$f_vec" && -f "$f_sca" ]]; then
      if diff <(grep -v '^[^,]*:' "$f_vec" | grep -v '^====') \
              <(grep -v '^[^,]*:' "$f_sca" | grep -v '^====') >/dev/null; then
        echo "[phase3] BIT-IDENTICAL: $learner scalar == vectorized"
      else
        echo "[phase3] MISMATCH: $learner scalar != vectorized"; ok=0
      fi
    fi
  done
  # Restore the default (vectorized) binary for the remaining phases.
  bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    bazel build -c opt --cxxopt="-O3" --cxxopt="-march=native" //examples:train_oblique_forest' \
    >/dev/null 2>&1 && echo "[phase3] default binary restored"
  return $((1 - ok))
}

phase3b() { SEEDS_OVERRIDE="2 3" bash benchmarks/evaluation/run_hist_vs_exact_accuracy.sh; }

phase4() {
  python3 benchmarks/utils/accuracy_stats.py \
    --out_dir "$RESULTS/hist_vs_exact_accuracy" \
    "$RESULTS"/accuracy_hve_*.csv
  # Seed-1-only view (the primary analysis).
  python3 benchmarks/utils/accuracy_stats.py \
    --out_dir "$RESULTS/hist_vs_exact_accuracy/seed1_only" \
    "$RESULTS"/accuracy_hve_*_s1.csv "$RESULTS"/accuracy_hve_*_s1_auc.csv \
    "$RESULTS"/accuracy_hve_*_s1_logloss.csv "$RESULTS"/accuracy_hve_physics*.csv
}

for ph in phase0 phase1 phase2 phase3 phase3b phase4; do
  echo "[chain] ===== $ph start $(date -u +%FT%TZ)"
  if $ph; then echo "[chain] ===== $ph OK"; else echo "[chain] ===== $ph FAILED (continuing)"; fi
done
echo "[chain] all done $(date -u +%FT%TZ)"
