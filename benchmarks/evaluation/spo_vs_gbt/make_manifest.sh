#!/usr/bin/env bash
# Write the replicability manifest for the SPO-vs-GBT study (REPLICABILITY.md §3):
# sha256 + byte size of every input the recorded rows depend on, the harness
# binaries, the python environment, and the exact source state.
#
#   bash make_manifest.sh            # small inputs only (suite CSVs, fold CSVs, bins) ~1 min
#   bash make_manifest.sh --large    # also HIGGS/SUSY/EPSILON (~28 GB, ~5 min, single core)
#
# Output: /home/ubuntu/spo_vs_gbt/MANIFEST/{inputs.sha256,large_inputs.sha256,
#         pip_freeze.txt,git_state.txt,machine.txt}. Copy the directory into the
# repo (benchmarks/results/spo_vs_gbt/MANIFEST) when the study is published.
set -euo pipefail
WORK=/home/ubuntu/spo_vs_gbt
REPO=/home/ubuntu/yggdrasil-oblique-forests
OUT=$WORK/MANIFEST
mkdir -p "$OUT"
cd "$REPO"

{
  echo "date_utc: $(date -u +%FT%TZ)"
  echo "HEAD: $(git rev-parse HEAD)"
  echo "branch: $(git rev-parse --abbrev-ref HEAD)"
  echo "--- git status --short"; git status --short
  echo "--- git diff (tracked, excluding paper snapshot)"
  git diff -- . ':!benchmarks/results/overleaf-sept-3-26' ':!*.bib' ':!*.tex'
} > "$OUT/git_state.txt"

/home/ubuntu/gbt_venv/bin/python -m pip freeze > "$OUT/pip_freeze.txt" 2>/dev/null || true
/home/ubuntu/gbt_venv/bin/python -c 'import sys;print("python",sys.version)' >> "$OUT/pip_freeze.txt"

{
  echo "hostname: $(hostname)"; uname -a
  lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket|L3|NUMA node\(s\)"
  free -g | head -2
  echo "instance: $(curl -s -m 1 http://169.254.169.254/latest/meta-data/instance-type || echo unknown)"
  echo "icx: $( (source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; icx --version | head -1) )"
  echo "bazel: $(bazel --version 2>/dev/null || echo unknown)"
} > "$OUT/machine.txt" 2>&1

# --- small inputs: harness bins, suite source CSVs + train_nan, fold CSVs + folds.json
{
  for f in "$WORK"/bin/*; do sha256sum "$f"; done
  find "$REPO/benchmarks/data/tabarena_binary_csv" "$REPO/benchmarks/data/tabred_binary_csv" \
       -type f \( -name '*.csv' -o -name '*.json' \) | sort | xargs sha256sum
  find "$WORK/folds" -type f | sort | xargs sha256sum
  find "$REPO/benchmarks/evaluation/spo_vs_gbt" -maxdepth 1 -type f \
       \( -name '*.py' -o -name '*.json' -o -name '*.sh' \) | sort | xargs sha256sum
} > "$OUT/inputs.sha256"
echo "wrote $OUT/inputs.sha256 ($(wc -l < "$OUT/inputs.sha256") files)"

if [[ "${1:-}" == "--large" ]]; then
  sha256sum "$REPO/benchmarks/data/HIGGS_train_10500k.csv" "$REPO/benchmarks/data/HIGGS_test_500k.csv" \
            "$REPO/benchmarks/data/SUSY_train_4500k.csv" "$REPO/benchmarks/data/SUSY_test_500k.csv" \
            "$REPO/benchmarks/data/epsilon_normalized_train.csv" "$WORK/data/epsilon_test_100k.csv" \
    > "$OUT/large_inputs.sha256"
  echo "wrote $OUT/large_inputs.sha256"
fi
