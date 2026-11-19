#!/usr/bin/env bash
# Same-seed model identity: models from <base_ref> vs the working tree, compared by compare_models.sh.
# Usage: model_identity_check.sh [base_ref=HEAD] [out_dir]   exit 0 identical, 1 differ, 2 setup error
# Base binaries are cached by commit under $YDF_IDENTITY_CACHE (default ~/.cache/ydf_model_identity).
set -uo pipefail
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
BASE_REF=${1:-HEAD}
BASE=$(git -C "$REPO" rev-parse --verify "$BASE_REF^{commit}") || { echo "bad base ref $BASE_REF"; exit 2; }
OUT=${2:-$REPO/ariel_notes/results/$(date -u +%Y-%m-%d)_model_identity/${BASE:0:8}_vs_worktree_$(date -u +%H%M%S)}
CACHE=${YDF_IDENTITY_CACHE:-$HOME/.cache/ydf_model_identity}
TARGETS="//examples:train_oblique_forest //examples:train_distributed_gbt"
BFLAGS="-c opt"; [ "$(uname)" = Darwin ] && BFLAGS+=" --macos_minimum_os=12.0"
mkdir -p "$OUT" "$CACHE"; exec > >(tee "$OUT/run.log") 2>&1
echo "base $BASE_REF = $BASE | head = worktree of $(git -C "$REPO" rev-parse HEAD) | out $OUT"

build() {  # build <src_dir> <bin_dir>
  (cd "$1" && bazel build $BFLAGS $TARGETS) || return 1
  mkdir -p "$2"; for t in train_oblique_forest train_distributed_gbt; do cp -f "$1/bazel-bin/examples/$t" "$2/"; done
}
# Base: one persistent detached worktree, re-checked-out per commit (keeps bazel incremental).
BB=$CACHE/bin_$BASE
if [ ! -x "$BB/train_distributed_gbt" ]; then
  WT=$CACHE/base_src
  [ -d "$WT" ] || git -C "$REPO" worktree add --detach "$WT" "$BASE" || exit 2
  git -C "$WT" checkout -q --detach "$BASE" || exit 2
  build "$WT" "$BB.tmp" && mv "$BB.tmp" "$BB" || { echo "BASE BUILD FAILED"; exit 2; }
fi
build "$REPO" "$OUT/bin_head" || { echo "HEAD BUILD FAILED"; exit 2; }
(cd "$REPO" && bazel build $BFLAGS @com_google_protobuf//:protoc) || { echo "PROTOC BUILD FAILED"; exit 2; }
PROTOC=$(cd "$REPO" && bazel cquery $BFLAGS --output=files @com_google_protobuf//:protoc 2>/dev/null |
  while read -r p; do [ -x "$REPO/$p" ] && { echo "$REPO/$p"; break; }; done)
[ -n "$PROTOC" ] || { echo "PROTOC NOT FOUND after build"; exit 2; }
PINC=$(cd "$REPO" && bazel info output_base 2>/dev/null)/external/protobuf~/src

# Small deterministic binary dataset: continuous, 0/1, 3-valued and small-int columns.
python3 - "$OUT/data" <<'EOF'
import os, random, sys
d = sys.argv[1]; os.makedirs(d, exist_ok=True); r = random.Random(0)
cols = [f"c{i}" for i in range(8)] + ["b0", "t0", "k0", "k1"]
for name, n in (("train", 3000), ("test", 1000)):
    with open(f"{d}/{name}.csv", "w") as f:
        f.write(",".join(cols + ["label"]) + "\n")
        for _ in range(n):
            x = [r.gauss(0, 1) for _ in range(8)] + [r.randint(0, 1), r.randint(0, 2), r.randint(0, 9), r.randint(0, 40)]
            s = x[0] - 0.7 * x[1] + 0.5 * x[2] * x[3] + 0.8 * x[8] - 0.3 * x[9] + 0.05 * x[10] + r.gauss(0, 0.7)
            f.write(",".join(f"{v:.6g}" for v in x) + f",{int(s > 0)}\n")
EOF

ARMS="rf_spo_hist rf_spo_exact gbt_spo_exact gbt_aa dgbt_spo_w1 dgbt_spo_w2 dgbt_aa_w2"
train() {  # train <bin_dir> <arm> <model_dir>
  local B=$1 A=$2 M=$3 D=$OUT/data
  local S="$B/train_oblique_forest --input_mode csv --train_csv $D/train.csv --test_csv $D/test.csv --label_col label --seed 7 --num_threads 4 --model_out_dir $M"
  local G="$B/train_distributed_gbt --train_csv $D/train.csv --test_csv $D/test.csv --label label --seed 7 --num_threads 2 --num_trees 20 --worker_logs=false --model_out_dir $M --work_dir $M.work"
  case $A in
    rf_spo_hist)   $S --ensemble_method Bagging --num_trees 8 --tree_depth 10 --feature_split_type Oblique ;;
    rf_spo_exact)  $S --ensemble_method Bagging --num_trees 8 --tree_depth 10 --feature_split_type Oblique --numerical_split_type Exact ;;
    gbt_spo_exact) $S --ensemble_method Boosting --num_trees 20 --tree_depth 6 --feature_split_type Oblique --numerical_split_type Exact ;;
    gbt_aa)        $S --ensemble_method Boosting --num_trees 20 --tree_depth 6 --feature_split_type "Axis Aligned" --numerical_split_type Exact ;;
    dgbt_spo_w1)   $G --num_workers 1 --split_axis SPARSE_OBLIQUE ;;
    dgbt_spo_w2)   $G --num_workers 2 --split_axis SPARSE_OBLIQUE ;;
    dgbt_aa_w2)    $G --num_workers 2 --split_axis AXIS_ALIGNED ;;
  esac
}
FAIL=0
for a in $ARMS; do
  for side in base head; do
    B=$BB; [ $side = head ] && B=$OUT/bin_head
    rm -rf "$OUT/$side/$a" "$OUT/$side/$a.work"; mkdir -p "$OUT/$side"
    train "$B" $a "$OUT/$side/$a" > "$OUT/$side/$a.log" 2>&1 || { echo "  $a: $side TRAINING FAILED (rc=$?), see $OUT/$side/$a.log"; FAIL=1; continue 2; }
  done
  if PROTOC=$PROTOC PROTOC_INCLUDE=$PINC "$REPO/benchmarks/src/compare_models.sh" "$OUT/base/$a" "$OUT/head/$a" > "$OUT/$a.compare.txt" 2>&1; then echo "  $a: identical"
  else echo "  $a: MODEL DIFFERS"; grep -v -E ' MATCH' "$OUT/$a.compare.txt" | sed 's/^/      /'; FAIL=1; fi
done
if [ $FAIL = 0 ]; then echo "MODEL IDENTITY: PASS (7 arms identical: node bytes + decoded protos minus uid/created_date)"; exit 0; fi
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo "!!! MODEL IDENTITY: FAIL - this change alters trained models      !!!"
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
exit 1
