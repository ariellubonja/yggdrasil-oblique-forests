#!/bin/bash
# Regression test: every KDD28 run script's argv (stubbed binaries) == golden/<case>.argv. Usage: [--update] [work_dir]
# --update rewrites the goldens. Env: BASE=<git ref> diffs BASE vs working tree, CASES = subset,
# REAL=1 runs bazel-bin's binaries with --print_params_only and reports drift per case.
# Every case must mirror a recorded profile arm of kdd28_params.json: REAL=1 has to print NODRIFT for all of them.
set -u
UPDATE=0; [ "${1:-}" = --update ] && { UPDATE=1; shift; }
HERE=$(cd "$(dirname "$0")" && pwd)
python3 "$HERE/params.py" selftest > /dev/null || { python3 "$HERE/params.py" selftest | grep FAIL; echo "params.py selftest FAILED"; exit 1; }
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
GOLD=$HERE/golden
W=${1:-$(mktemp -d)}; mkdir -p "$W"; W=$(cd "$W" && pwd)
SRC_DIRS="benchmarks/src/distributed_spo_gbt benchmarks/src/cc18_aa_spo_gbt benchmarks/src/cc18_gbt_feature_study"
if [ -n "${BASE:-}" ]; then SIDES="old new"; echo "argv equivalence: BASE=$(git -C "$REPO" rev-parse --short "$BASE") vs working tree, sandbox $W"
else SIDES=new; echo "argv check: working tree vs $GOLD$([ $UPDATE = 1 ] && echo ' (UPDATE)'), sandbox $W"; fi
# Stubs print a fixed effective-params line, so the scripts' drift checks run with ALLOW_DRIFT=1 (warn only).
if [ "${REAL:-0}" = 1 ]; then unset ALLOW_DRIFT; SIDES=new; else export ALLOW_DRIFT=1; fi
EFF_O='echo "I0 effective-params: num_trees=1 max_depth=1 min_examples=1 num_candidate_attributes=1 split=EXACT spo_exponent=1 spo_max=1 spo_density=1 spo_min=1 validation_set_ratio=0 early_stopping=NONE seed=1"'
EFF_D='echo "I0 effective-params: num_trees=1 max_depth=1 min_examples=1 num_candidate_attributes=1 split=EXACT spo_exponent=1 spo_max=1 spo_density=1 spo_min=1 seed=1 split_budget=false local_projection_count=true column_replication=0"'
[ "${REAL:-0}" = 1 ] && { EFF_O="$REPO/bazel-bin/examples/train_oblique_forest \"\$@\" --print_params_only 2>&1 | grep effective-params || exit 1"
  EFF_D="$REPO/bazel-bin/examples/train_distributed_gbt \"\$@\" --print_params_only 2>&1 | grep effective-params || exit 1"; }

# ---- stubs -------------------------------------------------------------------------------------------------
STUB=$W/stub; rm -rf "$STUB"; mkdir -p "$STUB"
# One file per call (concurrent launchers); collected into $ARGV_LOG after each case.
rec='l=$(basename "$0"); for a in "$@"; do l+=" $(printf %q "$a")"; done; echo "$l" > "$(mktemp "$ARGV_LOG.part.XXXXXX")"'
cat > "$STUB/train_oblique_forest" <<EOF
#!/bin/bash
$rec
$EFF_O
echo "I0 Test-set evaluation (1 rows): test-accuracy:0.5 test-auc:0.5 test-logloss:0.7"
mkdir -p "\$(sed -n 's/.*--model_out_dir \([^ ]*\).*/\1/p' <<< "\$*")" 2>/dev/null; exit 0
EOF
cat > "$STUB/train_distributed_gbt" <<EOF
#!/bin/bash
$rec
$EFF_D
printf 'Accuracy: 0.5\n  auc: 0.5\nLogLoss: 0.7\n'; exit 0
EOF
cat > "$STUB/python" <<EOF
#!/bin/bash
$rec
EOF
cat > "$STUB/gcloud" <<EOF
#!/bin/bash
$rec
case "\$*" in
  *"auth print-access-token"*) echo tok ;;
  *"artifacts docker tags list"*) echo "x/tags/47fe64fa-bdad5080" ;;
esac
exit 0
EOF
cat > "$STUB/dump_forest_nodes" <<EOF
#!/bin/bash
$rec
while [ \$# -gt 0 ]; do [ "\$1" = --output ] && echo "tree,attribute" > "\$2"; shift; done
EOF
printf '#!/bin/sh\necho %s\n' "'{\"dimensionsInfos\":[{\"dimensions\":{\"region\":\"us-east1\"},\"details\":{\"value\":\"96\"}}]}'" > "$STUB/curl"
printf '#!/bin/sh\nexit 0\n' > "$STUB/sleep"
printf '#!/bin/sh\nexit 1\n' > "$STUB/pgrep"
printf '#!/bin/sh\nexec "$@"\n' > "$STUB/setsid"
chmod +x "$STUB"/*

# ---- fake inputs (paths only; stubs read nothing) ----------------------------------------------------------
D=$W/data; rm -rf "$D"; mkdir -p "$D/folds/dsA" "$D/folds/dsB" "$D/cc18/task_1_aa" "$D/cc18/task_2_bb" "$D/cc18aa/data/dsA" \
  "$D/trunk" "$W/home/vertex_venv/bin" "$W/home/dgbt_work"
for f in dsA dsB; do echo '{"indices": [[0], [1]], "label_col": "label"}' > "$D/folds/$f/folds.json"; done
echo '{"label_col": "label"}' > "$D/cc18aa/data/dsA/meta.json"
for t in task_1_aa task_2_bb; do : > "$D/cc18/$t/train.csv"; : > "$D/cc18/$t/test.csv"; done
cp "$STUB/python" "$W/home/vertex_venv/bin/python"
FH=$W/home/yggdrasil-oblique-forests; mkdir -p "$FH/bazel-bin/examples" "$FH/benchmarks/data/cc18_binary_csv/task_7_cc"
cp "$STUB/train_oblique_forest" "$STUB/dump_forest_nodes" "$FH/bazel-bin/examples/"
echo '{"entries": [{"status": "ok", "task_id": 7, "folder_name": "cc", "label_column": "label"}]}' > "$FH/benchmarks/data/cc18_manifest.json"

# ---- sandbox tree: BASE or working copy at the same path, a git repo whose objects are the real repo's -------
T=$W/tree
make_tree() { # old|new
  rm -rf "$T"; mkdir -p "$T"
  if [ "$1" = old ]; then git -C "$REPO" archive "$BASE" $SRC_DIRS | tar -x -C "$T"
  else for d in $SRC_DIRS; do mkdir -p "$T/$(dirname $d)"; cp -R "$REPO/$d" "$T/$d"; done; fi
  rm -rf "$T/benchmarks/src/distributed_spo_gbt/params/pystub"; cp -R "$HERE/pystub" "$T/pystub"
  git init -q "$T" && echo "$REPO/.git/objects" > "$T/.git/objects/info/alternates"
  git -C "$T" -c user.name=x -c user.email=x commit -q --allow-empty -m sandbox
  mkdir -p "$T/bazel-bin/examples" && cp "$STUB/train_distributed_gbt" "$T/bazel-bin/examples/"
  local v=$T/benchmarks/results/runtime/distributed_spo_gbt/barriers/vertex  # run_pf_followups waits for 24 cells
  for i in $(seq 1 24); do mkdir -p "$v/pfgrid20261025_x/c$i" && echo '{}' > "$v/pfgrid20261025_x/c$i/eval.json"; done
  for s in run_dgbt_suite.sh; do  # same path rewrite in both trees: the script hardcodes the m7i paths
    sed -e "s#^REPO=/home/ubuntu/yggdrasil-oblique-forests#REPO=$T/fakerepo#" -e "s#^FOLDS=/home/ubuntu/spo_vs_gbt/folds#FOLDS=$D/folds#" \
      "$T/benchmarks/src/distributed_spo_gbt/$s" > "$T/benchmarks/src/distributed_spo_gbt/argvcheck_$s"
  done
  mkdir -p "$T/fakerepo/bazel-bin/examples" && cp "$STUB/train_distributed_gbt" "$T/fakerepo/bazel-bin/examples/"
}

IMG=us-east1-docker.pkg.dev/ydf-distributed-spo-gbt/ydf-repo/train-ydf
DS=benchmarks/src/distributed_spo_gbt
CASES=${CASES:-"table8_local table8_fixed dgbt_suite dgbt_spo dgbt_aa barrier_spo barrier_aa pf_grid pf_followups vertex_grid_fixed
  single_cc18 seed_single seed_w1 vertex_fixed colrep_default colrep_table3 replication_sweep barriers_per_worker barriers_split
  barriers_split_budget  cc18_aa_spo cc18_exact300 cc18_gbt train_cells train_cells_nosplit train_cells_table2 train_spo train_aa"}
run_case() { # case -> runs it inside $T with argv captured in $ARGV_LOG
  local O=$W/out/$1; rm -rf "$O"; mkdir -p "$O"
  case $1 in
    table8_local) mkdir -p "$O/bin"; cp "$STUB"/train_* "$O/bin/"; ln -s "$D/folds" "$O/folds"; NCPU=8 bash $DS/run_table8_local.sh "$O" ;;
    table8_fixed) NCPU=16 bash $DS/fixed_reruns/run_table8_fixed.sh "$O" "$STUB/train_distributed_gbt" "$D/folds" ;;
    dgbt_suite) RESULTS=$O/suite.csv WORK=$O/work bash $DS/argvcheck_run_dgbt_suite.sh ;;
    dgbt_spo) BIN=$STUB/train_distributed_gbt OUT=$O WORK=$O/work GITSHA=x bash $DS/run_dgbt.sh HIGGS 4 SPARSE_OBLIQUE 50 --seed 123456 ;;
    dgbt_aa) BIN=$STUB/train_distributed_gbt OUT=$O WORK=$O/work GITSHA=x bash $DS/run_dgbt.sh SUSY 1 AXIS_ALIGNED 300 ;;
    barrier_spo) TRUNK_DIR=$D/trunk OUT=$O WORK=$O/work bash $DS/run_barrier_cell.sh SPARSE_OBLIQUE ;;
    barrier_aa) TRUNK_DIR=$D/trunk OUT=$O WORK=$O/work bash $DS/run_barrier_cell.sh AXIS_ALIGNED 10 2 1 ;;
    pf_grid) bash $DS/run_pf_grid.sh ;;
    pf_followups) bash $DS/run_pf_followups.sh ;;
    vertex_grid_fixed) bash $DS/runtime_reruns/run_vertex_grid_fixed.sh ;;
    single_cc18) bash $DS/fixed_reruns/run_single_machine_cc18.sh "$STUB/train_oblique_forest" "$D/cc18" "$O" ;;
    seed_single) FOLDS=$D/folds SINGLE=$STUB/train_oblique_forest bash $DS/fixed_reruns/run_seed_sweep.sh single 3 4 "$O" ;;
    seed_w1) FOLDS=$D/folds DIST=$STUB/train_distributed_gbt bash $DS/fixed_reruns/run_seed_sweep.sh w1 2 4 "$O" ;;
    vertex_fixed) PY=$STUB/python bash $DS/fixed_reruns/run_vertex_fixed.sh "$O" build table2 table2_split table3 ;;
    colrep_default) PYTHONPATH=$T/pystub python3 $DS/column_replication/run_vertex_colrep.py --image $IMG:x --run_id R --out_root "$O" ;;
    colrep_table3) PYTHONPATH=$T/pystub python3 $DS/column_replication/run_vertex_colrep.py --image $IMG:x --run_id R --out_root "$O" \
                     --seeds 1 --levels 0,ship --workers 4 --num_trees 300 --max_depth 6 --extra_args=--local_projection_count ;;
    replication_sweep) python3 $DS/column_replication/run_replication_sweep.py --bin "$STUB/train_distributed_gbt" --data "$D/cc18" --out "$O" --jobs 1 ;;
    barriers_per_worker) PYTHONPATH=$T/pystub python3 $DS/run_vertex_barriers.py --dataset susy --workers 1,2,16 --arms aa,spo --trees 10 \
                     --p_total 18 --machine n2-highmem-4 --image $IMG:fullfix-10509539 --run_id R --budget per_worker ;;
    barriers_split) PYTHONPATH=$T/pystub python3 $DS/run_vertex_barriers.py --dataset epsilon --workers 1,4 --arms spo --trees 50 \
                     --p_total 2000 --image $IMG:47fe64fa-bdad5080 --run_id R --budget per_worker --fresh_cache \
                     --extra_args=--max_unique_values_for_discretized_numerical=1 ;;
    barriers_split_budget) PYTHONPATH=$T/pystub python3 $DS/run_vertex_barriers.py --dataset susy --workers 1,4 --arms aa,spo --trees 10 \
                     --machine n2-highmem-4 --image $IMG:1311deba-x --run_id R ;;  # --budget split (default) on a split-budget image
    cc18_aa_spo) FOLDS="0 1" bash benchmarks/src/cc18_aa_spo_gbt/run_cc18_aa_spo_gbt.sh "$D/cc18aa" "$STUB/train_oblique_forest"; rm -rf "$D/cc18aa/logs" "$D/cc18aa/models" "$D/cc18aa/results.csv" ;;
    cc18_exact300) ARMS=spo_harness_exact300 bash benchmarks/src/cc18_aa_spo_gbt/run_cc18_aa_spo_gbt.sh "$D/cc18aa" "$STUB/train_oblique_forest"; rm -rf "$D/cc18aa/logs" "$D/cc18aa/models" "$D/cc18aa/results.csv" ;;
    cc18_gbt) python3 benchmarks/src/cc18_gbt_feature_study/run_cc18_gbt.py --jobs 1; rm -rf "$W/home/featstudy" ;;
    train_cells) (cd $DS/vertex_build && PYTHONPATH=$T/pystub python3 train_cells.py --data gs://b/cc18 --out gs://b/runs/x/s2 --seed 2 --levels 0,1) ;;
    train_cells_nosplit) (cd $DS/vertex_build && PYTHONPATH=$T/pystub python3 train_cells.py --data gs://b/cc18 --out gs://b/runs/x/s1 --seed 1 \
                     --levels 0.5 --no_split_budget --num_trees 10 --sparse_oblique_num_projections_exponent=1 \
                     --sparse_oblique_max_num_projections=2147483647 --sparse_oblique_projection_density_factor=1.5) ;;
    train_cells_table2) (cd $DS/vertex_build && PYTHONPATH=$T/pystub python3 train_cells.py --data gs://b/cc18 --out gs://b/runs/x/s3 --seed 3 \
                     --levels 0 --num_trees 10 --sparse_oblique_num_projections_exponent=1 --sparse_oblique_max_num_projections=2147483647 \
                     --sparse_oblique_projection_density_factor=1.5) ;;
    train_spo) (cd $DS/vertex_build && PYTHONPATH=$T/pystub python3 train.py --train_ds csv:gs://b/t.csv --test_ds csv:gs://b/v.csv \
                     --model gs://b/m --label class --distributed --split_axis SPARSE_OBLIQUE --num_trees 10 \
                     --sparse_oblique_num_projections_exponent=0.990254 --sparse_oblique_projection_density_factor=1.5 \
                     --meta='{"arm": "spo", "cols": 18, "workers": 1}' --profile vertex_grid_pf.spo_split) ;;
    train_aa) (cd $DS/vertex_build && PYTHONPATH=$T/pystub python3 train.py --train_ds csv:gs://b/t.csv --model gs://b/m --label class \
                     --distributed --split_axis AXIS_ALIGNED --num_trees 10 --max_unique_values_for_discretized_numerical=1 \
                     --meta='{"arm": "aa", "cols": 18, "workers": 1, "profile": "vertex_grid_pf.aa_split"}') ;;
    *) echo "unknown case $1"; return 1 ;;
  esac
}
# Lines that are not argv: per-run timestamps and the eval.json payloads (new adds the stamp).
# Pre-flight calls (--print_params_only) are not cells: counted, not compared. $W = <W> (goldens are path-free).
normalize() { sed -E -e "s#$W#<W>#g" -e 's/[0-9]{8}T[0-9]{4}Z/<RUN>/g' -e 's/cache_cc18_fixed_[0-9]{8}/cache_cc18_fixed_<DATE>/g' "$1" \
  | grep -v '"kind": "gcs_pipe"\|"kind": "gcs_upload"\|--print_params_only' | LC_ALL=C sort; }

for side in $SIDES; do
  make_tree $side; rm -rf "$W/argv/$side"; mkdir -p "$W/argv/$side" "$W/tmp"
  for c in $CASES; do
    (cd "$T" && PATH=$STUB:$PATH HOME=$W/home ARGV_LOG=$W/argv/$side/$c.log ARGV_SANDBOX_TMP=$W/tmp \
       run_case $c) > "$W/log_${side}_$c.txt" 2>&1 < /dev/null
    cat "$W/argv/$side/$c.log".part.* >> "$W/argv/$side/$c.log" 2>/dev/null; rm -f "$W/argv/$side/$c.log".part.*
  done
done
fail=0
if [ "${REAL:-0}" = 1 ]; then  # drift report only: a stopped script captures fewer argv
  for c in $CASES; do l=$W/log_new_$c.txt
    if grep -q "^STOP\|^DRIFT" "$l"; then echo "DRIFT $c:"; grep "^STOP\|^DRIFT" "$l" | sort -u | head -8 | sed "s#$W#<W>#g; s#^#  #"; fail=1
    elif grep -q "OK kwargs" "$l"; then echo "NODRIFT $c (container: $(grep -m1 "OK kwargs" "$l"))"
    elif grep -q "WARN profile check" "$l"; then echo "NOPROFILE $c: $(grep -m1 "WARN profile" "$l")"; fail=1
    else echo "NODRIFT $c ($(grep -c -- --print_params_only "$W/argv/new/$c.log" 2>/dev/null) pre-flight)"; fi
  done; exit $fail
fi
[ $UPDATE = 1 ] && mkdir -p "$GOLD"
for c in $CASES; do
  n=$W/argv/new/$c.log; o=${BASE:+$W/argv/old/$c.log}; o=${o:-$GOLD/$c.argv}
  pf=$(grep -c -- --print_params_only "$n" 2>/dev/null); pf=${pf:-0}
  if [ $UPDATE = 1 ]; then
    if [ ! -s "$n" ]; then echo "NOARGV $c (captured nothing: $W/log_new_$c.txt)"; fail=1; continue; fi
    normalize "$n" > "$GOLD/$c.argv"; echo "UPDATED $c ($(wc -l < "$GOLD/$c.argv" | tr -d ' ') argv)"; continue
  fi
  if [ ! -s "$o" ]; then echo "NOARGV $c (no reference: $o)"; fail=1; continue; fi
  if diff <(if [ -n "${BASE:-}" ]; then normalize "$o"; else cat "$o"; fi) <(normalize "$n") > "$W/diff_$c.txt"; then
    echo "PASS $c ($(normalize "$n" | wc -l | tr -d ' ') argv, $pf pre-flight)"
  else
    echo "FAIL $c: $W/diff_$c.txt"; head -6 "$W/diff_$c.txt" | cut -c1-240; fail=1
  fi
done
# The container scripts cannot read params/: train_cells.py defaults = kdd28_ydf.dist_spo (SPO budget kwargs omitted =
# proto defaults), and its table2 flags reproduce profile table2 (case train_cells_table2).
[ -s "$W/argv/new/train_cells.log" ] && { PYTHONPATH=$HERE python3 - "$W/argv/new/train_cells.log" "$W/argv/new/train_cells_table2.log" <<'PY' || fail=1; }
import json, sys
import params
SPO = ("sparse_oblique_num_projections_exponent", "sparse_oblique_projection_density_factor", "sparse_oblique_max_num_projections")
def kw_of(path):
    return [json.loads(l)["kwargs"] for l in open(path) if '"DistributedGradientBoostedTreesLearner"' in l][0]
bad = {}
for path, prof, spo in ((sys.argv[1], "kdd28_ydf.dist_spo", None), (sys.argv[2], "table2", ("spo_exponent", "spo_density", "spo_max_projections"))):
    p, kw = params.load_profile(prof), kw_of(path)
    want = {"num_trees": p["trees"], "max_depth": p["max_depth"], "sparse_oblique_weights": "BINARY",
            "max_unique_values_for_discretized_numerical": p["max_unique_discretized"]}
    want |= dict(zip(SPO, (p[f] for f in spo))) if spo else dict.fromkeys(SPO)  # None = omitted = proto default
    diff = {k: (kw.get(k), v) for k, v in want.items() if kw.get(k) != v}
    print(f"PASS train_cells kwargs == profile {prof}" if not diff else f"FAIL train_cells kwargs != profile {prof}: {diff}")
    bad = bad or diff
# The container copies of the profiles (PROFILES in train_cells.py and train.py) must equal kdd28_params.json.
import ast, os
vb = os.path.join(os.path.dirname(params.__file__), "..", "vertex_build")
for fn in ("train_cells.py", "train.py"):
    tree = ast.parse(open(os.path.join(vb, fn)).read())
    names = {t.id: n.value for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if hasattr(t, "id")}
    env = {"__builtins__": {}, "dict": dict}
    if "_SPO" in names:  # train.py builds PROFILES from _SPO
        env["_SPO"] = eval(compile(ast.Expression(names["_SPO"]), fn, "eval"), env)  # noqa: S307 (repo file)
    copy = eval(compile(ast.Expression(names["PROFILES"]), fn, "eval"), env)  # noqa: S307 (repo file)
    for prof, fields in copy.items():
        p = params.load_profile(prof)
        diff = {k: (v, p.get(k)) for k, v in fields.items() if p.get(k) != v}
        print(f"PASS {fn} PROFILES[{prof}] == profile" if not diff else f"FAIL {fn} PROFILES[{prof}] != profile: {diff}")
        bad = bad or diff
sys.exit(1 if bad else 0)
PY
[ $fail = 0 ] && echo "ALL PASS" || echo "SOME CASES FAILED"
exit $fail
