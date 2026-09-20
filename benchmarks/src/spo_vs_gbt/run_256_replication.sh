#!/usr/bin/env bash
# Replication of the three 256-bin AVX-512 arms (spo_rf_rand256_vec, spo_rf_dyn256_vec,
# spo_gbt_dyn256_vec) on a fresh m7i box (2026-09-18), following REPLICABILITY.md §6:
# every recorded suite + large row of those arms is re-run through replicate_check.py
# (same run_suite.run_ydf path, argv asserted equal to the recorded `cmd`, --acc-tol 0).
# Timing is NOT comparable across instances (REPLICABILITY.md §7), so --time-tol is
# effectively off; train_s_rel_delta is still recorded per row.
# One replicate_check invocation per dataset so each writes its own CSV (the driver
# writes only at the end). Large tier: HIGGS/SUSY train splits are materialised from
# *_with_header.csv just before their rows and removed after (root volume is small);
# their sha256 is checked against MANIFEST/large_inputs.sha256 first.
set -uo pipefail
W=/home/ubuntu/spo_vs_gbt
REPO=/home/ubuntu/yggdrasil-oblique-forests
SD="$REPO/benchmarks/src/spo_vs_gbt"
PY=/home/ubuntu/gbt_venv/bin/python
DATA="$REPO/benchmarks/data"
ARMS="spo_rf_rand256_vec spo_rf_dyn256_vec spo_gbt_dyn256_vec"
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$W/results/replication/avx512_256_$TS"
mkdir -p "$OUT" "$W/logs/replication"
LOG="$OUT/run.log"
log() { echo "[run_256_replication] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
log "start out=$OUT binary=$(cat $W/bin/default.gitsha) host=$(hostname)"

STAGES="${STAGES:-suite,large}"
if [[ ",$STAGES," == *,suite,* ]]; then
  # suite datasets in the recorded CSV, with their recorded folds
  "$PY" - <<'PYEOF' > "$OUT/suite_rows.txt"
import csv
arms={"spo_rf_rand256_vec","spo_rf_dyn256_vec","spo_gbt_dyn256_vec"}
seen={}
for r in csv.DictReader(open("/home/ubuntu/spo_vs_gbt/results/suite_results.csv")):
    if r["method"] in arms and r["rep"]=="0":
        seen.setdefault(r["dataset"],set()).add(int(r["fold"]))
for ds in sorted(seen):
    print(ds, " ".join(str(f) for f in sorted(seen[ds])))
PYEOF
  while read -r ds folds; do
    rows=()
    for f in $folds; do for a in $ARMS; do rows+=(--rows "$ds,$f,$a"); done; done
    log "suite $ds (${#rows[@]} rows)"
    "$PY" "$SD/replicate_check.py" "${rows[@]}" --acc-tol 0 --time-tol 1e9 \
        --out "$OUT/suite_$ds.csv" --log-dir "$W/logs/replication" >> "$OUT/suite.log" 2>&1
    log "suite $ds exit $?"
  done < "$OUT/suite_rows.txt"
fi

if [[ ",$STAGES," == *,large,* ]]; then
  for ds in EPSILON SUSY HIGGS; do
    case $ds in
      HIGGS) head -n 10500001 "$DATA/HIGGS_with_header.csv" > "$DATA/HIGGS_train_10500k.csv"; tmp="$DATA/HIGGS_train_10500k.csv" ;;
      SUSY)  head -n 4500001 "$DATA/SUSY_with_header.csv" > "$DATA/SUSY_train_4500k.csv"; tmp="$DATA/SUSY_train_4500k.csv" ;;
      *) tmp="" ;;
    esac
    if [ -n "$tmp" ]; then
      grep "$(basename $tmp)" "$W/MANIFEST/large_inputs.sha256" | sha256sum -c 2>&1 | tee -a "$LOG"
      df -h / | tail -1 | tee -a "$LOG"
    fi
    rows=(); for a in $ARMS; do rows+=(--rows "$ds,0,$a"); done
    log "large $ds"
    "$PY" "$SD/replicate_check.py" "${rows[@]}" --acc-tol 0 --time-tol 1e9 --timeout 10800 \
        --out "$OUT/large_$ds.csv" --log-dir "$W/logs/replication" >> "$OUT/large.log" 2>&1
    log "large $ds exit $?"
    [ -n "$tmp" ] && rm -f "$tmp" && log "removed $tmp"
  done
fi

# merge
"$PY" - "$OUT" <<'PYEOF'
import csv, glob, os, sys
out=sys.argv[1]; rows=[]; keys=[]
for f in sorted(glob.glob(os.path.join(out,"suite_*.csv")))+sorted(glob.glob(os.path.join(out,"large_*.csv"))):
    for r in csv.DictReader(open(f)):
        rows.append(r)
        for k in r:
            if k not in keys: keys.append(k)
with open(os.path.join(out,"replication_avx512_256.csv"),"w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=keys); w.writeheader(); w.writerows(rows)
n=len(rows); p=sum(1 for r in rows if r["result"]=="PASS")
print(f"merged {n} rows, {p} PASS, {n-p} FAIL")
PYEOF
log "ALL DONE"
touch "$OUT/DONE"
