# Sourced by KDD28 run scripts ($PARAMS = params.py). run_cell <profile> <log> <cmd...>: pre-flight per profile, run, check.
# Drift (effective-params != profile): exit 2, or ALLOW_DRIFT=1 warns and sets DRIFT=!drift. LOG_APPEND=1 appends.
# DRIFT_EXPECT="trees=300 ..." = knobs the script takes from its own args (replace the profile's expectation).
DRIFT=""; _PREFLIGHTED=" "
drift_check() { # <profile> <log>
  local out rc=0; out=$(python3 "$PARAMS" check-effective "$1" "$2" ${DRIFT_EXPECT:-}) || rc=$?; DRIFT=""
  [ $rc = 0 ] && return 0
  sed "s#^#DRIFT $2: #" <<< "$out" >&2
  if [ "${ALLOW_DRIFT:-0}" = 1 ]; then echo "WARN $2: drift from profile $1, continuing (ALLOW_DRIFT=1), row stamped $1!drift" >&2; DRIFT='!drift'; return 0; fi
  echo "STOP $2: drift from profile $1 (ALLOW_DRIFT=1 to continue)" >&2; exit 2
}
preflight() { # <profile> <log> <cmd...>
  local p=$1 l=$2 rc=0; shift 2
  "$@" --print_params_only > "$l" 2>&1 || rc=$?
  [ $rc = 0 ] || { echo "STOP pre-flight $l: rc=$rc" >&2; exit 2; }
  drift_check "$p" "$l"; _PREFLIGHTED+="$p "
}
run_cell() { # <profile> <log> <cmd...>; returns the cell's rc
  local p=$1 l=$2 rc=0; shift 2
  [[ "$_PREFLIGHTED" == *" $p "* ]] || preflight "$p" "$l.preflight" "$@"
  if [ "${LOG_APPEND:-0}" = 1 ]; then "$@" >> "$l" 2>&1 || rc=$?; else "$@" > "$l" 2>&1 || rc=$?; fi
  drift_check "$p" "$l"; return $rc
}
