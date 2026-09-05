#!/usr/bin/env bash
# Event stream for a runtime.sh/accuracy.sh log, meant for the Monitor tool:
#   watch_runs.sh <logfile> [limit_seconds=7200]
# Emits one line per run start/finish, LONG_RUN once when a run passes the
# limit, every OOM/ERROR/WARNING line, and exits when the script finishes.
set -u
LOG="$1"; LIMIT="${2:-7200}"
until [[ -f "$LOG" ]]; do sleep 5; done
start=0; cur=""; warned=0
tail -n +1 -F "$LOG" 2>/dev/null | while IFS= read -r line; do
  now=$(date +%s)
  case "$line" in
    ./bazel-bin/*)          cur="${line#*--input_mode }"; cur="${cur%% --num_trees*}" ;;
    "----- Run "*)          start=$now; warned=0; echo "RUN_START [$cur] ${line//-/} $(date +%H:%M:%S)" ;;
    *"Training block took:"*) t="${line##*took: }"; echo "RUN_DONE  [$cur] $t (wall $((now-start)) s)" ;;
    *MEDIAN*|*OOM*|*ERROR*|*WARNING*) echo "$line" ;;
    CSV:*|*"parser failed"*) echo "FINISHED: $line"; exit 0 ;;
  esac
  if (( start > 0 && warned == 0 && now - start > LIMIT )); then
    echo "LONG_RUN [$cur] single run exceeded $LIMIT s and is still going"; warned=1
  fi
done
