#!/bin/bash
# Usage: memsample.sh <out.csv> <log> -- cmd args...
out=$1; log=$2; shift 3
t0=$(date +%s.%N)
"$@" > "$log" 2>&1 &
pid=$!
echo "t_s,rss_mb,vmhwm_mb,sys_used_mb,threads" > "$out"
while kill -0 $pid 2>/dev/null; do
  now=$(date +%s.%N)
  st=$(cat /proc/$pid/status 2>/dev/null) || break
  rss=$(awk '/^VmRSS/{print $2/1024}' <<<"$st")
  hwm=$(awk '/^VmHWM/{print $2/1024}' <<<"$st")
  thr=$(awk '/^Threads/{print $2}' <<<"$st")
  used=$(awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}END{print (t-a)/1024}' /proc/meminfo)
  printf "%.2f,%s,%s,%s,%s\n" "$(echo "$now - $t0" | bc)" "$rss" "$hwm" "$used" "$thr" >> "$out"
  sleep 1
done
wait $pid; echo "EXIT=$? at $(echo "$(date +%s.%N) - $t0" | bc)" >> "$log"
