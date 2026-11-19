#!/usr/bin/env bash
# shard_csv.sh IN.csv OUT_DIR [N=16]: split a headered CSV into N row-balanced shards, header on each.
# Rows keep their order (shard_00 = first rows); use for the natural datasets uploaded to GCS.
set -euo pipefail
IN=$1; OUT=$2; N=${3:-16}
mkdir -p "$OUT"; HDR=$(head -n1 "$IN")
split -n l/"$N" -d -a 2 --additional-suffix=.csv "$IN" "$OUT/shard_"   # needs a seekable file, not a pipe
for f in "$OUT"/shard_*.csv; do [ "$(head -n1 "$f")" = "$HDR" ] || sed -i "1i $HDR" "$f"; done
ls -la "$OUT" | tail -n +2 | awk '{print $5, $9}'
