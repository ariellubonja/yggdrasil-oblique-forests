#!/usr/bin/env bash
set -euo pipefail

# Bit-identity check for two saved models (--model_out_dir outputs).
#
# The bit-identity contract (OBLIQUE_CONTEXT.md §12.1) is: a code change must
# produce bit-identical trees at the same seed and scheduler. Test accuracy is
# NECESSARY BUT NOT SUFFICIENT -- two builds can tie on accuracy while producing
# structurally different trees. This tool proves the model itself is unchanged
# by hashing the serialized forest byte-for-byte.
#
# Compares two model directories produced by
#   train_oblique_forest --model_out_dir=<dir>
# On-disk files (model/random_forest/random_forest.cc):
#   nodes-NNNNN-of-MMMMM  <- tree topology, thresholds, oblique weights. THE signal.
#   header.pb / random_forest_header.pb / data_spec.pb  <- metadata & counts.
#   done                  <- empty completion marker (ignored).
# These RF files carry no embedded timestamps, so an exact hash match is
# meaningful.
#
# Usage:  $0 <dir_a> <dir_b>
#   e.g.  $0 dw1_sr_baseline dw1_sr_no_block
#
# Exit code: 0 = models bit-identical, 1 = differ, 2 = usage/IO error.

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '3,26p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi

if [[ $# -ne 2 ]]; then
  echo "ERROR: expected 2 directories, got $#" >&2
  echo "Usage: $0 <dir_a> <dir_b>" >&2
  exit 2
fi

DIR_A="$1"
DIR_B="$2"

for d in "$DIR_A" "$DIR_B"; do
  if [[ ! -d "$d" ]]; then
    echo "ERROR: not a directory: $d" >&2
    exit 2
  fi
done

# Pick a sha256 tool (macOS: shasum -a 256; Linux: sha256sum).
if command -v sha256sum >/dev/null 2>&1; then
  hash_of() { sha256sum "$1" | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then
  hash_of() { shasum -a 256 "$1" | awk '{print $1}'; }
else
  echo "ERROR: need sha256sum or shasum on PATH" >&2
  exit 2
fi

# Union of relative filenames across both dirs, excluding the empty 'done'
# marker. Portable to BSD find (macOS) and bash 3.2 -- no -printf, no mapfile.
list_files() { ( cd "$1" && find . -type f ! -name done | sed 's|^\./||' ); }

FILES=()
while IFS= read -r f; do
  [[ -n "$f" ]] && FILES+=("$f")
done < <(
  { list_files "$DIR_A"; list_files "$DIR_B"; } 2>/dev/null | sort -u
)

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "ERROR: no model files found in either directory" >&2
  exit 2
fi

printf '%-32s %-12s %s\n' "FILE" "STATUS" "SHA256 (a / b)"
printf '%s\n' "--------------------------------------------------------------------------------"

overall_match=1
nodes_match=1
nodes_seen=0

for f in "${FILES[@]}"; do
  pa="$DIR_A/$f"
  pb="$DIR_B/$f"

  if [[ ! -f "$pa" ]]; then
    printf '%-32s %-12s %s\n' "$f" "MISSING-A" "(only in $DIR_B)"
    overall_match=0
    [[ "$f" == nodes-* ]] && { nodes_match=0; nodes_seen=1; }
    continue
  fi
  if [[ ! -f "$pb" ]]; then
    printf '%-32s %-12s %s\n' "$f" "MISSING-B" "(only in $DIR_A)"
    overall_match=0
    [[ "$f" == nodes-* ]] && { nodes_match=0; nodes_seen=1; }
    continue
  fi

  ha=$(hash_of "$pa")
  hb=$(hash_of "$pb")

  if [[ "$ha" == "$hb" ]]; then
    printf '%-32s %-12s %s\n' "$f" "MATCH" "${ha:0:16}…"
  else
    printf '%-32s %-12s %s\n' "$f" "DIFFER" "${ha:0:16}… / ${hb:0:16}…"
    overall_match=0
    [[ "$f" == nodes-* ]] && nodes_match=0
  fi
  [[ "$f" == nodes-* ]] && nodes_seen=1
done

echo
if [[ "$nodes_seen" -eq 0 ]]; then
  echo "WARNING: no nodes-* shard found -- is this a saved RF model dir?" >&2
fi

if [[ "$overall_match" -eq 1 ]]; then
  echo "RESULT: BIT-IDENTICAL ✅  (all model files match)"
  exit 0
elif [[ "$nodes_match" -eq 1 && "$nodes_seen" -eq 1 ]]; then
  echo "RESULT: TREES IDENTICAL ✅ but metadata differs ⚠️  (nodes-* match; other files differ above)"
  exit 1
else
  echo "RESULT: MODEL CHANGED ❌  (tree nodes differ)"
  exit 1
fi
