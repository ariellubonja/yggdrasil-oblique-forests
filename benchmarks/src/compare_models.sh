#!/usr/bin/env bash
# Usage: compare_models.sh <dir_a> <dir_b>  (two --model_out_dir models)  exit 0 identical, 1 differ, 2 usage/IO error
# nodes-* compare as bytes. Known .pb files compare as protoc-decoded text (maps sorted), ignoring only the
# random metadata uid/created_date. Without protoc ($PROTOC or bazel-bin) they fall back to bytes.
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then sed -n '2,4p' "$0" | sed 's/^# \{0,1\}//'; exit 0; fi
if [[ $# -ne 2 ]]; then echo "Usage: $0 <dir_a> <dir_b>" >&2; exit 2; fi
DIR_A="$1"; DIR_B="$2"
for d in "$DIR_A" "$DIR_B"; do [[ -d "$d" ]] || { echo "ERROR: not a directory: $d" >&2; exit 2; }; done

REPO=$(cd "$(dirname "$0")/../.." && pwd)
PROTOC=${PROTOC:-$REPO/bazel-bin/external/protobuf~/protoc}
if [[ -x "$PROTOC" ]]; then
  PINC=${PROTOC_INCLUDE:-$(cd "$REPO" && bazel info output_base 2>/dev/null)/external/protobuf~/src}
else
  echo "WARNING: no protoc (set PROTOC or bazel build @com_google_protobuf//:protoc): .pb files compare as bytes" >&2
  PROTOC=""
fi
proto_of() {  # <file> -> "<proto path> <message>" for known model protos, else empty
  case "$1" in
    header.pb) echo "model/abstract_model.proto model.proto.AbstractModel" ;;
    data_spec.pb) echo "dataset/data_spec.proto dataset.proto.DataSpecification" ;;
    gradient_boosted_trees_header.pb) echo "model/gradient_boosted_trees/gradient_boosted_trees.proto model.gradient_boosted_trees.proto.Header" ;;
    random_forest_header.pb) echo "model/random_forest/random_forest.proto model.random_forest.proto.Header" ;;
  esac
}
decode() {  # <proto path> <message> <pb file>
  (cd "$REPO" && "$PROTOC" -I. -I"$PINC" --decode="yggdrasil_decision_forests.$2" "yggdrasil_decision_forests/$1" < "$3") |
    if [[ "$2" == model.proto.AbstractModel ]]; then grep -v -E '^  (uid|created_date): ' || true; else cat; fi
}
if command -v sha256sum >/dev/null 2>&1; then hash_of() { sha256sum "$1" | awk '{print $1}'; }
else hash_of() { shasum -a 256 "$1" | awk '{print $1}'; }; fi

list_files() { ( cd "$1" && find . -type f ! -name done | sed 's|^\./||' ); }
FILES=()
while IFS= read -r f; do [[ -n "$f" ]] && FILES+=("$f"); done < <({ list_files "$DIR_A"; list_files "$DIR_B"; } | sort -u)
[[ ${#FILES[@]} -gt 0 ]] || { echo "ERROR: no model files found in either directory" >&2; exit 2; }

printf '%-34s %-12s %s\n' "FILE" "STATUS" "SHA256 (a / b)"
overall_match=1; nodes_match=1; nodes_seen=0
for f in "${FILES[@]}"; do
  pa="$DIR_A/$f"; pb="$DIR_B/$f"
  [[ "$f" == nodes-* ]] && nodes_seen=1
  if [[ ! -f "$pa" || ! -f "$pb" ]]; then
    printf '%-34s %-12s\n' "$f" "MISSING-$([[ -f "$pa" ]] && echo B || echo A)"
    overall_match=0; [[ "$f" == nodes-* ]] && nodes_match=0; continue
  fi
  ha=$(hash_of "$pa"); hb=$(hash_of "$pb")
  if [[ "$ha" == "$hb" ]]; then printf '%-34s %-12s %s\n' "$f" "MATCH" "${ha:0:16}…"; continue; fi
  spec=$(proto_of "$f")
  if [[ -n "$PROTOC" && -n "$spec" ]] && diff <(decode $spec "$pa") <(decode $spec "$pb") > "${TMPDIR:-/tmp}/cmpmodels.$$" 2>&1; then
    printf '%-34s %-12s %s\n' "$f" "MATCH-DEC" "(decoded; bytes differ by map order / uid)"; continue
  fi
  printf '%-34s %-12s %s\n' "$f" "DIFFER" "${ha:0:16}… / ${hb:0:16}…"
  [[ -s "${TMPDIR:-/tmp}/cmpmodels.$$" ]] && head -20 "${TMPDIR:-/tmp}/cmpmodels.$$" | sed 's/^/    /'
  overall_match=0; [[ "$f" == nodes-* ]] && nodes_match=0
done
rm -f "${TMPDIR:-/tmp}/cmpmodels.$$"

echo
[[ "$nodes_seen" -eq 1 ]] || echo "WARNING: no nodes-* shard found -- is this a saved model dir?" >&2
if [[ "$overall_match" -eq 1 ]]; then echo "RESULT: BIT-IDENTICAL ✅  (all model files match)"; exit 0
elif [[ "$nodes_match" -eq 1 && "$nodes_seen" -eq 1 ]]; then
  echo "RESULT: TREES IDENTICAL ✅ but metadata differs ⚠️  (nodes-* match; other files differ above)"; exit 1
else echo "RESULT: MODEL CHANGED ❌  (tree nodes differ)"; exit 1; fi
