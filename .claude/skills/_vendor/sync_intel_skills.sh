#!/usr/bin/env bash
# Re-vendor intel/intel-performance-skills into .claude/skills/ (project scope).
# Usage: bash .claude/skills/_vendor/sync_intel_skills.sh [git-ref]
set -euo pipefail
ref="${1:-main}"
root="$(git rev-parse --show-toplevel)"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
git clone -q https://github.com/intel/intel-performance-skills "$tmp/src"
git -C "$tmp/src" checkout -q "$ref"
for s in "$tmp"/src/skills/*/; do
  name="$(basename "$s")"
  rm -rf "$root/.claude/skills/$name"
  cp -R "$s" "$root/.claude/skills/$name"
  cp "$tmp/src/COPYRIGHT.md" "$root/.claude/skills/$name/LICENSE.md"
done
echo "$(git -C "$tmp/src" rev-parse HEAD) $(date -u +%F)" > "$root/.claude/skills/_vendor/intel_skills.rev"
echo "Vendored intel-performance-skills @ $(cat "$root/.claude/skills/_vendor/intel_skills.rev")"
