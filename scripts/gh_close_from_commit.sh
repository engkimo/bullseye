#!/usr/bin/env bash
set -euo pipefail

# Close issues referenced by a commit message. Default: HEAD.
# Requires: gh CLI authenticated, push rights to repo.

sha="${1:-HEAD}"
repo=$(gh repo view --json nameWithOwner -q .nameWithOwner)
msg=$(git log -1 --pretty=%B "$sha")

echo "Repo: $repo"
echo "Commit: $(git rev-parse "$sha")"

# Extract issue numbers: patterns like #123 or numbers in Issues: 1,2
nums_from_hash=$(printf '%s' "$msg" | grep -oE '#[0-9]+' | sed 's/#//' | tr '\n' ' ' || true)
nums_from_list=$(printf '%s' "$msg" | sed -n 's/^Issues:\s*//p' | tr ',' ' ' || true)

issues=$(printf '%s %s\n' "$nums_from_hash" "$nums_from_list" | tr ' ' '\n' | sed '/^$/d' | sort -u)

if [[ -z "$issues" ]]; then
  echo "No issue references found in commit message. Add '#<num>' or 'Issues: 1,2' to commit message, or pass numbers explicitly."
  exit 0
fi

for n in $issues; do
  echo "Closing issue #$n ..."
  gh issue close "$n" --comment "Closed by commit $(git rev-parse "$sha")." || true
done

echo "Done."

