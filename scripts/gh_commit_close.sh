#!/usr/bin/env bash
set -euo pipefail

# Commit all changes with a message and close issues via gh CLI.
# Usage: gh_commit_close.sh -m "message" [-i "1,2 3"] [-b main] [-r origin]

branch="main"
remote="origin"
issues=""
message=""

while getopts ":m:i:b:r:" opt; do
  case $opt in
    m) message="$OPTARG" ;;
    i) issues="$OPTARG" ;;
    b) branch="$OPTARG" ;;
    r) remote="$OPTARG" ;;
    *) echo "Usage: $0 -m \"message\" [-i \"1,2 3\"] [-b main] [-r origin]"; exit 2 ;;
  esac
done

if [[ -z "$message" ]]; then
  echo "ERROR: commit message (-m) is required"; exit 2
fi

git add -A
if git diff --cached --quiet; then
  echo "No staged changes. Skipping commit."
else
  git commit -m "$message"
fi

git push "$remote" "$branch"

sha=$(git rev-parse HEAD)
repo=$(gh repo view --json nameWithOwner -q .nameWithOwner)
echo "Pushed $sha to $remote/$branch ($repo)"

# If no explicit issues, try to parse from commit message
if [[ -z "$issues" ]]; then
  msg=$(git log -1 --pretty=%B "$sha")
  from_hash=$(printf '%s' "$msg" | grep -oE '#[0-9]+' | sed 's/#//' | tr '\n' ' ' || true)
  from_list=$(printf '%s' "$msg" | sed -n 's/^Issues:\s*//p' | tr ',' ' ' || true)
  issues=$(printf '%s %s\n' "$from_hash" "$from_list" | tr ' ' '\n' | sed '/^$/d' | sort -u | tr '\n' ' ')
fi

if [[ -z "$issues" ]]; then
  echo "No issues to close. Add '#<num>' in message or pass -i '1,2'"
  exit 0
fi

for token in $issues; do
  n=$(echo "$token" | sed 's/#//g')
  [[ -z "$n" ]] && continue
  echo "Closing issue #$n ..."
  gh issue close "$n" --comment "Closed by commit $sha." || true
done

echo "Done."

