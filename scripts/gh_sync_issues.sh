#!/usr/bin/env bash
set -euo pipefail

# Create milestones and issues on GitHub from local markdown files.
# Prereqs: gh CLI authenticated (gh auth login), repo remote resolvable.

detect_repo() {
  if [[ -n "${GITHUB_REPOSITORY:-}" ]]; then
    echo "$GITHUB_REPOSITORY"
    return
  fi
  gh repo view --json nameWithOwner -q .nameWithOwner
}

ensure_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' is required"; exit 1; }
}

create_milestone() {
  local title="$1"; shift
  local description="$1"; shift || true

  # If exists, skip
  if gh api repos/:owner/:repo/milestones --jq '.[].title' | grep -Fxq "$title"; then
    echo "Milestone exists: $title"
    return
  fi
  echo "Creating milestone: $title"
  gh api -X POST repos/:owner/:repo/milestones \
    -f title="$title" \
    -f description="$description" >/dev/null
}

ensure_label() {
  local name="$1"
  # check existence
  if gh label list --limit 200 --json name --jq '.[].name' | grep -Fxq "$name"; then
    return
  fi
  echo "Creating label: $name"
  gh label create "$name" --color BFD4F2 >/dev/null || true
}

create_issue_from_file() {
  local file="$1"
  local title milestone labels body
  title=$(sed -n '1p' "$file" | sed -E 's/^#\s*//')
  milestone=$(grep -m1 '^Milestone:' "$file" | cut -d: -f2- | xargs || true)
  labels=$(grep -m1 '^Labels:' "$file" | cut -d: -f2- | xargs || true)
  body=$(awk 'NR>1{print}' "$file")

  # Skip if title already exists (exact match)
  if gh issue list --limit 200 --json title --jq '.[].title' | grep -Fxq "$title"; then
    echo "Issue exists: $title"
    return
  fi

  echo "Creating issue: $title"
  local args=(issue create --title "$title" --body "$body")
  if [[ -n "$milestone" ]]; then args+=(--milestone "$milestone"); fi
  if [[ -n "$labels" ]]; then 
    IFS=',' read -ra LBL <<<"$labels"
    for l in "${LBL[@]}"; do 
      local lab
      lab=$(echo "$l" | xargs)
      [[ -z "$lab" ]] && continue
      ensure_label "$lab"
      args+=(--label "$lab")
    done
  fi
  gh "${args[@]}" >/dev/null
}

main() {
  ensure_cmd gh
  # Ensure repo is resolvable
  REPO=$(detect_repo)
  echo "Target repo: $REPO"

  echo "Syncing milestones..."
  create_milestone "M0: Refactor/Abstraction Foundation" "Clean Architecture scaffolding and adapters"
  create_milestone "M1: P0 — DI v1 + UDJ Core" "v1 API/UDJ/table OCR/vis/PDF/LLM/metrics"
  create_milestone "M2: P1 — Eval/Lite/Label/Bootstrap" "evaluation, lite mode, label map, bootstrap/.env, compat"
  create_milestone "M3: P0.5 — Flow/Gantt v0.4.5" "Flow/Graph JSON and Gantt/Chart JSON v0 with overlays"
  create_milestone "M4: P2 — Performance/Security/Monitoring" "perf p95<=2s, security/RBAC, monitoring"

  echo "Syncing issues from docs/pm/issues/*.md ..."
  shopt -s nullglob
  for f in docs/pm/issues/*.md; do
    create_issue_from_file "$f"
  done
  echo "Done."
}

main "$@"
