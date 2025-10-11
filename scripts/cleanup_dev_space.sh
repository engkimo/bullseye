#!/usr/bin/env bash
set -euo pipefail

echo "[DocJA] Dev space cleanup (conda/cache/legacy venv)"
echo "This will permanently delete local caches and conda installs under your home directory."
echo "Targets: ~/miniconda ~/miniconda3 ~/.cache/{huggingface,uv,pip} ./venv"

FORCE="${FORCE:-0}"
if [[ "$FORCE" != "1" ]]; then
  echo "Safety: set FORCE=1 to proceed. Example: FORCE=1 bash $0"
  exit 1
fi

echo "[Before] Disk usage:" && df -h /

targets=(
  "$HOME/miniconda"
  "$HOME/miniconda3"
  "$HOME/.cache/huggingface"
  "$HOME/.cache/uv"
  "$HOME/.cache/pip"
  "$(pwd)/venv"
)

for t in "${targets[@]}"; do
  if [[ -e "$t" ]]; then
    echo "Removing: $t"
    rm -rf "$t"
  else
    echo "Skip (not found): $t"
  fi
done

echo "[After] Disk usage:" && df -h /
echo "Done. Consider exporting UV_CACHE_DIR/XDG_CACHE_HOME/TMPDIR to a larger volume."

