#!/usr/bin/env bash
set -euo pipefail

# Create a subset directory of table images and (if present) copy/symlink matching GT files.
# Usage:
#   scripts/make_table_subset.sh --src data/table/images --dst data/table/sample_50 --n 50 --mode symlink
# Options:
#   --src   Source images directory (PNG/JPG/TIFF)
#   --dst   Destination directory (created if missing)
#   --n     Number of files to take (default: 50)
#   --mode  copy|symlink (default: symlink)

SRC=""
DST=""
N=50
MODE="symlink"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src) SRC="$2"; shift 2;;
    --dst) DST="$2"; shift 2;;
    --n) N="$2"; shift 2;;
    --mode) MODE="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$SRC" || -z "$DST" ]]; then
  echo "Usage: $0 --src <src_dir> --dst <dst_dir> [--n 50] [--mode copy|symlink]";
  exit 1
fi

mkdir -p "$DST"

mapfile -t FILES < <(find "$SRC" -maxdepth 1 \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.tif' -o -iname '*.tiff' \) | sort | head -n "$N")

link_or_copy() {
  local src="$1" dst="$2"
  if [[ "$MODE" == "copy" ]]; then
    cp -f "$src" "$dst"
  else
    ln -sf "$(realpath "$src")" "$dst"
  fi
}

for f in "${FILES[@]}"; do
  base=$(basename "$f")
  stem="${base%.*}"
  link_or_copy "$f" "$DST/$base"
  # Bring ground-truths if they exist
  for gt in "$SRC/$stem.gt.html" "$SRC/$stem.html" "$SRC/$stem.tables.json"; do
    if [[ -f "$gt" ]]; then
      link_or_copy "$gt" "$DST/$(basename "$gt")"
    fi
  done
done

echo "Subset prepared at $DST"

