#!/usr/bin/env bash
set -euo pipefail

# Dataset downloader (license-aware). Does not hardcode URLs.
# Provide URLs via env vars or prompt. Intended datasets (examples):
#  - SynthDoG-ja (text detection/recognition)
#  - DocLayNet (layout) [CC BY-NC 4.0]
#  - PubTabNet (tables) [CC BY-NC-SA 4.0]
# This script only acts as a hook and asks for explicit confirmation.

confirm() {
  local prompt="$1"
  read -r -p "$prompt [y/N]: " ans || true
  case "$ans" in
    [yY][eE][sS]|[yY]) return 0 ;;
    *) return 1 ;;
  esac
}

ROOT_DIR=${1:-$(pwd)}
DATA_DIR="$ROOT_DIR/data"
mkdir -p "$DATA_DIR"

echo "[datasets] Destination: $DATA_DIR"

# 1) SynthDoG-ja
if confirm "Download SynthDoG-ja (research permitted). Provide URL via SYNTHDOG_URL env?"; then
  if [ -z "${SYNTHDOG_URL:-}" ]; then
    read -r -p "Enter SynthDoG-ja archive URL: " SYNTHDOG_URL
  fi
  echo "[datasets] Downloading SynthDoG-ja..."
  mkdir -p "$DATA_DIR/synthdog_ja"
  curl -L "$SYNTHDOG_URL" -o "$DATA_DIR/synthdog_ja/synthdog_ja.tar.gz"
  tar -xzf "$DATA_DIR/synthdog_ja/synthdog_ja.tar.gz" -C "$DATA_DIR/synthdog_ja" || true
fi

# 2) DocLayNet
if confirm "Download DocLayNet (CC BY-NC 4.0). Confirm non-commercial or permission?"; then
  if [ -z "${DOCLAYNET_URL:-}" ]; then
    read -r -p "Enter DocLayNet archive URL: " DOCLAYNET_URL
  fi
  echo "[datasets] Downloading DocLayNet..."
  mkdir -p "$DATA_DIR/doclaynet"
  curl -L "$DOCLAYNET_URL" -o "$DATA_DIR/doclaynet/doclaynet.zip"
  unzip -o "$DATA_DIR/doclaynet/doclaynet.zip" -d "$DATA_DIR/doclaynet" || true
fi

# 3) PubTabNet
if confirm "Download PubTabNet (CC BY-NC-SA 4.0). Confirm license compliance?"; then
  if [ -z "${PUBTABNET_URL:-}" ]; then
    read -r -p "Enter PubTabNet archive URL: " PUBTABNET_URL
  fi
  echo "[datasets] Downloading PubTabNet..."
  mkdir -p "$DATA_DIR/pubtabnet"
  curl -L "$PUBTABNET_URL" -o "$DATA_DIR/pubtabnet/pubtabnet.zip"
  unzip -o "$DATA_DIR/pubtabnet/pubtabnet.zip" -d "$DATA_DIR/pubtabnet" || true
fi

echo "[datasets] Completed. Configure dataset paths in configs/*.yaml or .env."

