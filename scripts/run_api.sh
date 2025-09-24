#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"

if ! command -v uvicorn >/dev/null 2>&1; then
  echo "[error] uvicorn not found. Try: pip install uvicorn fastapi" >&2
  exit 1
fi

echo "[run] DocJA API starting at http://${HOST}:${PORT}"
exec uvicorn src.api:app --host "$HOST" --port "$PORT"

