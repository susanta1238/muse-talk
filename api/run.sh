#!/usr/bin/env bash
set -euo pipefail

# Run from the MuseTalk project root (parent of api/)
cd "$(dirname "$0")/.."

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

exec uvicorn api.app.main:app --host "$HOST" --port "$PORT" --workers 1
