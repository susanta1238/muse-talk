#!/usr/bin/env bash
set -euo pipefail

# Run from the MuseTalk project root (parent of api/)
cd "$(dirname "$0")/.."

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# -u forces unbuffered stdout/stderr so tqdm bars and print() calls
# from inside the MuseTalk pipeline stream to the terminal in real time.
export PYTHONUNBUFFERED=1

# uvicorn default log level plus disable access log buffering. --workers 1
# because models live in process memory and the app is stateful (async
# queue, SQLite). A second worker would duplicate the whole model load.
exec python -u -m uvicorn api.app.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers 1 \
    --log-level info \
    --access-log \
    --timeout-keep-alive 30
