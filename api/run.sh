#!/usr/bin/env bash
set -euo pipefail

# Run from the MuseTalk project root (parent of api/)
cd "$(dirname "$0")/.."

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Force unbuffered stdout/stderr so tqdm bars and print() calls from
# inside the MuseTalk pipeline stream to the terminal in real time.
export PYTHONUNBUFFERED=1

# Pick the python interpreter where uvicorn was actually installed.
# On RunPod's pytorch image, pip installs to python3.10's site-packages
# but /usr/bin/python is often python3.8. Probe for a working one.
PY=""
for candidate in python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c "import uvicorn" 2>/dev/null; then
            PY="$candidate"
            break
        fi
    fi
done

if [[ -z "$PY" ]]; then
    echo "ERROR: could not find a python with uvicorn installed." >&2
    echo "Tried: python3.10, python3, python" >&2
    echo "Run 'bash api/bootstrap.sh' first." >&2
    exit 1
fi

echo "Launching uvicorn via $PY ($(command -v "$PY"))"

exec "$PY" -u -m uvicorn api.app.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers 1 \
    --log-level info \
    --access-log \
    --timeout-keep-alive 30
