#!/bin/sh
# Start the API in the background, wait until it is healthy, then run the UI in the
# foreground. If the API fails to start, exit with its error so the Space shows it in the logs.
set -eu

uvicorn voxguard.api.main:create_app --factory --host 127.0.0.1 --port 8000 &
api_pid=$!

python - <<'PY'
import sys
import time
import urllib.request

for _ in range(120):  # up to 60 s (the real detector may download a checkpoint first)
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
        sys.exit(0)
    except OSError:
        time.sleep(0.5)
sys.exit("VoxGuard API did not become healthy within 60 s")
PY
if ! kill -0 "$api_pid" 2>/dev/null; then
    echo "VoxGuard API exited during startup; see the error above." >&2
    exit 1
fi

exec streamlit run src/voxguard/ui/app.py \
    --server.address=0.0.0.0 --server.port=7860 --server.headless=true
