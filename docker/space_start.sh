#!/bin/sh
# Start the API in the background, wait until it is healthy, then run the UI in the
# foreground. If the API fails to start, exit with its error so the Space shows it in the logs.
# VOXGUARD_API_START_TIMEOUT (seconds) bounds the wait; the default is 600 s for the real
# detector (a cold start downloads the checkpoint) and 60 s otherwise.
set -eu

uvicorn voxguard.api.main:create_app --factory --host 127.0.0.1 --port 8000 &
api_pid=$!
export VOXGUARD_API_PID="$api_pid"

python - <<'PY'
import os
import sys
import time
import urllib.request

real = os.environ.get("VOXGUARD_DETECTOR", "stub").strip().lower() == "real"
raw = os.environ.get("VOXGUARD_API_START_TIMEOUT", "").strip()
try:
    timeout = float(raw) if raw else (600.0 if real else 60.0)
except ValueError:
    sys.exit(f"VOXGUARD_API_START_TIMEOUT must be a number of seconds, got {raw!r}")
pid = int(os.environ["VOXGUARD_API_PID"])
deadline = time.monotonic() + timeout
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        sys.exit("VoxGuard API exited during startup; see the error above.")
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
        sys.exit(0)
    except OSError:
        time.sleep(0.5)
sys.exit(f"VoxGuard API did not become healthy within {timeout:g} s")
PY

exec streamlit run src/voxguard/ui/app.py \
    --server.address=0.0.0.0 --server.port=7860 --server.headless=true
