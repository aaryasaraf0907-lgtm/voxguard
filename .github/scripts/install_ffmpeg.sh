#!/usr/bin/env bash
# Install ffmpeg on a GitHub-hosted Ubuntu runner, resilient to one apt mirror being down.
#
# The runner's sources use apt's mirror method (mirror+file:/etc/apt/apt-mirrors.txt), which
# tries the listed mirrors in order and falls back to the next one. Plain-HTTP mirrors stopped
# answering runners on 11 Nov 2026, so an HTTPS mirror goes first; the runner's own mirrors are
# kept after it as fallbacks (overwriting the list made one mirror outage fail the whole job,
# 18 Nov). IPv4 is forced (one outage was an IPv6 connect failure), and the update + install is
# retried up to 3 times with a back-off.
#
# Overridable for tests: MIRRORS_FILE, SUDO, APT_GET, SLEEP.
set -euo pipefail
mirrors_file="${MIRRORS_FILE:-/etc/apt/apt-mirrors.txt}"
sudo_cmd="${SUDO-sudo}"
apt_get="${APT_GET:-apt-get}"
sleep_cmd="${SLEEP:-sleep}"
primary="https://mirrors.edge.kernel.org/ubuntu/"

existing=""
[[ -f "$mirrors_file" ]] && existing="$(cat "$mirrors_file")"
{
  echo "$primary"
  printf '%s\n' "$existing" | grep -v -x -F "$primary" | grep -v '^[[:space:]]*$' || true
} | $sudo_cmd tee "$mirrors_file.new" >/dev/null
$sudo_cmd mv "$mirrors_file.new" "$mirrors_file"
echo "apt mirrors, in order:"
sed 's/^/  /' "$mirrors_file"

opts=(-o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30
      -o Acquire::ForceIPv4=true)
for attempt in 1 2 3; do
  if $sudo_cmd "$apt_get" "${opts[@]}" update \
      && $sudo_cmd "$apt_get" "${opts[@]}" install -y --no-install-recommends ffmpeg; then
    exit 0
  fi
  echo "ffmpeg install attempt $attempt failed" >&2
  [[ $attempt -lt 3 ]] && $sleep_cmd $((attempt * 20))
done
echo "::error::could not install ffmpeg from any apt mirror after 3 attempts" >&2
exit 1
