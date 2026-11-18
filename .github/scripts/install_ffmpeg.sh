#!/usr/bin/env bash
# Install ffmpeg on a GitHub-hosted Ubuntu runner, resilient to an apt mirror being down or slow.
#
# The runner's sources use apt's mirror method (mirror+file:/etc/apt/apt-mirrors.txt), which tries
# the listed mirrors in order and falls back to the next one on failure (not on slowness).
# History: plain-HTTP mirrors stopped answering runners on 11 Nov 2026, so CI pinned one HTTPS
# mirror; on 18 Nov that mirror was unreachable for some runners and crawling (about 100 kB/s) for
# others. So:
#   1. candidates = two HTTPS mirrors plus the runner's own mirrors, deduplicated;
#   2. each is probed with a small timed download (its InRelease file, 10 s cap);
#   3. the list is rewritten fastest first with explicit priority:1..N tags, unreachable ones last
#      but kept as fallbacks. The tags matter: without them apt's mirror method picks among the
#      listed mirrors at random (apt-transport-mirror(1), "Fallback order for mirrors"), as the
#      first CI run of this script showed (requests spread over all four, 82-135 kB/s);
#   4. apt runs with IPv4 forced and more retries; update + install are retried 3 times.
# The same list then serves later apt users in the job (e.g. Playwright's --with-deps).
#
# Overridable for tests: MIRRORS_FILE, SUDO, APT_GET, SLEEP, CURL, CODENAME.
set -euo pipefail
mirrors_file="${MIRRORS_FILE:-/etc/apt/apt-mirrors.txt}"
sudo_cmd="${SUDO-sudo}"
apt_get="${APT_GET:-apt-get}"
sleep_cmd="${SLEEP:-sleep}"
curl_cmd="${CURL:-curl}"
codename="${CODENAME:-$(. /etc/os-release && echo "${VERSION_CODENAME:?}")}"

candidates=("https://archive.ubuntu.com/ubuntu/" "https://mirrors.edge.kernel.org/ubuntu/")
if [[ -f "$mirrors_file" ]]; then
  while read -r url _; do
    [[ -n "$url" && "$url" != \#* ]] && candidates+=("$url")
  done < "$mirrors_file"
fi

declare -A seen=()
ranked=()
for url in "${candidates[@]}"; do
  [[ "$url" == */ ]] || url="$url/"
  [[ -n "${seen[$url]:-}" ]] && continue
  seen[$url]=1
  speed="$("$curl_cmd" -fsS -o /dev/null --max-time 10 -w '%{speed_download}' \
    "${url}dists/${codename}/InRelease" 2>/dev/null || true)"
  speed="${speed%%.*}"
  [[ "$speed" =~ ^[0-9]+$ ]] || speed=0
  ranked+=("$speed $url")
done

printf '%s\n' "${ranked[@]}" | sort -s -k1,1nr | awk '{ print $2 "\tpriority:" NR }' \
  | $sudo_cmd tee "$mirrors_file.new" >/dev/null
$sudo_cmd mv "$mirrors_file.new" "$mirrors_file"
echo "apt mirrors, fastest first (probe speed in bytes/s):"
printf '%s\n' "${ranked[@]}" | sort -s -k1,1nr | sed 's/^/  /'

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
