#!/usr/bin/env bash
# Dependency audit: every package pinned in uv.lock (all extras, groups and platforms, transitive
# ones included; see audit_inputs.py) against the PyPA / OSV advisory databases. Nothing is
# installed. Fails on any known vulnerability, unless it is listed in
# .github/pip-audit-ignore.txt with a reason and a review date that has not passed. Any error
# (network, malformed exception, unauditable package) fails too. Runs on bash 3.2 (macOS) too: an
# empty array is expanded as ${a[@]+"${a[@]}"}, since bash before 4.4 calls "${a[@]}" unbound
# under set -u.
set -euo pipefail
cd "$(dirname "$0")/../.."
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
summary="${GITHUB_STEP_SUMMARY:-/dev/null}"

python3 .github/scripts/audit_inputs.py "$work" | tee -a "$summary"
# Read line by line, not with mapfile: macOS ships bash 3.2, which has no mapfile.
ignore=()
while IFS= read -r arg; do
  if [[ -n "$arg" ]]; then ignore+=("$arg"); fi
done < "$work/ignore-args.txt"

status=0
for req in "$work"/req-*.txt; do
  uvx "pip-audit==${PIP_AUDIT_VERSION:-2.10.1}" -r "$req" --no-deps --disable-pip \
    --progress-spinner off --desc on ${ignore[@]+"${ignore[@]}"} || status=1
done
exit "$status"
