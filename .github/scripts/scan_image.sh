#!/usr/bin/env bash
# Image scan with Trivy (pinned by digest in CI's TRIVY_IMAGE): OS and Python packages.
#
# - Gate: fails on fixable CRITICAL vulnerabilities, except entries accepted in .trivyignore
#   (validated by check_trivyignore.py: reason and unexpired date required).
# - Report: every HIGH and CRITICAL finding, unfixed and accepted ones included, goes to the job
#   summary, so nothing excluded from the gate is hidden.
# - Identity: with EXPECTED_REVISION set (always in CI), each image must carry the label
#   org.opencontainers.image.revision=$EXPECTED_REVISION, i.e. be built from this commit.
# Any Trivy error (e.g. its database cannot be downloaded) fails the scan.
# Usage: scan_image.sh IMAGE [IMAGE...]
set -euo pipefail
cd "$(dirname "$0")/../.."
trivy_image="${TRIVY_IMAGE:?set TRIVY_IMAGE (see .github/workflows/ci.yml)}"
summary="${GITHUB_STEP_SUMMARY:-/dev/stdout}"
[[ $# -ge 1 ]] || { echo "usage: $0 IMAGE [IMAGE...]" >&2; exit 2; }

trivy() {
  docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v "$PWD:/src:ro" \
    -v trivy-cache:/root/.cache "$trivy_image" image --quiet --scanners vuln "$@"
}

python3 .github/scripts/check_trivyignore.py | tee -a "$summary"

status=0
for image in "$@"; do
  id="$(docker image inspect -f '{{.Id}}' "$image")"
  revision="$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$image")"
  if [[ -n "${EXPECTED_REVISION:-}" && "$revision" != "$EXPECTED_REVISION" ]]; then
    echo "::error::$image was built from '${revision:-unknown}', not $EXPECTED_REVISION" >&2
    exit 1
  fi
  {
    echo "### Trivy: \`$image\`"
    echo "Image \`${id:7:12}\`, built from \`${revision:-unknown}\`. All HIGH and CRITICAL findings"
    echo "(unfixed and accepted ones included); the gate is fixable CRITICAL only."
    echo '```'
    trivy --severity HIGH,CRITICAL --format table "$image"
    echo '```'
  } >> "$summary"
  if ! trivy --severity CRITICAL --ignore-unfixed --ignorefile /src/.trivyignore \
      --exit-code 1 --format table "$image"; then
    echo "::error::$image has fixable CRITICAL vulnerabilities (or the scan failed)" >&2
    status=1
  fi
done
exit "$status"
