"""Download and prepare ASVspoof 2019 LA (Edinburgh DataShare, handle 10283/3336).

    python -m voxguard.data.download --yes          # ~7.1 GB download + extract + manifests
    python -m voxguard.data.download --manifests-only

Licence: Open Data Commons Attribution (ODC-By); cite Todisco et al., Interspeech 2019.
No checksum is published, so integrity is checked via size and zip CRCs.
"""

from __future__ import annotations

import argparse
import logging
import time
import urllib.request
import zipfile
from pathlib import Path

from voxguard.data.manifests import build_asvspoof2019_la_manifests

logger = logging.getLogger(__name__)

LA_URL = "https://datashare.ed.ac.uk/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/download"
LA_SIZE_BYTES = 7_640_952_520  # verified via HTTP range response on 2026-11-01
DEFAULT_ROOT = Path("data/raw")
CHUNK = 1 << 20
TIMEOUT_S = 60


def _fetch_once(url: str, part: Path, expected_size: int) -> None:
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        if have and resp.status != 206:  # server ignored Range: start over
            have = 0
        with part.open("ab" if have else "wb") as f:
            done = have
            while chunk := resp.read(CHUNK):
                f.write(chunk)
                done += len(chunk)
                if (done // CHUNK) % 256 == 0:
                    logger.info("downloaded %.2f / %.2f GB", done / 1e9, expected_size / 1e9)


def download_resumable(url: str, dest: Path, expected_size: int, retries: int = 20) -> None:
    """Download ``url`` to ``dest``, resuming a partial ``.part`` file via HTTP Range.

    Socket reads time out after ``TIMEOUT_S`` and the download is retried (resuming), so a
    dropped connection or a sleeping laptop cannot hang it forever.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == expected_size:
        logger.info("%s already complete", dest)
        return
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            _fetch_once(url, part, expected_size)
        except (OSError, TimeoutError) as exc:
            logger.warning("attempt %d/%d failed: %s; retrying", attempt, retries, exc)
            time.sleep(min(5 * attempt, 30))
            continue
        break
    else:
        raise OSError(f"download failed after {retries} attempts")
    if part.stat().st_size != expected_size:
        raise OSError(f"{part}: size {part.stat().st_size} != expected {expected_size}")
    part.rename(dest)


def safe_extract(zip_path: Path, out_dir: Path) -> None:
    out_dir = out_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad:
            raise OSError(f"corrupt member in {zip_path}: {bad}")
        for name in zf.namelist():
            if not (out_dir / name).resolve().is_relative_to(out_dir):
                raise ValueError(f"unsafe path in archive: {name}")
        zf.extractall(out_dir)


def find_la_root(root: Path) -> Path:
    hits = list(root.rglob("ASVspoof2019_LA_cm_protocols"))
    if not hits:
        raise FileNotFoundError(
            f"no ASVspoof2019_LA_cm_protocols under {root}; extract LA.zip first"
        )
    return hits[0].parent


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="raw data directory")
    p.add_argument("--manifests-dir", type=Path, default=Path("data/processed/manifests"))
    p.add_argument("--yes", action="store_true", help="confirm the ~7.1 GB download")
    p.add_argument("--manifests-only", action="store_true", help="skip download/extract")
    p.add_argument("--keep-zip", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.manifests_only:
        if not args.yes:
            p.error(f"LA.zip is {LA_SIZE_BYTES / 1e9:.2f} GB; re-run with --yes to download")
        zip_path = args.root / "LA.zip"
        download_resumable(LA_URL, zip_path, LA_SIZE_BYTES)
        safe_extract(zip_path, args.root)
        if not args.keep_zip:
            zip_path.unlink()
    la_root = find_la_root(args.root)
    build_asvspoof2019_la_manifests(la_root, args.root, args.manifests_dir)


if __name__ == "__main__":
    main()
