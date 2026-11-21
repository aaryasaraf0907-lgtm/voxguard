"""Read single members of a large remote zip over HTTP range requests (no full download).

Used by the head-only experiment to fetch only the ASVspoof 2019 LA and In-the-Wild files it
needs. ``python remote_zip.py list URL`` prints the members; ``fetch URL OUT NAME...`` extracts
them (existing files are skipped, so a fetch can be resumed). Stdlib only.
"""

from __future__ import annotations

import io
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

BLOCK = int(os.environ.get("REMOTE_ZIP_BLOCK", 1 << 20))  # bytes per request: large for
# whole splits read in order, small (e.g. 65536) for scattered subsets


class HTTPRangeFile(io.RawIOBase):
    """A seekable, read-only file over HTTP Range requests, with a small block cache."""

    def __init__(self, url: str, retries: int = 5) -> None:
        self.url, self.retries, self.pos = url, retries, 0
        self.size = int(self._get(0, 0)[1])
        self._cache: dict[int, bytes] = {}

    def _get(self, start: int, end: int) -> tuple[bytes, str]:
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    if r.status != 206:
                        raise OSError(f"server ignored the Range header (HTTP {r.status})")
                    total = r.headers["Content-Range"].rsplit("/", 1)[1]
                    return r.read(), total
            except OSError:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 * (attempt + 1))
        raise AssertionError("unreachable")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            start = i * BLOCK
            self._cache[i] = self._get(start, min(start + BLOCK, self.size) - 1)[0]
        return self._cache[i]

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while len(out) < n:
            i, off = divmod(self.pos + len(out), BLOCK)
            out += self._block(i)[off : off + n - len(out)]
        self.pos += n
        return bytes(out)

    def readinto(self, b: bytearray) -> int:  # type: ignore[override]
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def open_zip(url: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(url), buffer_size=BLOCK))


def fetch(zf: zipfile.ZipFile, out: Path, names: list[str]) -> int:
    """Extract ``names`` in archive order (so neighbouring members share range requests)."""
    done = 0
    for name in sorted(names, key=lambda n: zf.getinfo(n).header_offset):
        dest = out / name
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(zf.read(name))
        tmp.rename(dest)
        done += 1
    return done


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "list":
        for info in open_zip(argv[1]).infolist():
            print(f"{info.file_size}\t{info.filename}")
        return 0
    if len(argv) >= 4 and argv[0] == "fetch":
        names = argv[3:]
        if len(names) == 1 and names[0].startswith("@"):  # @file: one member name per line
            names = Path(names[0][1:]).read_text().split()
        n = fetch(open_zip(argv[1]), Path(argv[2]), names)
        print(f"fetched {n}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
