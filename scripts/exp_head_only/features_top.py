"""Cache the input to XLS-R's last ``TOP`` layers (round 2d, README "Round 2d").

The same jobs and the same prepared waveforms as ``features.py`` (same seeds, so row i is the same
clip, copy and augmentation as in the head-only cache and ``teacher.npy``), but the stored tensor is
``hidden_states[24 - TOP]``: the input to layer 24 - TOP of p3's own front-end, before the last
layers and the final layer norm. Usage: ``python features_top.py SET`` -> ``features/SET_top``.
Workers (env ``TOP_WORKERS``, default 4) and batch 8 keep memory low next to a running training.
"""

from __future__ import annotations

import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from features import DIM, FRAMES, P3, X, jobs, prepare

TOP = 2


def main(name: str) -> None:
    import torch

    from voxguard.models import load_from_checkpoint

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    todo = jobs(name)
    out = X / "features" / f"{name}_top"
    out.mkdir(parents=True, exist_ok=True)
    progress, start = out / "progress.txt", 0
    if (out / "feats.npy").exists() and progress.exists():
        start = int(progress.read_text())
        feats = np.load(out / "feats.npy", mmap_mode="r+")
        n = feats.shape[0]
        assert n == len(todo), f"{out}/feats.npy has {n} rows, not {len(todo)}"
    else:
        feats = np.lib.format.open_memmap(out / "feats.npy", mode="w+", dtype=np.float16,
                                          shape=(len(todo), FRAMES, DIM))  # fmt: skip
    model, _ = load_from_checkpoint(P3, "cuda")
    w2v = model.frontend.model.eval()
    k = len(w2v.encoder.layers) - TOP
    bs, window = 8, 128
    with ProcessPoolExecutor(max_workers=int(os.environ.get("TOP_WORKERS", "4"))) as pool:
        nxt = pool.map(prepare, enumerate(todo[start : start + window]), chunksize=4)
        for lo in range(start, len(todo), window):
            xs = list(nxt)
            if lo + window < len(todo):
                part = enumerate(todo[lo + window : lo + 2 * window], lo + window)
                nxt = pool.map(prepare, part, chunksize=4)
            for b in range(0, len(xs), bs):
                x = torch.from_numpy(np.stack(xs[b : b + bs])).cuda()
                x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-7)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    h = w2v(x, output_hidden_states=True).hidden_states[k]
                feats[lo + b : lo + b + len(x)] = h.half().cpu().numpy()
            feats.flush()
            progress.write_text(str(lo + len(xs)))
            if (lo // window) % 16 == 0:
                print(f"{name}_top: {lo + len(xs)}/{len(todo)}", flush=True)
    with (out / "meta.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "file", "label", "group", "condition", "copy"])
        for i, j in enumerate(todo):
            w.writerow([i, j["path"].name, j["label"], j["group"], j["cond"], j["copy"]])
    print(f"{name}_top: done, {len(todo)} rows", flush=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
