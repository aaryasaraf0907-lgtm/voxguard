"""Cache frozen XLS-R features (p3's own front-end) for the head-only experiment.

For every clip and condition: load 16 kHz mono, degrade, crop to 64,600 samples, run p3's frozen
front-end (fp16 autocast) and store ``last_hidden_state`` (201 x 1024, fp16) in one memory-mapped
``feats.npy`` per set, with ``meta.csv`` (row, file, label 0 = bona fide / 1 = spoof, group =
attack or "bonafide", condition, copy).

Training copies mirror ``train.py`` with the realistic fine-tune's settings: codec on the whole
file (``codecs.yaml``: random bitrate, noise before the codec on 30 %), seeded random crop, then
``data/realistic.py`` on bona fide and spoof alike (``ssl_aasist_realistic.yaml`` policy). One
deliberate change: Opus is sampled twice as often as each other codec. Evaluation sets: the first
4 s, the default bitrate, no noise and no realistic effects, as ``evaluate.py`` scores them.

Usage: ``python features.py SET`` with SET in train, dev, eval, itw_gate, itw_report.
"""

from __future__ import annotations

import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from voxguard.data.augment import DegradeSpec, Policy, default_spec, degrade, derive_rng  # noqa: E402
from voxguard.data.datasets import fix_length  # noqa: E402
from voxguard.data.realistic import RealisticPolicy, apply_realistic  # noqa: E402

X = Path.home() / "voxguard-private/headonly"
SPLITS = ROOT / "experiments/head_only/splits"
P3 = Path.home() / (
    ".cache/huggingface/hub/models--voxguard-team--voxguard-ssl-aasist-codec-aug/snapshots/"
    "f376f4fde40fa8ac5feff4996e413bf8ff944b13/best.pt"
)
SEED = 20261121
N, FRAMES, DIM = 64_600, 201, 1024
EVAL_CONDITIONS = ("clean", "g711_mulaw", "g711_alaw", "opus")
TRAIN_CODECS = ("g711_mulaw", "g711_alaw", "amr_nb", "amr_wb", "opus")
COPIES = {"bonafide": 4, "spoof": 2, "svarah": 2}  # copy 0 has no codec; the rest get one
REALISTIC = RealisticPolicy(probability=0.5, reverb_probability=0.5, rt60_s=(0.15, 0.8),
                            eq_probability=0.5, eq_max_db=8.0, noise_probability=0.5,
                            snr_db=(10.0, 35.0), gain_db=(-6.0, 6.0))  # fmt: skip
CODEC_POLICY = Policy(codecs={c: (2.0 if c == "opus" else 1.0) for c in TRAIN_CODECS},
                      noise_probability=0.3, snr_db=(10.0, 30.0))  # fmt: skip


def jobs(name: str) -> list[dict]:
    """(path, label, group, condition, copy) for every feature row of a set."""
    out = []
    if name in ("train", "dev", "eval"):
        sub = {"train": "train", "dev": "dev", "eval": "eval"}[name]
        split = {"train": "la_train", "dev": "la_dev_gate", "eval": "la_eval_report"}[name]
        for line in (SPLITS / f"{split}.txt").read_text().split("\n"):
            if not line:
                continue
            fid, group = line.split("\t")
            path = X / f"data/la/LA/ASVspoof2019_LA_{sub}/flac/{fid}.flac"
            label = int(group != "bonafide")
            if name == "train":
                for copy in range(COPIES["spoof" if label else "bonafide"]):
                    out.append(dict(path=path, label=label, group=group, cond="train", copy=copy))
            else:
                for cond in EVAL_CONDITIONS:
                    out.append(dict(path=path, label=label, group=group, cond=cond, copy=0))
        if name == "train":  # Svarah (CC BY 4.0): genuine Indian-accented English, clips >= 2 s
            for path in sorted((X / "data/svarah/wav").glob("*.wav")):
                for copy in range(COPIES["svarah"]):
                    out.append(dict(path=path, label=0, group="svarah", cond="train", copy=copy))
    else:
        for line in (SPLITS / f"{name}.txt").read_text().split("\n"):
            if line:
                fid, lab = line.split("\t")
                spoof = lab == "spoof"
                path = X / f"data/itw/release_in_the_wild/{fid}"
                group = "spoof" if spoof else "bonafide"
                out.append(dict(path=path, label=int(spoof), group=group, cond="clean", copy=0))
    return out


def load(path: Path) -> np.ndarray:
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    x = data.mean(axis=1)
    if sr != 16_000:
        import torch
        import torchaudio.functional as AF

        x = AF.resample(torch.from_numpy(x), sr, 16_000).numpy()
    return x.astype(np.float32)


def prepare(args: tuple[int, dict]) -> np.ndarray:
    """One job -> the 64,600-sample waveform the front-end sees."""
    i, job = args
    x = load(job["path"])
    if job["cond"] == "train":
        key = f"{job['path'].name}:{job['copy']}"
        rng = derive_rng(SEED, key, 0)
        if job["copy"] > 0:
            from voxguard.data.augment import sample_spec

            x = degrade(x, sample_spec(rng, CODEC_POLICY, list(TRAIN_CODECS)), rng)
        x = fix_length(x, N, derive_rng(SEED, key, 1))
        return apply_realistic(x, REALISTIC, derive_rng(SEED, key, 2))
    if job["cond"] != "clean":
        spec = default_spec(job["cond"])
        x = degrade(x, DegradeSpec(spec.codec, spec.bitrate), np.random.default_rng(0))
    return fix_length(x, N)


def main(name: str) -> None:
    import torch

    from voxguard.models import load_from_checkpoint

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    todo = jobs(name)
    smoke = os.environ.get("HEADONLY_SMOKE")  # test run: 48 rows whose audio is already here
    if smoke:
        todo = [j for j in todo if j["path"].exists()][:48]
    out = X / "features" / (f"_smoke_{name}" if smoke else name)
    out.mkdir(parents=True, exist_ok=True)
    # Resumable: rows below the count in progress.txt are already written. Bounded: at most
    # WINDOW prepared clips wait for the GPU (an unbounded pool.map piled up 4.6 GB and the
    # process was OOM-killed on a 7 GB machine, 21 Nov).
    progress, start = out / "progress.txt", 0
    if (out / "feats.npy").exists() and progress.exists():
        start = int(progress.read_text())
        feats = np.load(out / "feats.npy", mmap_mode="r+")
    else:
        feats = np.lib.format.open_memmap(out / "feats.npy", mode="w+", dtype=np.float16,
                                          shape=(len(todo), FRAMES, DIM))  # fmt: skip
    model, _ = load_from_checkpoint(P3, "cuda")
    fe = model.frontend.eval()
    bs, window = 16, 256
    with ProcessPoolExecutor(max_workers=10) as pool:
        nxt = pool.map(prepare, enumerate(todo[start : start + window]), chunksize=4)
        for lo in range(start, len(todo), window):
            xs = list(nxt)  # this window's clips; prepare the next one while the GPU works
            if lo + window < len(todo):
                part = enumerate(todo[lo + window : lo + 2 * window], lo + window)
                nxt = pool.map(prepare, part, chunksize=4)
            for b in range(0, len(xs), bs):
                w = torch.from_numpy(np.stack(xs[b : b + bs])).cuda()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    f = fe(w)
                feats[lo + b : lo + b + len(w)] = f.half().cpu().numpy()
            feats.flush()
            progress.write_text(str(lo + len(xs)))
            if (lo // window) % 8 == 0:
                print(f"{name}: {lo + len(xs)}/{len(todo)}", flush=True)
    with (out / "meta.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "file", "label", "group", "condition", "copy"])
        for i, j in enumerate(todo):
            w.writerow([i, j["path"].name, j["label"], j["group"], j["cond"], j["copy"]])
    print(f"{name}: done, {len(todo)} rows", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
