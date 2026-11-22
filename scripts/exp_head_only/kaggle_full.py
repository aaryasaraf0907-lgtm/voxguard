"""Round 3 (README "Round 3"): full fine-tune of p3 on a Kaggle GPU, with the head-only lessons.

Runs as a Kaggle script kernel. Inputs are three private datasets, found by marker files wherever
Kaggle mounts them: the experiment's code (``pyproject.toml``), the training audio
(``exp-audio.marker``: ASVspoof 2019 LA train FLAC + Svarah genuine WAV) and p3's checkpoint
(``best.pt`` next to ``thresholds.json``).

Each epoch sees every training clip once with **fresh** augmentation, drawn from a seed of
(epoch, clip): a codec on the whole file with probability ``P_CODEC`` (the head-only
``CODEC_POLICY``: Opus weighted 2, noise before the codec on 30 %), a random 64,600-sample crop,
then RawBoost on half the clips (``rawboost``), then ``data/realistic.py`` on both classes.
Loss: inverse-frequency weighted cross-entropy + LAM x KL(p3 || model) at T=2 + MU x the
genuine-margin anchor to p3 (as ``train_head.py``). p3 runs alongside as a frozen fp16 teacher.
The CNN feature encoder stays p3's (frozen); the transformer layers and the back-end train.
Every epoch's weights are saved (fp16) to ``/kaggle/working``; the epoch is chosen later,
locally, by the unchanged dev gate.
``SMOKE=1``: 64 clips, 1 epoch, to check the environment.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

T0 = time.time()
BUDGET_S = float(os.environ.get("BUDGET_S", 10.5 * 3600))  # stop before Kaggle's 12 h limit
SMOKE = os.environ.get("SMOKE") == "1"
SEED = 20261121
N = 64_600
P_CODEC = 0.6
LAM, MU, TEMP = 2.0, 2.0, 2.0
LR_FRONTEND, LR_HEAD = 1e-6, 1e-5
BATCH = 4 if SMOKE else 12
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("runs/kaggle_full")


def find(marker: str) -> Path:
    for root in (Path("/kaggle/input"), Path(".")):
        for p in root.rglob(marker):
            return p.parent
    raise SystemExit(f"input with {marker} not found")


SRC = find("pyproject.toml")
sys.path.insert(0, str(SRC / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from voxguard.data.augment import Policy, degrade, derive_rng, sample_spec  # noqa: E402
from voxguard.data.codecs import usable_codecs  # noqa: E402
from voxguard.data.datasets import fix_length  # noqa: E402
from voxguard.data.realistic import RealisticPolicy, apply_realistic  # noqa: E402
from voxguard.models import load_from_checkpoint  # noqa: E402

P_RAWBOOST = 0.5
REALISTIC = RealisticPolicy(probability=0.5, reverb_probability=0.5, rt60_s=(0.15, 0.8),
                            eq_probability=0.5, eq_max_db=8.0, noise_probability=0.5,
                            snr_db=(10.0, 35.0), gain_db=(-6.0, 6.0))  # fmt: skip
WANTED = ["g711_mulaw", "g711_alaw", "amr_nb", "amr_wb", "opus"]


def _notch_coeffs(rng: np.random.Generator, n_bands: int = 5) -> np.ndarray:
    """A random multi-band FIR filter (RawBoost, Tak et al., ICASSP 2022, eq. 2-3; own code)."""
    from scipy.signal import firwin, freqz

    b = np.array([1.0])
    for _ in range(n_bands):
        fc = rng.uniform(20, 8000)
        bw = rng.uniform(100, 1000)
        c = int(rng.integers(10, 101)) | 1  # odd number of taps
        lo, hi = max(fc - bw / 2, 1.0), min(fc + bw / 2, 7999.0)
        if hi - lo < 10:
            continue
        b = np.convolve(b, firwin(c, [lo, hi], window="hamming", fs=16_000))
    _, h = freqz(b, 1, fs=16_000)
    return b / np.max(np.abs(h))


def rawboost(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """RawBoost's LA setting: linear + non-linear convolutive noise, then impulsive
    signal-dependent noise (Tak, Kamble, Patino, Todisco, Evans, "RawBoost", ICASSP 2022).
    Written from the paper's description; peak-normalised like the original."""
    from scipy.signal import lfilter

    y = np.zeros_like(x, dtype=np.float64)
    for i in range(5):  # N_f = 5 non-linear orders
        g = rng.uniform(-5.0, 0.0) if i == 0 else rng.uniform(-20.0, -5.0)  # dB per order
        y += 10 ** (g / 20) * lfilter(_notch_coeffs(rng), 1, x.astype(np.float64) ** (i + 1))
    y -= y.mean()
    y /= np.max(np.abs(y)) + 1e-9
    n = int(len(y) * rng.uniform(0, 10) / 100)  # impulsive: up to 10 % of samples
    idx = rng.permutation(len(y))[:n]
    f = (2 * rng.random(n) - 1) * (2 * rng.random(n) - 1)
    y[idx] += 2.0 * y[idx] * f
    y /= np.max(np.abs(y)) + 1e-9
    return (y * np.max(np.abs(x))).astype(np.float32)  # keep the clip's original peak level


def log(**kw) -> None:
    kw["t_min"] = round((time.time() - T0) / 60, 1)
    line = json.dumps(kw)
    print(line, flush=True)
    with (OUT / "log.jsonl").open("a") as fh:
        fh.write(line + "\n")


def items(audio: Path) -> list[tuple[str, int]]:
    """(path, label) for every training clip: LA train (label from the key) + Svarah genuine."""
    key = {}
    for line in (audio / "la_train.txt").read_text().split("\n"):
        if line:
            fid, group = line.split("\t")
            key[fid] = int(group != "bonafide")
    la = [(str(audio / "la" / f"{f}.flac"), y) for f, y in sorted(key.items())]
    sv = [(str(p), 0) for p in sorted((audio / "svarah").glob("*.wav"))]
    return la + sv


class Clips(Dataset):
    def __init__(self, rows: list[tuple[str, int]], codecs: list[str]) -> None:
        self.rows, self.codecs, self.epoch = rows, codecs, 0
        self.policy = Policy(codecs={c: (2.0 if c == "opus" else 1.0) for c in codecs},
                             noise_probability=0.3, snr_db=(10.0, 30.0))  # fmt: skip

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        import soundfile as sf

        path, y = self.rows[i]
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        assert sr == 16_000, f"{path}: {sr} Hz"
        key = f"{Path(path).name}:e{self.epoch}"
        rng = derive_rng(SEED, key, 0)
        if self.codecs and rng.random() < P_CODEC:
            x = degrade(x, sample_spec(rng, self.policy, self.codecs), rng)
        x = fix_length(x, N, derive_rng(SEED, key, 1))
        r3 = derive_rng(SEED, key, 3)
        if r3.random() < P_RAWBOOST:
            x = rawboost(x, r3)
        x = apply_realistic(x, REALISTIC, derive_rng(SEED, key, 2))
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), y


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    audio, ckpt = find("exp-audio.marker"), find("thresholds.json") / "best.pt"
    codecs = usable_codecs(WANTED)
    rows = items(audio)
    if SMOKE:
        rows = [rows[i] for i in np.random.default_rng(0).permutation(len(rows))[:64]]
    y = np.array([r[1] for r in rows])
    freq = np.bincount(y, minlength=2) / len(y)
    weight = torch.tensor((1 / freq) / (1 / freq).sum(), dtype=torch.float32, device="cuda")
    log(event="start", rows=len(rows), bonafide=int((y == 0).sum()), spoof=int((y == 1).sum()),
        codecs=codecs, weight=weight.tolist(), gpu=torch.cuda.get_device_name(0))  # fmt: skip

    model, info = load_from_checkpoint(ckpt, "cuda")
    teacher, _ = load_from_checkpoint(ckpt, "cuda")
    teacher = teacher.half().eval()
    for p in teacher.parameters():
        p.requires_grad = False
    model.train()
    for p in model.parameters():
        p.requires_grad = True
    model.frontend.model.gradient_checkpointing_enable()
    for p in model.frontend.model.feature_extractor.parameters():  # CNN encoder stays p3's
        p.requires_grad = False
    front = [p for p in model.frontend.parameters() if p.requires_grad]
    ids = {id(p) for p in front}
    cnn = {id(p) for p in model.frontend.model.feature_extractor.parameters()}
    rest = [p for p in model.parameters() if id(p) not in ids and id(p) not in cnn]
    opt = torch.optim.Adam([{"params": front, "lr": LR_FRONTEND}, {"params": rest, "lr": LR_HEAD}],
                           weight_decay=1e-4)  # fmt: skip
    scaler = torch.amp.GradScaler("cuda")
    data = Clips(rows, codecs)
    epochs = 1 if SMOKE else 99
    epoch_s = None
    for epoch in range(1, epochs + 1):
        if epoch_s and time.time() - T0 + epoch_s > BUDGET_S:
            log(event="stop", reason="time budget", next_epoch=epoch)
            break
        t_ep = time.time()
        data.epoch = epoch
        g = torch.Generator().manual_seed(SEED + epoch)
        loader = DataLoader(data, batch_size=BATCH, shuffle=True, generator=g, num_workers=4,
                            drop_last=True, persistent_workers=False)  # fmt: skip
        total, n = 0.0, 0
        for step, (x, lab) in enumerate(loader):
            x, lab = x.cuda(non_blocking=True), lab.cuda(non_blocking=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                t = teacher(x.half()).float()
            with torch.autocast("cuda", dtype=torch.float16):
                s = model(x).float()
            loss = F.cross_entropy(s, lab, weight=weight)
            loss = loss + LAM * TEMP * TEMP * F.kl_div(
                F.log_softmax(s / TEMP, -1), F.softmax(t / TEMP, -1), reduction="batchmean"
            )
            gen = lab == 0
            if gen.any():
                loss = loss + MU * F.mse_loss(s[gen, 1] - s[gen, 0], t[gen, 1] - t[gen, 0])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total, n = total + float(loss.detach()) * len(lab), n + len(lab)
            if step % 200 == 0:
                log(event="step", epoch=epoch, step=step, of=len(loader), loss=round(total / n, 4))
            if time.time() - T0 > BUDGET_S + 0.5 * 3600:  # hard stop: keep the finished epochs
                log(event="stop", reason="hard limit mid-epoch", epoch=epoch)
                return
        state = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
        torch.save({"model": state, "epoch": epoch, "base": info}, OUT / f"epoch{epoch}.pt")
        epoch_s = time.time() - t_ep
        log(
            event="epoch",
            epoch=epoch,
            loss=round(total / max(n, 1), 4),
            minutes=round(epoch_s / 60, 1),
        )
        model.train()
    log(event="done")


if __name__ == "__main__":
    main()
