"""Score a round-3 checkpoint (a full fine-tune saved by ``kaggle_full.py``) on the gate sets.

The same clips and conditions as ``features.py`` (dev gate set: clean, G.711 µ/A-law, Opus;
In-the-Wild gate part: clean), prepared by the same ``prepare`` function, but run through the
whole fine-tuned model (front-end included), since its front-end is no longer p3's. Writes
``<out>/<tag>_<set>.npy`` (P(fake) per row, in ``jobs`` order) and prints the gate numbers next
to p3's (from ``runs/gate.json``). Usage::

    python score_full.py CHECKPOINT TAG [--sets dev itw_gate] [--out DIR]

CHECKPOINT is ``epoch<N>.pt`` or ``ema_last.pt`` from ``kaggle_full.py`` (fp16 weights of p3's
architecture). Nothing here is used for training; it only reads the gate sets.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from features import P3, X, jobs, prepare
from head_lib import miss_at_far, summary

PHONE = ("g711_mulaw", "g711_alaw", "opus")


def load(ckpt: Path):
    from voxguard.models import load_from_checkpoint

    model, _ = load_from_checkpoint(P3, "cuda")
    state = torch.load(ckpt, map_location="cuda")["model"]
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    return model.eval().half()  # fp16 weights: fits next to a training run on a 6 GB GPU


@torch.no_grad()
def score(model, name: str, step: int = 1, workers: int = 4, bs: int = 8):
    """``step`` > 1: every step-th row only (a fixed preview subset, same for every checkpoint)."""
    todo = jobs(name)[::step]
    out = np.zeros(len(todo), dtype=np.float32)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for lo in range(0, len(todo), 256):
            xs = list(pool.map(prepare, enumerate(todo[lo : lo + 256]), chunksize=4))
            for b in range(0, len(xs), bs):
                x = torch.from_numpy(np.stack(xs[b : b + bs])).cuda()
                with torch.autocast("cuda", dtype=torch.float16):
                    p = torch.softmax(model(x.half()).float(), -1)[:, 1]
                out[lo + b : lo + b + len(x)] = p.cpu().numpy()
    return out, todo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("tag")
    ap.add_argument("--sets", nargs="+", default=["dev", "itw_gate"])
    ap.add_argument("--out", type=Path, default=X / "runs" / "round3_scores")
    ap.add_argument("--preview", type=int, default=1, help="score every N-th row only")
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    a.out.mkdir(parents=True, exist_ok=True)
    model = load(a.checkpoint)
    res: dict = {"tag": a.tag, "checkpoint": str(a.checkpoint)}
    for name in a.sets:
        sfx = f"_preview{a.preview}" if a.preview > 1 else ""
        path = a.out / f"{a.tag}_{name}{sfx}.npy"
        todo = jobs(name)[:: a.preview]
        if path.exists():
            s = np.load(path)
        else:
            s, todo = score(model, name, a.preview)
            np.save(path, s)
        y = np.array([j["label"] for j in todo])
        cond = np.array([j["cond"] for j in todo])
        if name == "dev":
            ph = np.isin(cond, PHONE)
            d = summary(s[ph], y[ph])
            res["dev_phone"] = d
            res["dev_thr"] = d["eer_thr"]
        else:
            res[name] = summary(s, y, res.get("dev_thr"))
    p3 = json.loads((X / "runs" / "gate.json").read_text())
    g3, thr3 = p3["gate"]["p3"], p3["dev_phone_thresholds"]["p3"]
    line = {"tag": a.tag}
    if "dev_phone" in res:
        d = res["dev_phone"]
        line |= {
            "dev_phone_eer": [round(d["eer"] * 100, 3), round(g3["dev_phone"]["eer"] * 100, 3)],
            "dev_miss_1far": [round(d["miss_at_1far"] * 100, 2),
                              round(g3["dev_phone"]["miss_at_1far"] * 100, 2)],
            "thr_x_p3": round(d["eer_thr"] / thr3, 2),
        }  # fmt: skip
    if "itw_gate" in res:
        i = res["itw_gate"]
        line |= {
            "itw_eer": [round(i["eer"] * 100, 2), round(g3["itw_gate"]["eer"] * 100, 2)],
            "itw_fa_at_thr": [round(i["far_at_thr"] * 100, 2),
                              round(g3["itw_gate"]["far_at_thr"] * 100, 2)],
        }  # fmt: skip
    line["preview_every"] = a.preview
    sfx = f"_preview{a.preview}" if a.preview > 1 else ""
    (a.out / f"{a.tag}{sfx}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(line), flush=True)  # [this checkpoint, p3]


if __name__ == "__main__":
    _ = miss_at_far  # same metric helpers as the gate
    main()
