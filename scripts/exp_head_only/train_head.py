"""Train the head-only candidate on cached features (see experiments/head_only/README.md).

Loss: class-weighted cross-entropy (weights = inverse class frequency of the training rows, as
p3's 0.9 / 0.1 is for its data) + lam x KL(p3 || head) at temperature 2 on the same features.
After every epoch the dev gate set is scored; the epoch with the fewest phone misses at 1 % false
alarms (then the lowest phone EER) is kept. Usage:
``python train_head.py NAME --lam 0.5 --lr 1e-4 [--epochs 8]``.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from head_lib import FEATS, X, Head, fake_prob, load_p3, load_set, logits, phone_mask, summary

T = 2.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    p3, _ = load_p3()
    teacher_head = Head(p3)
    tr_f, tr_m = load_set("train")
    y = np.array([int(m["label"]) for m in tr_m])
    tpath = FEATS / "train" / "teacher.npy"
    if not tpath.exists():
        np.save(tpath, logits(teacher_head, tr_f))
    teacher = np.load(tpath)
    dev_f, dev_m = load_set("dev")
    dev_y = np.array([int(m["label"]) for m in dev_m])
    ph = phone_mask(dev_m)

    freq = np.bincount(y, minlength=2) / len(y)
    weight = torch.tensor((1 / freq) / (1 / freq).sum(), dtype=torch.float32, device="cuda")

    head = Head(p3).cuda()
    opt = torch.optim.Adam(head.parameters(), lr=a.lr, weight_decay=1e-4)
    out = X / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    hist, best = [], None

    def gate(epoch: int) -> dict:
        s = fake_prob(logits(head, dev_f))
        rec = {"epoch": epoch, "phone": summary(s[ph], dev_y[ph]),
               "clean": summary(s[~ph], dev_y[~ph])}  # fmt: skip
        print(json.dumps({"epoch": epoch, "phone_eer": round(rec["phone"]["eer"] * 100, 3),
                          "phone_miss1": round(rec["phone"]["miss_at_1far"] * 100, 2),
                          "phone_thr": round(rec["phone"]["eer_thr"], 5),
                          "clean_eer": round(rec["clean"]["eer"] * 100, 3)}), flush=True)  # fmt: skip
        return rec

    hist.append(gate(0))  # epoch 0 = p3 itself
    for epoch in range(1, a.epochs + 1):
        head.train()
        order = rng.permutation(len(y))
        t0, total = time.time(), 0.0
        for i in range(0, len(order), a.bs):
            idx = np.sort(order[i : i + a.bs])
            f = torch.from_numpy(np.asarray(tr_f[idx])).cuda()
            lab = torch.from_numpy(y[idx]).cuda()
            t = torch.from_numpy(teacher[idx]).cuda()
            s = head(f)
            loss = F.cross_entropy(s, lab, weight=weight)
            loss = loss + a.lam * T * T * F.kl_div(
                F.log_softmax(s / T, -1), F.softmax(t / T, -1), reduction="batchmean"
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        rec = gate(epoch)
        rec["train_loss"], rec["seconds"] = total / len(y), time.time() - t0
        hist.append(rec)
        key = (rec["phone"]["miss_at_1far"], rec["phone"]["eer"])
        if best is None or key < best[0]:
            best = (key, epoch)
            torch.save(head.state_dict(), out / "best_head.pt")
    (out / "history.json").write_text(json.dumps(
        {"args": vars(a), "class_weight": weight.tolist(), "best_epoch": best[1], "history": hist},
        indent=2))  # fmt: skip
    print(f"best epoch {best[1]}", flush=True)


if __name__ == "__main__":
    main()
