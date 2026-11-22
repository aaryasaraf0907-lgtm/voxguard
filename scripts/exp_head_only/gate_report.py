"""Apply the pre-registered gate, then the report, to p3 and the candidates.

Candidates: ``A:<run>`` (a trained head) and ``B:<run>`` (the mean of p3's and that head's fake
scores). The gate uses the dev gate set and the In-the-Wild gate part only; the eval report
subset and the In-the-Wild report part are scored for the report, for the candidates named, once.
Usage: ``python gate_report.py RUN [RUN ...] [--report]``. Writes ``runs/gate[-report].json/.md``.
"""

from __future__ import annotations  # noqa: I001

import argparse
import json
from collections import defaultdict

import numpy as np
import torch
from head_lib import X, Head, fake_prob, load_p3, load_set, phone_mask, summary

WATCH = ("A10", "A11", "A17", "A18")


def scorers(runs: list[str]) -> dict:
    p3, _ = load_p3()
    out = {"p3": Head(p3)}
    for r in runs:
        h = Head(p3)
        h.load_state_dict(torch.load(X / "runs" / r / "best_head.pt", map_location="cpu"))
        out[f"A:{r}"] = h
    return out


@torch.no_grad()
def scores_for(heads: dict, set_name: str, bs: int = 256) -> tuple[dict, list[dict]]:
    """Every head's fake scores; each feature batch is read from disk once, for all heads."""
    f, meta = load_set(set_name)
    for h in heads.values():
        h.eval().cuda()
    out: dict = {k: [] for k in heads}
    for i in range(0, len(f), bs):
        x = torch.from_numpy(np.asarray(f[i : i + bs])).cuda()
        for k, h in heads.items():
            out[k].append(h(x).float().cpu())
    s = {k: fake_prob(torch.cat(v).numpy()) for k, v in out.items()}
    for k in [k for k in s if k.startswith("A:")]:
        s["B:" + k[2:]] = (s["p3"] + s[k]) / 2
    return s, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    heads = scorers(a.runs)
    res: dict = {"gate": {}, "report": {}}

    dev, dm = scores_for(heads, "dev")
    y = np.array([int(m["label"]) for m in dm])
    ph = phone_mask(dm)
    thr = {k: summary(s[ph], y[ph])["eer_thr"] for k, s in dev.items()}
    itw, im = scores_for(heads, "itw_gate")
    iy = np.array([int(m["label"]) for m in im])
    for k in dev:
        d = summary(dev[k][ph], y[ph])
        i = summary(itw[k], iy, thr[k])
        res["gate"][k] = {"dev_phone": d, "dev_clean": summary(dev[k][~ph], y[~ph]), "itw_gate": i}
    base = res["gate"]["p3"]
    for k, g in res["gate"].items():
        if k == "p3":
            continue
        checks = {
            "dev phone EER <= p3 + 0.10 pp": g["dev_phone"]["eer"] <= base["dev_phone"]["eer"] + 0.001,  # noqa: E501
            "dev phone miss@1%FAR <= p3": g["dev_phone"]["miss_at_1far"] <= base["dev_phone"]["miss_at_1far"],  # noqa: E501
            "dev phone threshold within 1.5x of p3": 1 / 1.5 <= thr[k] / thr["p3"] <= 1.5,
            "ITW gate EER <= p3": g["itw_gate"]["eer"] <= base["itw_gate"]["eer"],
            "ITW gate FA at own dev thr <= p3": g["itw_gate"]["far_at_thr"] <= base["itw_gate"]["far_at_thr"],  # noqa: E501
        }  # fmt: skip
        g["checks"], g["passed"] = checks, all(checks.values())

    if a.report:
        ev, em = scores_for(heads, "eval")
        ey = np.array([int(m["label"]) for m in em])
        cond = np.array([m["condition"] for m in em])
        grp = np.array([m["group"] for m in em])
        for k in ev:
            r: dict = {"per_condition": {}, "per_attack": defaultdict(dict)}
            for c in sorted(set(cond)):
                mk = cond == c
                r["per_condition"][c] = summary(ev[k][mk], ey[mk], thr[k])
            phone = np.isin(cond, ["g711_mulaw", "g711_alaw", "opus"])
            r["phone"] = summary(ev[k][phone], ey[phone], thr[k])
            for att in sorted(set(grp) - {"bonafide"}):
                mk = phone & ((grp == att) | (grp == "bonafide"))
                r["per_attack"][att] = summary(ev[k][mk], ey[mk], thr[k])
            res["report"][k] = r
        it, rm = scores_for(heads, "itw_report")
        ry = np.array([int(m["label"]) for m in rm])
        for k in it:
            res["report"][k]["itw_report"] = summary(it[k], ry, thr[k])
        for k in res["report"]:
            if k == "p3":
                continue
            pa, pb = res["report"][k]["per_attack"], res["report"]["p3"]["per_attack"]
            worse = {att: pa[att]["miss_at_1far"] - pb[att]["miss_at_1far"] for att in WATCH}
            res["report"][k]["stop_rule"] = {"rise_pp": {a_: round(v * 100, 2) for a_, v in worse.items()},  # noqa: E501
                                             "triggered": any(v > 0.05 for v in worse.values())}  # fmt: skip  # noqa: E501
    res["dev_phone_thresholds"] = thr
    name = "gate-report" if a.report else "gate"
    (X / "runs" / f"{name}.json").write_text(json.dumps(res, indent=2, default=float))
    for k, g in res["gate"].items():
        line = (f"{k:24s} dev phone EER {g['dev_phone']['eer'] * 100:.3f}% "
                f"miss@1%FA {g['dev_phone']['miss_at_1far'] * 100:.2f}% thr {thr[k]:.5f} | "
                f"ITW gate EER {g['itw_gate']['eer'] * 100:.2f}% FA@thr {g['itw_gate']['far_at_thr'] * 100:.2f}%")  # fmt: skip  # noqa: E501
        print(line + ("" if k == "p3" else f" | {'PASS' if g['passed'] else 'fail'}"), flush=True)


if __name__ == "__main__":
    main()
