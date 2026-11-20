"""Local operating-point advice from a real-audio sanity report (``real_audio_sanity.py advise``).

The decision threshold ships with the checkpoint (``thresholds.json``, chosen on the ASVspoof
development set). A deployment's own callers (other microphones, rooms, languages, phone lines)
can sit closer to it than the development set did. This reads a sanity report of the deployer's
own **consented genuine recordings** (optionally with known fakes) and says, per condition (as
uploaded, each codec):

- the local false-alarm count at the current threshold, with an exact one-sided 95 % upper bound
  on the false-alarm rate (Clopper-Pearson), so a small set is not mistaken for a guarantee;
- for each target false-alarm rate, the lowest threshold at or above the current one whose 95 %
  upper bound meets it (it never suggests lowering the shipped threshold), or how many genuine
  recordings are needed before any threshold can (zero false alarms on N recordings
  supports a rate of at most 1 - 0.05^(1/N), about 3/N);
- when fakes are included, how many each threshold would miss.

It only recommends; nothing changes until someone sets ``VOXGUARD_THRESHOLD``. Raising the
threshold trades missed fakes for fewer false alarms, so the report always shows both sides.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

CONFIDENCE = 0.95
DEFAULT_TARGETS = (0.01, 0.05)


def binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def far_upper_bound(k: int, n: int, confidence: float = CONFIDENCE) -> float:
    """Exact one-sided upper bound on a rate from k events in n trials (Clopper-Pearson)."""
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(100):  # bisection: P(X <= k; n, p) = 1 - confidence
        mid = (lo + hi) / 2
        if binom_cdf(k, n, mid) > 1 - confidence:
            lo = mid
        else:
            hi = mid
    return hi


def clips_needed(target: float, confidence: float = CONFIDENCE) -> int:
    """Genuine recordings needed so that zero false alarms supports ``target`` at ``confidence``."""
    return math.ceil(math.log(1 - confidence) / math.log(1 - target))


def advise_condition(
    genuine: list[float], fakes: list[float], threshold: float, targets: tuple[float, ...]
) -> dict[str, Any]:
    n = len(genuine)
    k = sum(s >= threshold for s in genuine)
    out: dict[str, Any] = {
        "genuine": n,
        "fakes": len(fakes),
        "threshold": threshold,
        "false_alarms": k,
        "far_upper_95": round(far_upper_bound(k, n), 4),
        "missed_fakes": sum(s < threshold for s in fakes) if fakes else None,
        "targets": [],
    }
    # candidates: the current threshold, and just above each genuine score over it. Never lower:
    # the shipped threshold was chosen with fakes on the development set, and a lower one fitted
    # to a few local genuine recordings would be overfitted to them.
    candidates = sorted(
        {threshold, *(math.nextafter(s, math.inf) for s in genuine if s >= threshold)}
    )
    for target in targets:
        row: dict[str, Any] = {"target": target, "clips_needed": clips_needed(target)}
        fit = [t for t in candidates if far_upper_bound(sum(s >= t for s in genuine), n) <= target]
        if not fit:
            row["suggested"] = None  # too few recordings for this target at any threshold
        else:
            best = fit[0]
            row["suggested"] = best
            row["keeps_current"] = best <= threshold
            row["false_alarms"] = sum(s >= best for s in genuine)
            row["missed_fakes"] = sum(s < best for s in fakes) if fakes else None
        out["targets"].append(row)
    return out


def advise(report: dict[str, Any], targets: tuple[float, ...] = DEFAULT_TARGETS) -> dict[str, Any]:
    by_condition: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    thresholds: dict[str, float] = {}
    for r in report["results"]:
        if r.get("outcome") != "ok" or r.get("fake_score") is None:
            continue
        cond = r.get("codec_requested") or "as uploaded"
        by_condition[cond][r["ground_truth"]].append(r["fake_score"])
        thresholds[cond] = r["threshold"]
    if not by_condition:
        raise ValueError("the report has no scored recordings")
    conditions = {
        cond: advise_condition(
            groups.get("genuine", []), groups.get("spoof", []), thresholds[cond], targets
        )
        for cond, groups in sorted(by_condition.items())
    }
    run = report.get("run", {})
    return {
        "model_version": (run.get("health") or {}).get("model_version"),
        "source_report": run.get("started_utc"),
        "confidence": CONFIDENCE,
        "conditions": conditions,
    }


def render_markdown(advice: dict[str, Any]) -> str:
    lines = [
        "# Local operating-point advice (recommendation only)",
        "",
        f"Model `{advice['model_version']}`, from the sanity report of {advice['source_report']}. "
        "Bounds are exact one-sided 95 % upper bounds (Clopper-Pearson) on the false-alarm rate "
        "for this deployment's recordings. Nothing changes until someone sets "
        "`VOXGUARD_THRESHOLD`; the shipped threshold comes from `thresholds.json`.",
        "",
    ]
    for cond, c in advice["conditions"].items():
        missed = (
            ""
            if c["missed_fakes"] is None
            else f", {c['missed_fakes']} of {c['fakes']} fakes missed"
        )
        lines += [
            f"## {cond}",
            "",
            f"Current threshold {c['threshold']:.6f}: {c['false_alarms']} of {c['genuine']} "
            f"genuine recordings flagged{missed}. False-alarm rate here: at most "
            f"{c['far_upper_95']:.1%} (95 %).",
            "",
            "| Target false-alarm rate | Suggested threshold | Genuine flagged | Fakes missed |",
            "|---|---|---|---|",
        ]
        for t in c["targets"]:
            if t["suggested"] is None:
                lines.append(
                    f"| {t['target']:.0%} | not enough recordings: need at least "
                    f"{t['clips_needed']} genuine ones | – | – |"
                )
                continue
            keep = " (the current one)" if t["keeps_current"] else ""
            fm = "–" if t["missed_fakes"] is None else f"{t['missed_fakes']} of {c['fakes']}"
            lines.append(
                f"| {t['target']:.0%} | {t['suggested']:.6f}{keep} | "
                f"{t['false_alarms']} of {c['genuine']} | {fm} |"
            )
        lines.append("")
    lines.append(
        "A higher threshold means fewer false alarms and more missed fakes. Recordings must be "
        "consented, from this deployment's own conditions, and the fakes here are not proof of "
        "how real attacks will score."
    )
    return "\n".join(lines) + "\n"


def load_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or "results" not in data:
        raise ValueError(f"{path.name} is not a sanity report")
    return data
