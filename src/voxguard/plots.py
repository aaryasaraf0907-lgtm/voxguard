"""Figures for docs/results.md (CLAUDE.md section 10: tables + plots).

Two figures per model family, both generated from evaluate.py output, never by hand:

- **DET curves:** missed fakes against false alarms for the clean and phone-degraded test sets,
  one line per training condition, from the per-clip scores evaluate.py caches.
- **Per-codec EER bars:** clean, each phone codec and the unseen codec(s), from results/*.json.

Needs matplotlib (the ``plots`` extra). Without it the figures are skipped with a warning and
the tables still render.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# validated two-slot categorical palette (dataviz reference instance, light surface)
COLOURS = {"clean": "#2a78d6", "codec_aug": "#eb6834"}
LINESTYLES = {"clean": "--", "codec_aug": "-"}  # second encoding, so colour is never alone
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
LABELS = {"clean": "Clean only", "codec_aug": "Codec-augmented"}
DET_TESTS = {"clean": "Clean test set", "degraded": "Phone-degraded test set"}
CODEC_NAMES = {"amr_nb": "AMR-NB", "amr_wb": "AMR-WB", "g711_alaw": "G.711 A-law",
               "g711_mulaw": "G.711 µ-law", "gsm": "GSM", "opus": "Opus"}  # fmt: skip


def det_points(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(false-alarm rate, miss rate) at every threshold; label 1 = fake, high score = fake."""
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    y = np.asarray(labels)[order]
    n_fake, n_real = int(y.sum()), int(len(y) - y.sum())
    if n_fake == 0 or n_real == 0:
        raise ValueError("DET needs both real and fake clips")
    caught = np.concatenate([[0], np.cumsum(y)])  # fakes at or above each cut
    flagged_real = np.concatenate([[0], np.cumsum(1 - y)])
    return flagged_real / n_real, 1.0 - caught / n_fake


def scores_path(scores_dir: Path, family: str, train: str, test: str) -> Path:
    return Path(scores_dir) / family / f"{train}__{test}.npz"


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def det_figure(family: str, scores_dir: Path, out: Path) -> bool:
    """Write the DET figure; False (nothing written) if no cached scores exist."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True, facecolor=SURFACE)
    drawn = False
    for ax, (test, title) in zip(axes, DET_TESTS.items(), strict=True):
        _style(ax)
        for train in LABELS:
            path = scores_path(scores_dir, family, train, test)
            if not path.exists():
                continue
            d = np.load(path)
            fa, miss = det_points(d["scores"], d["labels"])
            keep = (fa > 0) & (miss > 0)  # log axes
            ax.plot(fa[keep] * 100, miss[keep] * 100, color=COLOURS[train],
                    linestyle=LINESTYLES[train], linewidth=2, label=LABELS[train])  # fmt: skip
            drawn = True
        ax.plot([0.1, 50], [0.1, 50], color=MUTED, linewidth=0.8, linestyle=":")  # EER line
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.1, 50)
        ax.set_ylim(0.1, 50)
        pct = FuncFormatter(lambda v, _: f"{v:g}%")
        ax.xaxis.set_major_formatter(pct)
        ax.yaxis.set_major_formatter(pct)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("False alarms (real clips called fake)", color=INK, fontsize=8)
    axes[0].set_ylabel("Missed fakes", color=INK, fontsize=8)
    if not drawn:
        plt.close(fig)
        return False
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    fig.suptitle(f"{family}: DET curves (lower left is better; dotted line = EER)",
                 color=INK, fontsize=10, x=0.01, ha="left")  # fmt: skip
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


def codec_bars(res: dict[str, Any], out: Path) -> bool:
    """Write grouped EER bars per test condition/codec; False if there is nothing to plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups: list[tuple[str, str, str | None]] = [("Clean", "clean", None)]
    for test, suffix in (("degraded", ""), ("heldout", " (unseen)")):
        codecs: set[str] = set()
        for train in LABELS:
            cond = (res.get("matrix", {}).get(train) or {}).get(test) or {}
            codecs |= set(cond.get("per_codec", {}))
        groups += [(f"{CODEC_NAMES.get(c, c)}{suffix}", test, c) for c in sorted(codecs)]

    def eer(train: str, test: str, codec: str | None) -> float | None:
        cond = (res.get("matrix", {}).get(train) or {}).get(test)
        if not cond:
            return None
        stats = cond["pooled"] if codec is None else cond.get("per_codec", {}).get(codec)
        return None if stats is None else 100 * stats["eer"]

    values = {t: [eer(t, test, c) for _, test, c in groups] for t in LABELS}
    if all(v is None for vs in values.values() for v in vs):
        return False
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=SURFACE)
    _style(ax)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.grid(False, axis="x")
    x = np.arange(len(groups))
    width = 0.38
    for i, train in enumerate(LABELS):
        xs = x + (i - 0.5) * width
        for xi, v in zip(xs, values[train], strict=True):
            if v is None:
                continue
            ax.bar(xi, v, width=width - 0.03, color=COLOURS[train],
                   label=LABELS[train] if xi == xs[0] else None)  # fmt: skip
            ax.annotate(f"{v:.1f}", (xi, v), xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=7, color=INK)  # fmt: skip
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOURS[t]) for t in LABELS]
    ax.legend(handles, LABELS.values(), frameon=False, fontsize=8, labelcolor=INK)
    ax.set_xticks(x, [g[0] for g in groups], fontsize=8)
    ax.set_ylabel("EER (%)", color=INK, fontsize=8)
    ax.set_title(f"{res.get('family', '')}: EER by test condition (lower is better)",
                 color=INK, fontsize=10, loc="left")  # fmt: skip
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


def make_figures(
    families: dict[str, dict[str, Any]], scores_dir: Path, fig_dir: Path, rel_prefix: str
) -> dict[str, list[tuple[str, str]]]:
    """Write every family's figures; returns family -> [(alt text, path relative to the doc)]."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        logger.warning("matplotlib not installed (the 'plots' extra): figures skipped")
        return {}
    out: dict[str, list[tuple[str, str]]] = {}
    for family, res in families.items():
        figs = []
        bars = fig_dir / f"{family}_eer_by_condition.png"
        if codec_bars(res, bars):
            figs.append((f"{family} EER by test condition", f"{rel_prefix}{bars.name}"))
        det = fig_dir / f"{family}_det.png"
        if det_figure(family, scores_dir, det):
            figs.append((f"{family} DET curves", f"{rel_prefix}{det.name}"))
        else:
            logger.warning("%s: no cached scores in %s, DET figure skipped", family, scores_dir)
        out[family] = figs
    return out
