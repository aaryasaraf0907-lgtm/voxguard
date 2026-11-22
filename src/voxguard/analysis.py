"""Analyses from cached scores: per-attack EER, score calibration, a sequential live-alert rule.

Run after ``evaluate`` (which caches eval scores) and ``evaluate --select-thresholds`` (which
caches dev scores next to each checkpoint as ``dev_scores.npz``). Nothing is re-scored:

    python -m voxguard.analysis --config configs/eval_final.yaml

It adds an ``analysis`` block to ``results/<family>.json`` and re-renders ``docs/results.md``.

1. **Per-attack EER** (ASVspoof 2019 LA eval attacks A07-A19, clean and phone-degraded): each
   attack's fakes against all bona fide clips. Most eval attacks use generation methods that are
   not in the training set, so this is the "unseen generation method" number.
2. **Calibration**: a Platt fit, P(fake) = sigmoid(a * logit(score) + b), on the dev phone
   scores. It is monotonic, so it changes no decision; it makes scores readable as
   probabilities. Reported: the expected calibration error (ECE) and Brier score on eval phone,
   before and after, and what the API's phone threshold means as a calibrated probability.
3. **Sequential alert rule** for the live monitor. Calls are simulated from one speaker's clips
   on one codec (ASVspoof clips are about 3-4 s, so each clip stands for one ~4 s window; the
   live monitor's windows overlap, which this does not model). A threshold per window count is
   chosen on dev so that at most ``alpha`` of whole genuine calls ever alert. It is compared on
   eval calls with the fixed rule the API uses (``ALERT_FACTOR`` x the phone threshold, at least
   ``MIN_ALERT_WINDOWS`` windows; ``api/risk.py``) at the same dev false-alarm rate, and with the
   same rule at the threshold itself. The API's rule is also broken down per codec.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from voxguard.api.risk import ALERT_FACTOR, MIN_ALERT_WINDOWS
from voxguard.metrics import compute_eer

logger = logging.getLogger(__name__)

EPS = 1e-7
CALL_WINDOWS = 8  # windows per simulated call (about 16 s of live audio at a 2 s hop)
N_CALLS = 2000  # simulated calls per kind
FIXED_MIN_WINDOWS = MIN_ALERT_WINDOWS  # the live monitor's rule (api/risk.py)
TARGET_ALPHA = 0.01  # call-level false-alarm target for the second sequential rule


# --- ids and protocols -------------------------------------------------------------------------


def file_id(path: str) -> str:
    """``LA_E_1234567`` from a manifest path (clean, or a degraded copy named ``<id>__...``)."""
    return Path(path).name.split("__")[0].split(".")[0]


def load_protocol(path: Path) -> dict[str, tuple[str, str]]:
    """ASVspoof 2019 LA CM protocol: file id -> (speaker, attack); bona fide is ``bonafide``."""
    out = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            speaker, fid, _, attack, _ = parts
            out[fid] = (speaker, "bonafide" if attack == "-" else attack)
    return out


# --- 1. per-attack EER ---------------------------------------------------------------------------


def per_attack_eer(
    scores: np.ndarray, labels: np.ndarray, attacks: np.ndarray
) -> dict[str, dict[str, float]]:
    """EER of each attack's fakes against every bona fide clip."""
    scores, labels, attacks = np.asarray(scores), np.asarray(labels), np.asarray(attacks)
    bona = labels == 0
    out = {}
    for attack in sorted(set(attacks[labels == 1])):
        mask = bona | ((labels == 1) & (attacks == attack))
        eer, _ = compute_eer(scores[mask], labels[mask])
        out[str(attack)] = {"eer": eer, "n_fake": int(((labels == 1) & (attacks == attack)).sum())}
    return out


# --- 2. calibration ------------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + np.tanh(0.5 * np.asarray(x, dtype=np.float64)))  # no overflow


def _logit(s: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(s, dtype=np.float64), EPS, 1 - EPS)
    return np.log(s) - np.log1p(-s)


RIDGE = 1e-4  # on (a, b); keeps the fit finite when dev is (nearly) separable


def _platt_loss(theta: np.ndarray, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    m = x @ theta  # log(1 + e^m) - y * m, computed without overflow
    return float(np.sum(w * (np.logaddexp(0.0, m) - y * m)) + 0.5 * RIDGE * theta @ theta)


def platt_fit(scores: np.ndarray, labels: np.ndarray, iters: int = 100) -> tuple[float, float]:
    """(a, b) maximising the likelihood of sigmoid(a * logit(score) + b): damped Newton steps
    (halved until the loss drops) with a small ridge, so it converges on separable data too.
    Bona fide and fake are weighted equally, so ASVspoof's 1:9 class ratio does not pull the
    calibrated probability."""
    z, y = _logit(scores), np.asarray(labels, dtype=np.float64)
    w = np.where(y == 1, 0.5 / max(y.sum(), 1), 0.5 / max((1 - y).sum(), 1))
    x = np.stack([z, np.ones_like(z)], axis=1)
    theta = np.array([1.0, 0.0])
    loss = _platt_loss(theta, x, y, w)
    for _ in range(iters):
        p = _sigmoid(x @ theta)
        grad = x.T @ (w * (p - y)) + RIDGE * theta
        hess = (x * (w * p * (1 - p))[:, None]).T @ x + RIDGE * np.eye(2)
        step = np.linalg.solve(hess, grad)
        t = 1.0
        while t > 1e-8:
            new = theta - t * step
            new_loss = _platt_loss(new, x, y, w)
            if new_loss <= loss:
                break
            t /= 2
        else:
            break  # no step improves the loss: converged
        theta, done = new, loss - new_loss < 1e-12
        loss = new_loss
        if done:
            break
    return float(theta[0]), float(theta[1])


def platt_apply(scores: np.ndarray, a: float, b: float) -> np.ndarray:
    return _sigmoid(a * _logit(scores) + b)


def calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> dict[str, float]:
    """Class-balanced expected calibration error (equal-width bins) and Brier score."""
    p, y = np.asarray(probs, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    w = np.where(y == 1, 0.5 / max(y.sum(), 1), 0.5 / max((1 - y).sum(), 1))
    idx = np.minimum((p * bins).astype(int), bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += abs(np.average(p[m], weights=w[m]) - np.average(y[m], weights=w[m])) * w[m].sum()
    return {"ece": float(ece), "brier": float(np.sum(w * (p - y) ** 2))}


def calibration_report(
    dev: tuple[np.ndarray, np.ndarray], test: tuple[np.ndarray, np.ndarray], threshold: float
) -> dict[str, Any]:
    a, b = platt_fit(*dev)
    return {
        "a": a,
        "b": b,
        "before": calibration_error(test[0], test[1]),
        "after": calibration_error(platt_apply(test[0], a, b), test[1]),
        "threshold": threshold,
        "threshold_calibrated": float(platt_apply(np.array([threshold]), a, b)[0]),
    }


# --- 3. sequential alert rule --------------------------------------------------------------------


def simulate_calls(
    scores: np.ndarray, groups: np.ndarray, n_calls: int, n_windows: int, seed: int
) -> np.ndarray:
    """(n_calls, n_windows) window scores: each call draws ``n_windows`` different clips of one
    group (a speaker on one codec), chosen among groups with enough clips."""
    rng = np.random.default_rng(seed)
    scores, groups = np.asarray(scores), np.asarray(groups)
    members = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    eligible = [g for g, idx in members.items() if len(idx) >= n_windows]
    if not eligible:
        raise ValueError(f"no group has {n_windows} clips to build a call from")
    picks = rng.choice(len(eligible), size=n_calls)
    return np.stack(
        [scores[rng.choice(members[eligible[i]], size=n_windows, replace=False)] for i in picks]
    )


def running_mean(calls: np.ndarray) -> np.ndarray:
    return np.cumsum(calls, axis=1) / np.arange(1, calls.shape[1] + 1)


def first_alert(running: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Window index (1-based) of each call's first alert, 0 if it never alerts."""
    hit = running >= np.asarray(thresholds)[None, :]
    return np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, 0)


def fixed_rule(
    threshold: float, n_windows: int, min_windows: int = FIXED_MIN_WINDOWS
) -> np.ndarray:
    """The live monitor's rule as per-window thresholds (never before ``min_windows``)."""
    return np.array([np.inf if n < min_windows else threshold for n in range(1, n_windows + 1)])


def sequential_rule(genuine_running: np.ndarray, alpha: float) -> np.ndarray:
    """Per-window thresholds t_n = the (1 - beta) quantile of genuine running means after n
    windows, with beta found by bisection so that at most ``alpha`` of the genuine calls ever
    alert. Early windows, which are noisier, get higher thresholds by themselves."""
    lo, hi = 0.0, alpha
    for _ in range(40):
        beta = (lo + hi) / 2
        t = np.quantile(genuine_running, 1 - beta, axis=0, method="higher")
        if (first_alert(genuine_running, t) > 0).mean() > alpha:
            hi = beta
        else:
            lo = beta
    return np.quantile(genuine_running, 1 - lo, axis=0, method="higher")


def rule_stats(
    thresholds: np.ndarray, genuine: np.ndarray, fake: np.ndarray, mixed: np.ndarray, switch: int
) -> dict[str, float]:
    """Call-level false alarms on genuine calls; detection and median windows to alert on fake
    calls; on mixed calls (genuine for ``switch`` windows, then fake), early false alerts and
    the median windows after the switch."""
    g, f, m = (first_alert(running_mean(c), thresholds) for c in (genuine, fake, mixed))
    after = m[m > switch] - switch
    return {
        "false_alarm_calls": float((g > 0).mean()),
        "fake_detected": float((f > 0).mean()),
        "fake_median_windows": float(np.median(f[f > 0])) if (f > 0).any() else float("nan"),
        "mixed_early_alert": float(((m > 0) & (m <= switch)).mean()),
        "mixed_detected": float((m > switch).mean()),
        "mixed_median_windows_after_switch": float(np.median(after))
        if after.size
        else float("nan"),
    }


def _mixed(genuine: np.ndarray, fake: np.ndarray, switch: int) -> np.ndarray:
    return np.concatenate([genuine[:, :switch], fake[:, : genuine.shape[1] - switch]], axis=1)


def sequential_report(
    dev: dict[str, np.ndarray], test: dict[str, np.ndarray], threshold: float, seed: int = 0
) -> dict[str, Any]:
    """``dev``/``test``: phone ``scores``, ``labels``, ``groups`` (speaker|codec). Mixed calls
    pair a group's genuine windows with fakes of the same group (ASVspoof fakes imitate a target
    speaker)."""

    def calls(d: dict[str, np.ndarray], label: int, s: int) -> np.ndarray:
        m = d["labels"] == label
        return simulate_calls(d["scores"][m], d["groups"][m], N_CALLS, CALL_WINDOWS, s)

    dg = running_mean(calls(dev, 0, seed))
    fixed = fixed_rule(ALERT_FACTOR * threshold, CALL_WINDOWS)
    fixed_dev_alpha = float((first_alert(dg, fixed) > 0).mean())
    rules = {
        "fixed": fixed,
        "fixed_at_threshold": fixed_rule(threshold, CALL_WINDOWS),
        "sequential_same_fa": sequential_rule(dg, fixed_dev_alpha),
        "sequential_1pct": sequential_rule(dg, TARGET_ALPHA),
    }
    tg, tf = calls(test, 0, seed + 1), calls(test, 1, seed + 2)
    switch = CALL_WINDOWS // 2
    tm = _mixed(tg, calls(test, 1, seed + 3), switch)
    per_codec = {}
    codecs = np.array([g.rsplit("|", 1)[-1] for g in test["groups"]])
    for codec in sorted(set(codecs)):  # the API's rule on calls of one codec
        sub = {k: v[codecs == codec] for k, v in test.items()}
        cg, cf = calls(sub, 0, seed + 1), calls(sub, 1, seed + 2)
        per_codec[codec] = {
            "genuine_clips": int((sub["labels"] == 0).sum()),
            **rule_stats(fixed, cg, cf, _mixed(cg, calls(sub, 1, seed + 3), switch), switch),
        }
    return {
        "call_windows": CALL_WINDOWS,
        "n_calls": N_CALLS,
        "switch": switch,
        "alert_factor": ALERT_FACTOR,
        "min_windows": FIXED_MIN_WINDOWS,
        "fixed_dev_false_alarm_calls": fixed_dev_alpha,
        "rules": {
            name: {"thresholds": [float(x) for x in t], **rule_stats(t, tg, tf, tm, switch)}
            for name, t in rules.items()
        },
        "per_codec": per_codec,
    }


# --- driver --------------------------------------------------------------------------------------


def _rows(manifests: list[str], max_samples: int | None, seed: int, drop_clean: bool) -> list[Any]:
    """The rows ``evaluate.score_manifests`` scored, in the same order (no audio is read)."""
    from voxguard.data.datasets import load_rows

    rows = load_rows([Path(m) for m in manifests], None, max_samples, seed)
    return [r for r in rows if not (drop_clean and r.codec == "clean")]


def analyse_model(
    eval_cache: dict[str, Any], eval_rows: dict[str, list[Any]], dev: dict[str, Any] | None,
    protocols: tuple[dict, dict], phone_threshold: float | None,
) -> dict[str, Any]:  # fmt: skip
    """One trained model's analyses. ``eval_cache``: test condition -> npz; ``dev``: its
    ``dev_scores.npz`` or None (then only per-attack EER)."""
    eval_proto, dev_proto = protocols
    out: dict[str, Any] = {"per_attack": {}}
    for test in ("clean", "degraded"):
        z, rows = eval_cache.get(test), eval_rows.get(test)
        if z is None or rows is None:
            continue
        if len(rows) != len(z["scores"]) or not np.array_equal(
            np.array([r.label for r in rows]), z["labels"]
        ):
            raise ValueError(f"{test}: cached scores do not line up with the manifest rows")
        attacks = np.array([eval_proto[file_id(r.path)][1] for r in rows])
        out["per_attack"][test] = per_attack_eer(z["scores"], z["labels"], attacks)
    phone = eval_cache.get("degraded")
    if dev is None or phone is None or phone_threshold is None:
        return out
    rows = eval_rows["degraded"]
    out["calibration"] = calibration_report(
        (dev["phone_scores"], dev["phone_labels"]), (phone["scores"], phone["labels"]),
        phone_threshold,
    )  # fmt: skip

    def grouped(scores, labels, files, codecs, proto) -> dict[str, np.ndarray]:
        groups = np.array(
            [f"{proto[file_id(f)][0]}|{c}" for f, c in zip(files, codecs, strict=True)]
        )
        return {"scores": np.asarray(scores), "labels": np.asarray(labels), "groups": groups}

    out["sequential"] = sequential_report(
        grouped(dev["phone_scores"], dev["phone_labels"], dev["phone_files"], dev["phone_codecs"],
                dev_proto),
        grouped(phone["scores"], phone["labels"], [r.path for r in rows], phone["codecs"],
                eval_proto),
        phone_threshold,
    )  # fmt: skip
    return out


def run(cfg: Any, eval_protocol: Path, dev_protocol: Path) -> dict[str, dict[str, Any]]:
    from voxguard.evaluate import CODEC_TESTS, load_dev_thresholds, scores_root
    from voxguard.plots import scores_path

    protocols = (load_protocol(eval_protocol), load_protocol(dev_protocol))
    out_dir = Path(cfg.out_dir)
    results = {}
    for family, ckpts in cfg.models.items():
        res_path = out_dir / f"{family}.json"
        if not res_path.exists():
            logger.warning("%s: no %s, run evaluate first", family, res_path)
            continue
        res = json.loads(res_path.read_text())
        res["analysis"] = {}
        for train, ckpt in ckpts.items():
            cache, rows = {}, {}
            for test in ("clean", "degraded"):
                p = scores_path(scores_root(cfg), family, train, test)
                if p.exists() and cfg.test_sets.get(test):
                    cache[test] = dict(np.load(p))
                    rows[test] = _rows(cfg.test_sets[test], cfg.max_samples, cfg.seed,
                                       test in CODEC_TESTS)  # fmt: skip
            if not cache:
                continue
            dev_path = Path(ckpt).parent / "dev_scores.npz"
            dev = dict(np.load(dev_path)) if dev_path.exists() else None
            thr = load_dev_thresholds(Path(ckpt)) if Path(ckpt).parent.exists() else None
            if dev is None:
                logger.warning("%s/%s: no %s; per-attack EER only", family, train, dev_path)
            res["analysis"][train] = analyse_model(
                cache, rows, dev, protocols, thr["phone"] if thr else None
            )
            logger.info("%s/%s: analysis done", family, train)
        res_path.write_text(json.dumps(res, indent=2))
        results[family] = res
    return results


def main(argv: list[str] | None = None) -> None:
    from voxguard import evaluate

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    proto = Path("data/raw/LA/ASVspoof2019_LA_cm_protocols")
    p.add_argument("--eval-protocol", type=Path, default=proto / "ASVspoof2019.LA.cm.eval.trl.txt")
    p.add_argument("--dev-protocol", type=Path, default=proto / "ASVspoof2019.LA.cm.dev.trl.txt")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = evaluate.load_eval_config(args.config)
    run(cfg, args.eval_protocol, args.dev_protocol)
    evaluate.main(["--config", str(args.config), "--render-only"])  # docs + figures, as usual


if __name__ == "__main__":
    main()
