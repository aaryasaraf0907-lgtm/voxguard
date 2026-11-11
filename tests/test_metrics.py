import math

import numpy as np
import pytest

from voxguard.metrics import (
    accuracy_at,
    compute_eer,
    confusion_matrix,
    latency_stats,
    measure_latency,
    summarize,
)


def test_perfect_separation_has_zero_eer() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
    labels = np.array([0, 0, 0, 1, 1, 1])
    eer, thr = compute_eer(scores, labels)
    assert eer == 0.0
    assert 0.3 < thr <= 0.8
    assert accuracy_at(scores, labels, thr) == 1.0


def test_fully_inverted_scores_have_eer_one() -> None:
    assert compute_eer(np.array([0.9, 0.8, 0.1, 0.2]), np.array([0, 0, 1, 1]))[0] == 1.0


def test_constant_scores_have_eer_half() -> None:
    eer, _ = compute_eer(np.full(10, 0.5), np.array([0, 1] * 5))
    assert eer == pytest.approx(0.5)


def test_hand_computed_interpolated_eer() -> None:
    # 2 real, 2 fake. Sorted desc: 0.9(F) 0.6(R) 0.4(F) 0.2(R).
    # Points (fpr,fnr): (0,1) (0,.5) (.5,.5) (.5,0) (1,0); crossing at (0.5,0.5) -> exact hit.
    scores = np.array([0.9, 0.6, 0.4, 0.2])
    labels = np.array([1, 0, 1, 0])
    assert compute_eer(scores, labels)[0] == pytest.approx(0.5)
    # Between points: 3 fake, 1 real. desc: 0.9(F) 0.7(R) 0.5(F) 0.3(F)
    # (fpr,fnr): (0,1) (0,2/3) (1,2/3) (1,1/3) (1,0). d = fnr-fpr: 1, 2/3, -1/3 -> w = 2/3
    # eer = 0 + 2/3 * 1 = 2/3
    eer, thr = compute_eer(np.array([0.9, 0.7, 0.5, 0.3]), np.array([1, 0, 1, 1]))
    assert eer == pytest.approx(2 / 3)
    assert 0.5 < thr < 0.9  # interpolated between the 0.9 and 0.7 thresholds


@pytest.mark.parametrize("shift", [1.0, 2.0, 3.0])
def test_gaussian_eer_matches_theory(shift: float) -> None:
    # real ~ N(0,1), fake ~ N(shift,1) -> EER = Phi(-shift/2)
    rng = np.random.default_rng(0)
    n = 200_000
    scores = np.r_[rng.normal(0, 1, n), rng.normal(shift, 1, n)]
    labels = np.r_[np.zeros(n), np.ones(n)]
    expected = 0.5 * math.erfc(shift / 2 / math.sqrt(2))
    eer, thr = compute_eer(scores, labels)
    assert eer == pytest.approx(expected, abs=0.003)
    assert thr == pytest.approx(shift / 2, abs=0.05)


def test_ties_are_handled() -> None:
    scores = np.array([0.5, 0.5, 0.5, 0.5, 0.9, 0.1])
    labels = np.array([0, 1, 0, 1, 1, 0])
    eer, _ = compute_eer(scores, labels)
    assert 0.0 <= eer <= 1.0


def test_invariant_to_order_and_monotone_transform() -> None:
    rng = np.random.default_rng(1)
    scores = rng.random(500)
    labels = (rng.random(500) < scores).astype(int)
    perm = rng.permutation(500)
    base = compute_eer(scores, labels)[0]
    assert compute_eer(scores[perm], labels[perm])[0] == pytest.approx(base)
    assert compute_eer(scores**3, labels)[0] == pytest.approx(base)


def test_confusion_and_accuracy() -> None:
    scores = np.array([0.9, 0.6, 0.4, 0.2])
    labels = np.array([1, 0, 1, 0])
    assert confusion_matrix(scores, labels, 0.5) == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    assert accuracy_at(scores, labels, 0.5) == 0.5
    assert accuracy_at(scores, labels, 0.0) == 0.5  # all predicted fake
    assert confusion_matrix(scores, labels, 0.9)["tp"] == 1  # score == threshold counts as fake


def test_summarize_keys() -> None:
    out = summarize(np.array([0.1, 0.9, 0.2, 0.8]), np.array([0, 1, 0, 1]))
    assert out["eer"] == 0.0 and out["accuracy_at_0.5"] == 1.0 and out["n"] == 4


def test_validation_errors() -> None:
    with pytest.raises(ValueError):
        compute_eer(np.array([0.1, 0.2]), np.array([0, 0]))
    with pytest.raises(ValueError):
        compute_eer(np.array([0.1, np.nan]), np.array([0, 1]))
    with pytest.raises(ValueError):
        compute_eer(np.array([0.1]), np.array([0, 1]))
    with pytest.raises(ValueError):
        compute_eer(np.array([0.1, 0.2]), np.array([0, 2]))


def test_latency_stats_and_measure() -> None:
    s = latency_stats(list(range(1, 101)))
    assert s.n == 100 and s.p50_ms == pytest.approx(50.5) and s.p95_ms == pytest.approx(95.05)
    calls = []
    stats = measure_latency(lambda: calls.append(1), repeats=5, warmup=2)
    assert len(calls) == 7 and stats.n == 5 and stats.p50_ms >= 0
    with pytest.raises(ValueError):
        latency_stats([])
