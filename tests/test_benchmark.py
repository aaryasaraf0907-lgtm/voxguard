import json
from itertools import count

import numpy as np
import pytest

from voxguard import benchmark
from voxguard.data.codecs import available_codecs, ffmpeg_path
from voxguard.inference import StubDetector


class StepClock:
    """Fake clock: every call advances by the next value of ``steps_s`` (seconds)."""

    def __init__(self, steps_s: list[float]) -> None:
        self.t = 0.0
        self.steps = iter(steps_s)

    def __call__(self) -> float:
        self.t += next(self.steps, 0.0)
        return self.t


def test_timed_runs_discards_warmup_and_times_each_call() -> None:
    calls = count()
    # each timed run: t0 = clock() (+0), end = clock() (+duration)
    clock = StepClock([0, 0.010, 0, 0.020, 0, 0.030])
    samples = benchmark.timed_runs(lambda: next(calls), warmup=2, iterations=3, clock=clock)
    assert next(calls) == 5  # 2 warm-up + 3 timed calls
    assert samples == pytest.approx([10.0, 20.0, 30.0])


@pytest.mark.parametrize(("warmup", "iterations"), [(-1, 5), (0, 0)])
def test_timed_runs_rejects_bad_counts(warmup: int, iterations: int) -> None:
    with pytest.raises(ValueError):
        benchmark.timed_runs(lambda: None, warmup, iterations)


def test_summarize_percentiles_and_extremes() -> None:
    samples = [float(v) for v in range(1, 101)]  # 1..100 ms
    s = benchmark.summarize(samples, warmup=7)
    assert s["n"] == 100 and s["warmup"] == 7
    assert s["p50_ms"] == pytest.approx(np.percentile(samples, 50)) == pytest.approx(50.5)
    assert s["p95_ms"] == pytest.approx(95.05)
    assert (s["min_ms"], s["max_ms"], s["mean_ms"]) == (1.0, 100.0, 50.5)


def test_signal_is_deterministic_and_well_formed() -> None:
    a = benchmark.make_signal(4.0, 16_000, seed=3)
    assert a.dtype == np.float32 and a.shape == (64_000,) and np.isfinite(a).all()
    assert np.array_equal(a, benchmark.make_signal(4.0, 16_000, seed=3))
    assert not np.array_equal(a, benchmark.make_signal(4.0, 16_000, seed=4))
    assert benchmark.make_signal(1.0, 8_000, 0).shape == (8_000,)


def test_detector_benchmark_excludes_warmup_from_reported_latency() -> None:
    class Counting(StubDetector):
        calls = 0

        def predict(self, waveform, sample_rate):
            Counting.calls += 1
            p = super().predict(waveform, sample_rate)
            return p.__class__(p.fake_score, p.label, p.threshold, float(Counting.calls), "c")

    out = benchmark.bench_detector(
        Counting(threshold=0.5), np.zeros(1600, "f4"), 16_000, warmup=3, iterations=4
    )
    assert Counting.calls == 7
    assert out["reported_latency_ms"]["n"] == 4
    assert out["reported_latency_ms"]["min_ms"] == 4.0  # calls 1-3 were warm-up
    assert out["wall"]["n"] == 4 and out["wall"]["warmup"] == 3


def test_mocked_clock_marks_result_mocked_and_records_metadata() -> None:
    det = StubDetector(threshold=0.5)
    r = benchmark.run_benchmark(
        det, detector_kind="stub", device="cpu", load_ms=1.5, seconds=0.5, sample_rate=8_000,
        seed=1, warmup=1, iterations=2, codecs=[], api=False, note="unit",
        clock=StepClock([0.001] * 50),
    )  # fmt: skip
    assert r["mocked"] is True and r["schema"] == benchmark.SCHEMA_VERSION
    assert r["detector"] == {
        "kind": "stub", "model_version": "stub-0", "threshold": 0.5, "device": "cpu",
        "load_ms": 1.5, "weights": "none",
    }  # fmt: skip
    assert r["input"] == {"seconds": 0.5, "sample_rate": 8_000, "seed": 1, "channels": 1}
    assert r["method"]["warmup"] == 1 and r["method"]["iterations"] == 2
    for key in ("platform", "python", "cpu", "cpu_count", "ffmpeg", "git_commit", "torch"):
        assert key in r["environment"]
    assert "codec_preprocessing" not in r and "api_request" not in r
    json.loads(json.dumps(r))  # serializable


def test_real_clock_is_not_mocked() -> None:
    r = benchmark.run_benchmark(
        StubDetector(threshold=0.5), detector_kind="stub", device="cpu", load_ms=None,
        seconds=0.25, sample_rate=16_000, seed=0, warmup=0, iterations=2, codecs=[], api=False,
    )  # fmt: skip
    assert r["mocked"] is False
    assert r["detector_inference"]["wall"]["p50_ms"] >= 0


def test_unavailable_codec_is_reported_as_skipped(monkeypatch) -> None:
    from voxguard.api import codec_sim

    monkeypatch.setattr(codec_sim, "ffmpeg_path", lambda: None)
    out = benchmark.bench_codecs(np.zeros(1600, "f4"), 16_000, ["g711_mulaw"], 0, 1)
    assert "ffmpeg" in out["g711_mulaw"]["skipped"]


@pytest.mark.skipif(
    ffmpeg_path() is None or not available_codecs().get("g711_mulaw"), reason="needs ffmpeg"
)
def test_full_stub_run_with_codec_and_api(tmp_path) -> None:
    r = benchmark.run_benchmark(
        StubDetector(threshold=0.5), detector_kind="stub", device="cpu", load_ms=0.1,
        seconds=1.0, sample_rate=16_000, seed=0, warmup=1, iterations=3,
        codecs=["g711_mulaw", "amr_nb_bogus_is_not_a_codec"][:1], api=True,
    )  # fmt: skip
    assert r["codec_preprocessing"]["g711_mulaw"]["n"] == 3
    api = r["api_request"]
    assert api["codec"] == "g711_mulaw" and api["clean"]["n"] == 3 and api["simulated"]["n"] == 3
    assert api["clean"]["sequential_requests_per_s"] > 0


def test_render_and_load_results_skip_mocked(tmp_path) -> None:
    real = benchmark.run_benchmark(
        StubDetector(threshold=0.5), detector_kind="stub", device="cpu", load_ms=2.0,
        seconds=0.25, sample_rate=16_000, seed=0, warmup=0, iterations=2, codecs=[], api=False,
        note="stub run",
    )  # fmt: skip
    mocked = {**real, "id": "mockedrun", "mocked": True}
    (tmp_path / "a.json").write_text(json.dumps(real))
    (tmp_path / "b.json").write_text(json.dumps(mocked))
    (tmp_path / "c.json").write_text(json.dumps({"schema": 999}))
    loaded = benchmark.load_results(tmp_path)
    assert [r["id"] for r in loaded] == [real["id"]]
    md = benchmark.render_markdown(loaded)
    assert "Auto-generated" in md and real["id"] in md and "stub-0" in md and "stub run" in md
    assert "mockedrun" not in md


def test_cli_writes_json(tmp_path) -> None:
    out = tmp_path / "r.json"
    benchmark.main(
        ["--detector", "stub", "--seconds", "0.25", "--warmup", "0", "--iterations", "2",
         "--no-api", "--out", str(out)]
    )  # fmt: skip
    r = json.loads(out.read_text())
    assert r["detector"]["kind"] == "stub" and r["mocked"] is False
    benchmark.main(["--render", str(tmp_path), "--render-out", str(tmp_path / "b.md")])
    assert r["id"] in (tmp_path / "b.md").read_text()


def test_real_detector_requires_model_repo() -> None:
    with pytest.raises(ValueError, match="model-repo"):
        benchmark.load_benchmark_detector("real", None, "cpu")


def test_untrained_checkpoint_loads_through_real_detector(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    import yaml

    cfg = tmp_path / "rawnet2.yaml"
    cfg.write_text(
        yaml.safe_dump({"model": {"name": "rawnet2", "gru_hidden": 16, "gru_layers": 1}})
    )
    ckpt = benchmark.write_untrained_checkpoint(cfg, tmp_path / "best.pt")
    det, load_ms = benchmark.load_benchmark_detector("real", str(ckpt), "cpu")
    assert det.model_version == f"rawnet2:{ckpt.name}:epoch0" and load_ms > 0
    out = tmp_path / "r.json"
    benchmark.main(
        ["--detector", "real", "--untrained-config", str(cfg), "--seconds", "1", "--warmup", "0",
         "--iterations", "2", "--no-api", "--out", str(out)]
    )  # fmt: skip
    r = json.loads(out.read_text())
    assert r["detector"]["weights"] == "untrained" and r["note"].startswith("UNTRAINED")


def test_untrained_checkpoint_refuses_models_that_need_downloads(tmp_path) -> None:
    pytest.importorskip("torch")
    cfg = tmp_path / "ssl.yaml"
    cfg.write_text("model:\n  name: ssl_aasist\n")
    with pytest.raises(ValueError, match="rawnet2 only"):
        benchmark.write_untrained_checkpoint(cfg, tmp_path / "x.pt")
