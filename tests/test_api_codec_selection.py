"""Evaluation-matched codec bitrates, codec_ms, request-level codec choice and GET /v1/codecs."""

import time

import numpy as np
import pytest
from sqlalchemy import func, select

from conftest import FixedDetector, make_wav
from voxguard.api import codec_sim
from voxguard.api.db import SessionRow
from voxguard.data import codecs
from voxguard.data.augment import default_spec, degrade
from voxguard.data.codecs import available_codecs, ffmpeg_path


@pytest.fixture
def fake_ffmpeg(monkeypatch):
    """ffmpeg 'present' with every encoder unless a test says otherwise; apply_codec recorded."""
    calls: list[tuple[str, int | None]] = []
    state = {"available": {c: True for c in codecs.CODECS}, "delay_s": 0.0}

    def fake_apply(x, codec, bitrate=None):
        calls.append((codec, bitrate))
        time.sleep(state["delay_s"])
        return (np.asarray(x) * 0.5).astype(np.float32)

    monkeypatch.setattr(codec_sim, "ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(codec_sim, "available_codecs", lambda: dict(state["available"]))
    monkeypatch.setattr(codec_sim, "apply_codec", fake_apply)
    return calls, state


def _post(c, **form):
    return c.post("/v1/analyze", files={"file": ("a.wav", make_wav(0.5))}, data=form)


def _stored(c) -> int:
    with c.app.state.session_factory() as db:
        return db.scalar(select(func.count()).select_from(SessionRow))


# --- evaluation parity ----------------------------------------------------------------------


@pytest.mark.parametrize("codec", list(codecs.CODECS))
def test_demo_uses_the_evaluation_bitrate(codec, fake_ffmpeg) -> None:
    calls, _ = fake_ffmpeg
    codec_sim.simulate_phone_line(np.full(1600, 0.1, np.float32), 16_000, codec)
    assert calls == [(codec, default_spec(codec).bitrate)]


@pytest.mark.parametrize("codec", list(codecs.CODECS))
def test_demo_output_equals_the_evaluation_degradation(codec) -> None:
    """With real ffmpeg: the API's phone line is bit-identical to the eval sets' default spec."""
    if not (ffmpeg_path() and available_codecs().get(codec)):
        pytest.skip(f"ffmpeg lacks {codec}")
    t = np.arange(16_000) / 16_000
    x = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 2500 * t)).astype(np.float32)
    demo, sr = codec_sim.simulate_phone_line(x, 16_000, codec)
    evaluation = degrade(x, default_spec(codec), np.random.default_rng(0))
    assert sr == 16_000 and np.array_equal(demo, evaluation)


# --- codec_ms -------------------------------------------------------------------------------


@pytest.mark.parametrize("simulate", [True, False])
def test_codec_ms_times_only_the_simulation(client_factory, fake_ffmpeg, simulate) -> None:
    fake_ffmpeg[1]["delay_s"] = 0.05
    with client_factory(FixedDetector(0.9), simulate_codec="opus") as c:
        body = _post(c, **({"simulate_codec": "true"} if simulate else {})).json()
    assert body["latency_ms"] == 1.0  # FixedDetector's own latency: the codec time is not in it
    if simulate:
        assert body["codec_applied"] == "opus" and body["codec_ms"] >= 50.0
    else:
        assert body["codec_applied"] is None and body["codec_ms"] is None


# --- request-level codec ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("server_default", "form", "applied", "bitrate"),
    [
        ("g711_mulaw", {"simulate_codec": "true", "codec": "gsm"}, "gsm", 13_000),  # overrides
        (None, {"codec": "opus"}, "opus", 12_000),  # codec alone implies simulation, no 501
        (None, {"codec": "opus", "simulate_codec": "false"}, "opus", 12_000),
        ("g711_alaw", {"simulate_codec": "true"}, "g711_alaw", None),  # env default
        ("opus", {"codec": ""}, None, None),  # FastAPI maps an empty field to None: omitted
    ],
)
def test_codec_selection(client_factory, fake_ffmpeg, server_default, form, applied, bitrate):
    with client_factory(FixedDetector(0.9), simulate_codec=server_default) as c:
        r = _post(c, **form)
    assert r.status_code == 200 and r.json()["codec_applied"] == applied
    assert fake_ffmpeg[0] == ([(applied, bitrate)] if applied else [])


@pytest.mark.parametrize(("name", "status"), [("mp3", 422), ("OPUS", 422), ("gsm", 503)])
def test_bad_codec_is_rejected_without_substitution(client_factory, fake_ffmpeg, name, status):
    fake_ffmpeg[1]["available"]["gsm"] = False  # known codec, missing encoder -> 503
    with client_factory(FixedDetector(0.9), simulate_codec="opus") as c:
        r = _post(c, codec=name)
        assert r.status_code == status and name in r.json()["detail"] and _stored(c) == 0
    assert fake_ffmpeg[0] == []  # the available server default (opus) is not used instead


# --- GET /v1/codecs ---------------------------------------------------------------------------


@pytest.mark.parametrize("default", ["opus", None])
def test_codecs_lists_every_registered_codec(client_factory, fake_ffmpeg, default) -> None:
    fake_ffmpeg[1]["available"].update(amr_nb=False, amr_wb=False)
    with client_factory(simulate_codec=default) as c:
        body = c.get("/v1/codecs").json()
    assert body["default"] == default
    assert [x["name"] for x in body["codecs"]] == list(codecs.CODECS)
    for entry in body["codecs"]:
        assert entry["available"] is fake_ffmpeg[1]["available"][entry["name"]]
        assert entry["bitrate"] == default_spec(entry["name"]).bitrate


def test_codecs_reflect_the_real_ffmpeg(client_factory) -> None:
    with client_factory() as c:
        body = c.get("/v1/codecs").json()
    real = available_codecs() if ffmpeg_path() else {}  # no ffmpeg: every codec unavailable
    assert {e["name"]: e["available"] for e in body["codecs"]} == {
        n: bool(real.get(n)) for n in codecs.CODECS
    }


def test_codecs_all_unavailable_without_ffmpeg(client_factory, monkeypatch) -> None:
    monkeypatch.setattr(codec_sim, "ffmpeg_path", lambda: None)
    with client_factory(simulate_codec="opus") as c:
        assert not any(e["available"] for e in c.get("/v1/codecs").json()["codecs"])
