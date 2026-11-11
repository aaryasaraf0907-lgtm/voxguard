from fastapi.testclient import TestClient

from voxguard.schemas import HealthResponse


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = HealthResponse.model_validate(r.json())
    assert body.status == "ok" and body.detector == "stub" and body.model_version == "stub-0"


def test_missing_ml_extra_gives_an_actionable_startup_error(monkeypatch) -> None:
    import sys

    import pytest

    from voxguard.api import main
    from voxguard.api.deps import Settings

    monkeypatch.setenv("VOXGUARD_DETECTOR", "real")
    monkeypatch.setenv("VOXGUARD_MODEL_REPO", "/models/best.pt")
    monkeypatch.setitem(sys.modules, "torch", None)  # behaves as if torch were not installed
    app = main.create_app(settings=Settings(db_url="sqlite://"))
    with pytest.raises(RuntimeError, match="INSTALL_ML=1") as exc, TestClient(app):
        pass
    assert "'torch'" in str(exc.value) and "ml" in str(exc.value)


def test_other_detector_errors_are_not_rewritten(monkeypatch) -> None:
    import pytest

    from voxguard.api import main
    from voxguard.api.deps import Settings

    monkeypatch.setenv("VOXGUARD_DETECTOR", "bogus")
    with (
        pytest.raises(ValueError, match="VOXGUARD_DETECTOR"),
        TestClient(main.create_app(settings=Settings(db_url="sqlite://"))),
    ):
        pass
