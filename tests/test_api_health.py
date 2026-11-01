from fastapi.testclient import TestClient

from voxguard.schemas import HealthResponse


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = HealthResponse.model_validate(r.json())
    assert body.status == "ok" and body.detector == "stub" and body.model_version == "stub-0"
