"""Tests for health check endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "model" in data
    assert "redis_connected" in data


def test_ready_when_model_loaded(client: TestClient) -> None:
    resp = client.get("/readiness")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"


def test_ready_when_model_not_loaded(app, mock_redis: AsyncMock) -> None:
    """Readiness should return 503 when model is not loaded."""
    from tests.conftest import MockModel

    unloaded = MockModel()  # _loaded = False
    app.state.model = unloaded
    app.state.redis = mock_redis

    with TestClient(app, raise_server_exceptions=False) as c:
        resp = c.get("/readiness")

    # The route returns JSONResponse with 503 directly
    assert resp.status_code == 503
    data = resp.json()
    assert data["status"] == "not_ready"


def test_live_always_ok(client: TestClient) -> None:
    resp = client.get("/liveness")
    assert resp.status_code == 200
    assert resp.json()["status"] == "alive"
