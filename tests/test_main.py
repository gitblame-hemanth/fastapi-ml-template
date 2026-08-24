"""Tests for the application factory and startup lifecycle."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.core.config import get_settings


def test_create_app_registers_routes() -> None:
    from src.main import create_app

    app = create_app()
    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/health" in paths
    assert "/readiness" in paths
    assert "/liveness" in paths
    assert "/api/v1/predict" in paths
    assert "/api/v1/predict/batch" in paths
    assert "/api/v1/model/info" in paths
    assert "/api/v1/model/reload" in paths


def test_lifespan_loads_model_and_serves_requests(monkeypatch) -> None:
    """Startup loads the registered model; missing Redis degrades to no caching."""
    monkeypatch.setenv("APP_MODEL_NAME", "sklearn_classifier")
    # Point at a closed port so startup falls back to running without cache
    monkeypatch.setenv("APP_REDIS_URL", "redis://localhost:1")
    get_settings.cache_clear()
    try:
        from src.main import create_app

        app = create_app()
        with TestClient(app) as c:
            resp = c.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "healthy"
            assert data["redis_connected"] is False

            resp = c.post("/api/v1/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
            assert resp.status_code == 200
            assert "prediction" in resp.json()
    finally:
        get_settings.cache_clear()
