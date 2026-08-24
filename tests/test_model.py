"""Tests for model management endpoints."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.core.config import get_settings


def test_model_info_returns_metadata(client: TestClient) -> None:
    resp = client.get("/api/v1/model/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "mock-model"
    assert data["version"] == "0.1.0"
    assert data["loaded"] is True
    assert "load_time_seconds" in data


def test_model_reload_open_when_auth_disabled(client: TestClient) -> None:
    """With API key auth disabled (test default), reload needs no header."""
    resp = client.post("/api/v1/model/reload")
    assert resp.status_code == 200
    data = resp.json()
    assert data["loaded"] is True
    assert data["name"] == "mock-model"


def test_model_reload_requires_auth_when_enabled(app: FastAPI, monkeypatch) -> None:
    """With auth enabled, reload rejects missing/invalid keys and accepts the right one."""
    monkeypatch.setenv("APP_API_KEY_ENABLED", "true")
    monkeypatch.setenv("APP_API_KEY", "test-secret-key")
    get_settings.cache_clear()
    try:
        with TestClient(app) as c:
            resp = c.post("/api/v1/model/reload")
            assert resp.status_code == 401

            resp = c.post("/api/v1/model/reload", headers={"X-API-Key": "wrong-key"})
            assert resp.status_code == 403

            resp = c.post(
                "/api/v1/model/reload", headers={"X-API-Key": "test-secret-key"}
            )
            assert resp.status_code == 200
            assert resp.json()["loaded"] is True
    finally:
        get_settings.cache_clear()
