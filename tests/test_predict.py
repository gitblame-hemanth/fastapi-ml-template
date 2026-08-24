"""Tests for prediction endpoints."""

from __future__ import annotations

import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.core.config import get_settings


def test_predict_single_success(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert data["model_name"] == "mock-model"


def test_predict_returns_request_id(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/predict",
        json={"features": [1.0, 2.0]},
    )
    assert resp.status_code == 200
    # The RequestIDMiddleware sets x-request-id on the response header
    assert "x-request-id" in resp.headers


def test_predict_invalid_input(client: TestClient) -> None:
    # Neither features nor text provided — should still be 200 per schema
    # (both are optional) but _build_input raises 422
    resp = client.post("/api/v1/predict", json={})
    assert resp.status_code == 422


def test_predict_batch_success(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/predict/batch",
        json={
            "inputs": [
                {"features": [5.1, 3.5, 1.4, 0.2]},
                {"features": [6.2, 2.9, 4.3, 1.3]},
            ]
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["predictions"]) == 2
    assert data["model_name"] == "mock-model"


def test_predict_batch_empty_list(client: TestClient) -> None:
    resp = client.post("/api/v1/predict/batch", json={"inputs": []})
    assert resp.status_code == 422


def test_predict_batch_too_many(client: TestClient) -> None:
    # max_length=100 per schema
    inputs = [{"features": [1.0]} for _ in range(101)]
    resp = client.post("/api/v1/predict/batch", json={"inputs": inputs})
    assert resp.status_code == 422


def test_predict_response_contains_model_metadata(client: TestClient) -> None:
    """The response carries the prediction plus model name/version and cache flag."""
    resp = client.post(
        "/api/v1/predict",
        json={"features": [1.0, 2.0]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert data["model_name"] == "mock-model"
    assert data["model_version"] == "0.1.0"
    assert data["cached"] is False


def test_predict_requires_auth_when_enabled(app: FastAPI, monkeypatch) -> None:
    """With auth enabled, predict rejects missing keys and accepts the right one."""
    monkeypatch.setenv("APP_API_KEY_ENABLED", "true")
    monkeypatch.setenv("APP_API_KEY", "test-secret-key")
    get_settings.cache_clear()
    try:
        with TestClient(app) as c:
            resp = c.post("/api/v1/predict", json={"features": [1.0, 2.0]})
            assert resp.status_code == 401

            resp = c.post(
                "/api/v1/predict",
                json={"features": [1.0, 2.0]},
                headers={"X-API-Key": "test-secret-key"},
            )
            assert resp.status_code == 200
    finally:
        get_settings.cache_clear()


def test_predict_times_out_when_model_is_slow(app: FastAPI, monkeypatch) -> None:
    """Inference slower than APP_INFERENCE_TIMEOUT returns 504."""
    from tests.conftest import MockModel

    class SlowModel(MockModel):
        def predict(self, input_data):
            time.sleep(0.5)
            return super().predict(input_data)

    slow = SlowModel()
    slow.load()
    app.state.model = slow

    monkeypatch.setenv("APP_INFERENCE_TIMEOUT", "0.1")
    get_settings.cache_clear()
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/api/v1/predict", json={"features": [1.0, 2.0]})
            assert resp.status_code == 504
            assert "timeout" in resp.json()["detail"].lower()

            resp = c.post(
                "/api/v1/predict/batch", json={"inputs": [{"features": [1.0]}]}
            )
            assert resp.status_code == 504
            assert "timeout" in resp.json()["detail"].lower()
    finally:
        get_settings.cache_clear()
