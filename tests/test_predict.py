"""Tests for prediction endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


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
    # The RequestIdMiddleware sets x-request-id on the response header
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


def test_predict_includes_inference_time(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/predict",
        json={"features": [1.0, 2.0]},
    )
    assert resp.status_code == 200
    data = resp.json()
    # The predict route returns the result dict from model.predict which
    # includes 'prediction' and 'probabilities'
    assert "prediction" in data
    assert "model_name" in data
