"""Tests for request-id and rate-limit middleware."""

from __future__ import annotations

import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.middleware import RateLimitMiddleware, RequestIdMiddleware


def test_request_id_generated_when_missing(client: TestClient) -> None:
    """A request without X-Request-ID gets one generated."""
    resp = client.get("/health")
    assert resp.status_code == 200
    rid = resp.headers.get("x-request-id")
    assert rid is not None
    # Should be a valid UUID4
    uuid.UUID(rid, version=4)


def test_request_id_preserved_when_provided(client: TestClient) -> None:
    """A request with X-Request-ID keeps that value."""
    custom_id = "my-custom-request-id-12345"
    resp = client.get("/health", headers={"x-request-id": custom_id})
    assert resp.status_code == 200
    assert resp.headers.get("x-request-id") == custom_id


def test_rate_limit_allows_under_limit(client: TestClient) -> None:
    """Requests under the limit should all succeed."""
    for _ in range(5):
        resp = client.get("/liveness")
        assert resp.status_code == 200


def test_rate_limit_blocks_over_limit(app) -> None:
    """Requests exceeding the limit should get 429."""
    # Build a minimal app with a tiny rate limit
    tiny_app = FastAPI()
    tiny_app.add_middleware(RateLimitMiddleware, max_requests=3, window_seconds=60)
    tiny_app.add_middleware(RequestIdMiddleware)

    # Add a simple endpoint
    @tiny_app.get("/ping")
    async def ping():
        return {"pong": True}

    with TestClient(tiny_app) as c:
        for i in range(3):
            resp = c.get("/ping")
            assert resp.status_code == 200, f"Request {i + 1} should pass"

        resp = c.get("/ping")
        assert resp.status_code == 429
        assert "rate limit" in resp.json()["detail"].lower()
