"""Tests for request-id and rate-limit middleware."""

from __future__ import annotations

import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.middleware import RateLimitMiddleware, RequestIDMiddleware
from src.core.config import get_settings


def _tiny_app() -> FastAPI:
    """Build a minimal app wired with the same middleware as ``src.main``."""
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestIDMiddleware)

    @app.get("/ping")
    async def ping() -> dict[str, bool]:
        return {"pong": True}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy"}

    return app


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


def test_rate_limit_blocks_over_limit(monkeypatch) -> None:
    """Requests exceeding the limit get 429 with a Retry-After header."""
    monkeypatch.setenv("APP_RATE_LIMIT_REQUESTS", "3")
    monkeypatch.setenv("APP_RATE_LIMIT_WINDOW", "60")
    get_settings.cache_clear()
    try:
        with TestClient(_tiny_app()) as c:
            for i in range(3):
                resp = c.get("/ping")
                assert resp.status_code == 200, f"Request {i + 1} should pass"

            resp = c.get("/ping")
            assert resp.status_code == 429
            assert "rate limit" in resp.json()["detail"].lower()
            assert "retry-after" in resp.headers
            assert int(resp.headers["retry-after"]) >= 1
    finally:
        get_settings.cache_clear()


def test_rate_limit_exempts_health_endpoints(monkeypatch) -> None:
    """Health endpoints are never rate limited."""
    monkeypatch.setenv("APP_RATE_LIMIT_REQUESTS", "1")
    monkeypatch.setenv("APP_RATE_LIMIT_WINDOW", "60")
    get_settings.cache_clear()
    try:
        with TestClient(_tiny_app()) as c:
            # Exhaust the limit on a normal endpoint
            assert c.get("/ping").status_code == 200
            assert c.get("/ping").status_code == 429

            # Health endpoint stays open regardless
            for _ in range(5):
                assert c.get("/health").status_code == 200
    finally:
        get_settings.cache_clear()
