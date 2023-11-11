"""Shared pytest fixtures for the FastAPI ML template test suite."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.models.base import BaseModel as MLBaseModel

# ---------------------------------------------------------------------------
# Environment — set BEFORE any settings import so pydantic-settings picks them up
# ---------------------------------------------------------------------------

os.environ.setdefault("APP_ENVIRONMENT", "test")
os.environ.setdefault("APP_DEBUG", "true")
os.environ.setdefault("APP_LOG_LEVEL", "DEBUG")
os.environ.setdefault("APP_API_KEY", "test-secret-key")
os.environ.setdefault("APP_API_KEY_ENABLED", "false")
os.environ.setdefault("APP_RATE_LIMIT_REQUESTS", "1000")
os.environ.setdefault("APP_RATE_LIMIT_WINDOW", "60")


# ---------------------------------------------------------------------------
# Mock ML model
# ---------------------------------------------------------------------------


class MockModel(MLBaseModel):
    """Concrete implementation of BaseModel for testing."""

    @property
    def name(self) -> str:
        return "mock-model"

    @property
    def version(self) -> str:
        return "0.1.0"

    def load(self) -> None:
        self._loaded = True
        self._load_time = 0.01
        import time

        self._load_timestamp = time.time()

    def predict(self, input_data: Any) -> Any:
        self._ensure_loaded()
        return {
            "prediction": 0,
            "probabilities": [0.95, 0.03, 0.02],
        }

    def predict_batch(self, input_data: list[Any]) -> list[Any]:
        self._ensure_loaded()
        return [self.predict(item) for item in input_data]

    def warmup(self) -> None:
        self.predict({"features": [1.0, 2.0]})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_model() -> MockModel:
    """Return a loaded MockModel instance."""
    m = MockModel()
    m.load()
    return m


@pytest.fixture()
def mock_model_unloaded() -> MockModel:
    """Return an un-loaded MockModel instance."""
    return MockModel()


@pytest.fixture()
def mock_redis() -> AsyncMock:
    """Return an AsyncMock standing in for an aioredis client."""
    redis = AsyncMock()
    redis.ping.return_value = True
    redis.get.return_value = None
    redis.set.return_value = True
    return redis


@pytest.fixture()
def app(mock_model: MockModel, mock_redis: AsyncMock) -> FastAPI:
    """Build a FastAPI test app with mocked model and redis (no lifespan)."""
    from src.core.config import get_settings

    # Clear the lru_cache so test env vars take effect
    get_settings.cache_clear()

    from src.api.middleware.metrics import MetricsMiddleware
    from src.api.middleware.rate_limit import RateLimitMiddleware
    from src.api.middleware.request_id import RequestIDMiddleware
    from src.api.routes import health, model, predict

    settings = get_settings()

    # Build app without lifespan to avoid real model loading
    application = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
    )
    application.include_router(health.router)
    application.include_router(predict.router)
    application.include_router(model.router)
    application.add_middleware(RequestIDMiddleware)
    application.add_middleware(MetricsMiddleware)
    application.add_middleware(RateLimitMiddleware)

    application.state.model = mock_model
    application.state.redis = mock_redis

    return application


@pytest.fixture()
def client(app: FastAPI) -> TestClient:
    """Return a synchronous TestClient wired to the test app."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
