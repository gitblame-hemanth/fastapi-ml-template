"""Tests for FastAPI dependency providers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.api.dependencies import get_model, get_redis, get_settings, verify_api_key
from src.core.config import Settings


def _request_with_state(**state) -> SimpleNamespace:
    """Build a stand-in for a Request exposing only ``app.state``."""
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(**state)))


def test_get_settings_returns_singleton() -> None:
    assert get_settings() is get_settings()


def test_get_model_missing_raises_503() -> None:
    request = _request_with_state()
    with pytest.raises(HTTPException) as exc_info:
        get_model(request)
    assert exc_info.value.status_code == 503


def test_get_model_unloaded_raises_503(mock_model_unloaded) -> None:
    request = _request_with_state(model=mock_model_unloaded)
    with pytest.raises(HTTPException) as exc_info:
        get_model(request)
    assert exc_info.value.status_code == 503


def test_get_model_returns_loaded_model(mock_model) -> None:
    request = _request_with_state(model=mock_model)
    assert get_model(request) is mock_model


async def test_get_redis_returns_client(mock_redis) -> None:
    request = _request_with_state(redis=mock_redis)
    assert await get_redis(request) is mock_redis


async def test_get_redis_returns_none_when_absent() -> None:
    request = _request_with_state()
    assert await get_redis(request) is None


async def test_verify_api_key_noop_when_disabled() -> None:
    settings = Settings(API_KEY_ENABLED=False)
    assert await verify_api_key(api_key=None, settings=settings) is None


async def test_verify_api_key_missing_key_401() -> None:
    settings = Settings(API_KEY_ENABLED=True, API_KEY="secret")
    with pytest.raises(HTTPException) as exc_info:
        await verify_api_key(api_key=None, settings=settings)
    assert exc_info.value.status_code == 401


async def test_verify_api_key_wrong_key_403() -> None:
    settings = Settings(API_KEY_ENABLED=True, API_KEY="secret")
    with pytest.raises(HTTPException) as exc_info:
        await verify_api_key(api_key="wrong", settings=settings)
    assert exc_info.value.status_code == 403


async def test_verify_api_key_valid_key_passes() -> None:
    settings = Settings(API_KEY_ENABLED=True, API_KEY="secret")
    assert await verify_api_key(api_key="secret", settings=settings) is None
