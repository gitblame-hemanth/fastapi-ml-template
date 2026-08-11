"""Tests for application configuration."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.core.config import Settings, get_settings


def test_default_config_values() -> None:
    """Settings should have sensible defaults when no env vars override."""
    get_settings.cache_clear()
    # Temporarily remove test env vars to verify actual defaults
    env_overrides = {
        "APP_ENVIRONMENT": "production",
        "APP_DEBUG": "false",
        "APP_LOG_LEVEL": "INFO",
        "APP_RATE_LIMIT_REQUESTS": "100",
    }
    # Remove keys set by conftest so we see real defaults
    keys_to_remove = ["APP_API_KEY", "APP_API_KEY_ENABLED"]
    import os

    saved = {k: os.environ.pop(k, None) for k in keys_to_remove}
    with patch.dict("os.environ", env_overrides):
        for k in keys_to_remove:
            os.environ.pop(k, None)
        s = Settings()
        assert s.APP_NAME == "FastAPI ML Service"
        assert s.APP_VERSION == "1.0.0"
        assert s.DEBUG is False
        assert s.ENVIRONMENT == "production"
        assert s.REDIS_URL == "redis://localhost:6379"
        assert s.REDIS_CACHE_TTL == 300
        assert s.RATE_LIMIT_REQUESTS == 100
        assert s.RATE_LIMIT_WINDOW == 60
        assert s.INFERENCE_TIMEOUT == 30.0
        assert s.LOG_LEVEL == "INFO"
        assert s.API_KEY_ENABLED is False
        assert s.API_KEY is None
        assert s.CORS_ORIGINS == ["*"]
    # Restore
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v
    get_settings.cache_clear()


def test_config_from_env_vars() -> None:
    """Settings should pick up APP_ prefixed env vars."""
    get_settings.cache_clear()
    with patch.dict(
        "os.environ",
        {
            "APP_APP_NAME": "TestService",
            "APP_DEBUG": "true",
            "APP_ENVIRONMENT": "staging",
            "APP_REDIS_URL": "redis://redis:6380",
            "APP_RATE_LIMIT_REQUESTS": "50",
            "APP_LOG_LEVEL": "WARNING",
            "APP_CORS_ORIGINS": '["http://a.com","http://b.com"]',
        },
    ):
        s = Settings()
        assert s.APP_NAME == "TestService"
        assert s.DEBUG is True
        assert s.ENVIRONMENT == "staging"
        assert s.REDIS_URL == "redis://redis:6380"
        assert s.RATE_LIMIT_REQUESTS == 50
        assert s.LOG_LEVEL == "WARNING"
        assert s.CORS_ORIGINS == ["http://a.com", "http://b.com"]
    get_settings.cache_clear()


def test_config_validation() -> None:
    """Invalid LOG_LEVEL should raise a validation error."""
    get_settings.cache_clear()
    with patch.dict("os.environ", {"APP_LOG_LEVEL": "INVALID_LEVEL"}):
        with pytest.raises(ValidationError):
            Settings()
    get_settings.cache_clear()


def test_cors_origins_json_string_parsing() -> None:
    """CORS_ORIGINS accepts a JSON array string from env."""
    get_settings.cache_clear()
    with patch.dict(
        "os.environ", {"APP_CORS_ORIGINS": '["http://x.com","http://y.com"]'}
    ):
        s = Settings()
        assert s.CORS_ORIGINS == ["http://x.com", "http://y.com"]
    get_settings.cache_clear()


def test_cors_origins_list() -> None:
    """CORS_ORIGINS also accepts a list directly."""
    s = Settings(CORS_ORIGINS=["http://a.com"])
    assert s.CORS_ORIGINS == ["http://a.com"]


def test_get_settings_caches() -> None:
    """get_settings should return the same instance on repeated calls."""
    get_settings.cache_clear()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    get_settings.cache_clear()
