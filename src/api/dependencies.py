"""FastAPI dependency injection providers."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from src.core.config import Settings
from src.core.config import get_settings as _get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_settings() -> Settings:
    """Return the cached application settings singleton."""
    return _get_settings()


def get_model(request: Request) -> Any:
    """Retrieve the ML model instance from ``app.state``.

    Raises:
        HTTPException(503): If no model has been loaded into app state.
    """
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    if not getattr(model, "_loaded", False):
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


async def get_redis(request: Request) -> Any | None:
    """Return the Redis client from ``app.state``, or ``None`` if unavailable."""
    return getattr(request.app.state, "redis", None)


async def verify_api_key(
    api_key: str | None = Security(_api_key_header),
    settings: Settings = Depends(get_settings),
) -> None:
    """Validate the ``X-API-Key`` header when authentication is enabled.

    If ``API_KEY_ENABLED`` is ``False`` in settings, this dependency is a
    no-op and every request passes through.

    Raises:
        HTTPException(401): Missing API key header.
        HTTPException(403): Invalid API key.
    """
    if not settings.API_KEY_ENABLED:
        return

    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing API key")

    if api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
