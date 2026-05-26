"""Health check endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

from src.api.schemas import HealthResponse
from src.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Return service health including model and Redis status."""
    settings = get_settings()
    model = request.app.state.model
    redis_client = getattr(request.app.state, "redis", None)

    redis_connected = False
    if redis_client is not None:
        try:
            await redis_client.ping()
            redis_connected = True
        except Exception:
            redis_connected = False

    return HealthResponse(
        status="healthy" if model._loaded else "degraded",
        version=settings.APP_VERSION,
        model=model.get_info(),
        redis_connected=redis_connected,
    )


@router.get("/readiness")
async def readiness(request: Request) -> dict[str, str]:
    """Kubernetes readiness probe — returns 200 only when model is loaded."""
    model = request.app.state.model
    if not model._loaded:
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=503, content={"status": "not_ready"})
    return {"status": "ready"}


@router.get("/liveness")
async def liveness() -> dict[str, str]:
    """Kubernetes liveness probe — always returns 200."""
    return {"status": "alive"}
