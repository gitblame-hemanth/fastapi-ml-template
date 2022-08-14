"""Model management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from src.api.schemas import ModelInfoResponse

router = APIRouter(prefix="/api/v1", tags=["model"])


@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info(request: Request) -> ModelInfoResponse:
    """Return metadata about the currently loaded model."""
    model = request.app.state.model
    info = model.get_info()
    return ModelInfoResponse(**info)


@router.post("/model/reload", response_model=ModelInfoResponse)
async def model_reload(request: Request) -> ModelInfoResponse:
    """Reload the model from disk (hot-reload without restart)."""
    model = request.app.state.model
    try:
        model.load()
        model.warmup()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model reload failed: {exc}",
        ) from exc
    return ModelInfoResponse(**model.get_info())
