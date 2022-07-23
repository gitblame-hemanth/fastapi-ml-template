"""Prediction endpoints."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request

from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.core.config import get_settings
from src.core.metrics import INFERENCE_COUNT, INFERENCE_TIME

router = APIRouter(prefix="/api/v1", tags=["predictions"])
logger = structlog.get_logger(__name__)


def _build_input(req: PredictionRequest) -> dict[str, Any]:
    """Convert a PredictionRequest into the dict the model expects."""
    if req.features is not None:
        return {"features": req.features}
    if req.text is not None:
        return {"text": req.text}
    raise HTTPException(
        status_code=422,
        detail="Provide either 'features' (for sklearn) or 'text' (for NLP models).",
    )


def _cache_key(model_name: str, input_data: dict[str, Any]) -> str:
    """Deterministic cache key for a prediction request."""
    raw = json.dumps({"model": model_name, "input": input_data}, sort_keys=True)
    return f"pred:{hashlib.sha256(raw.encode()).hexdigest()}"


async def _get_cached(redis: Any, key: str) -> dict[str, Any] | None:
    """Return cached prediction or None."""
    if redis is None:
        return None
    try:
        data = await redis.get(key)
        if data:
            return json.loads(data)
    except Exception:
        logger.warning("redis_cache_miss", key=key)
    return None


async def _set_cached(redis: Any, key: str, value: dict[str, Any], ttl: int) -> None:
    """Store prediction in Redis cache."""
    if redis is None:
        return
    try:
        await redis.set(key, json.dumps(value), ex=ttl)
    except Exception:
        logger.warning("redis_cache_set_failed", key=key)


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, body: PredictionRequest) -> PredictionResponse:
    """Run a single prediction through the loaded model."""
    settings = get_settings()
    model = request.app.state.model
    redis = getattr(request.app.state, "redis", None)

    input_data = _build_input(body)
    key = _cache_key(model.name, input_data)

    # Check cache
    cached = await _get_cached(redis, key)
    if cached is not None:
        return PredictionResponse(
            **cached,
            model_name=model.name,
            model_version=model.version,
            cached=True,
        )

    # Inference
    start = time.perf_counter()
    try:
        result = model.predict(input_data)
        elapsed = time.perf_counter() - start
        INFERENCE_TIME.labels(model_name=model.name).observe(elapsed)
        INFERENCE_COUNT.labels(model_name=model.name, status="success").inc()
    except Exception as exc:
        INFERENCE_COUNT.labels(model_name=model.name, status="error").inc()
        logger.error("inference_error", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    # Cache result
    await _set_cached(redis, key, result, settings.REDIS_CACHE_TTL)

    return PredictionResponse(
        **result,
        model_name=model.name,
        model_version=model.version,
        cached=False,
    )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: Request, body: BatchPredictionRequest
) -> BatchPredictionResponse:
    """Run batch predictions through the loaded model."""
    model = request.app.state.model
    inputs = [_build_input(item) for item in body.inputs]

    start = time.perf_counter()
    try:
        results = model.predict_batch(inputs)
        elapsed = time.perf_counter() - start
        INFERENCE_TIME.labels(model_name=model.name).observe(elapsed)
        INFERENCE_COUNT.labels(model_name=model.name, status="success").inc(len(inputs))
    except Exception as exc:
        INFERENCE_COUNT.labels(model_name=model.name, status="error").inc(len(inputs))
        logger.error("batch_inference_error", error=str(exc))
        raise HTTPException(
            status_code=500, detail=f"Batch inference failed: {exc}"
        ) from exc

    predictions = [
        PredictionResponse(
            **r,
            model_name=model.name,
            model_version=model.version,
        )
        for r in results
    ]
    return BatchPredictionResponse(
        predictions=predictions,
        model_name=model.name,
        model_version=model.version,
    )
