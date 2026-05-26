"""Pydantic request/response schemas for the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Prediction schemas
# ---------------------------------------------------------------------------


class PredictionRequest(BaseModel):
    """Single prediction request."""

    features: list[float] | None = Field(
        default=None, description="Feature vector for sklearn models."
    )
    text: str | None = Field(default=None, description="Text input for NLP models.")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""

    inputs: list[PredictionRequest] = Field(
        ..., min_length=1, max_length=100, description="List of inputs."
    )


class PredictionResponse(BaseModel):
    """Single prediction response."""

    prediction: Any
    probabilities: list[float] | None = None
    label: str | None = None
    score: float | None = None
    model_name: str
    model_version: str
    cached: bool = False


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: list[PredictionResponse]
    model_name: str
    model_version: str


# ---------------------------------------------------------------------------
# Health schemas
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Service health check response."""

    status: str
    version: str
    model: dict[str, Any]
    redis_connected: bool


# ---------------------------------------------------------------------------
# Model info schemas
# ---------------------------------------------------------------------------


class ModelInfoResponse(BaseModel):
    """Model metadata response."""

    name: str
    version: str
    loaded: bool
    load_time_seconds: float
    loaded_at: float | None = None


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
    request_id: str | None = None
