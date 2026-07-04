"""FastAPI ML Service — application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from src.api.middleware.metrics import MetricsMiddleware
from src.api.middleware.rate_limit import RateLimitMiddleware
from src.api.middleware.request_id import RequestIDMiddleware
from src.api.routes import health, model, predict
from src.core.config import get_settings
from src.core.logging import setup_logging
from src.core.metrics import MODEL_LOAD_TIME
from src.models.registry import create_model

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown lifecycle."""
    settings = get_settings()

    # --- Startup ---------------------------------------------------------
    setup_logging(settings.LOG_LEVEL)
    logger.info(
        "starting_service",
        app=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
    )

    # Load ML model
    model_instance = create_model(
        settings.MODEL_NAME,
        model_path=settings.MODEL_PATH,
    )
    try:
        model_instance.load()
        MODEL_LOAD_TIME.labels(model_name=model_instance.name).set(
            model_instance._load_time
        )
        logger.info(
            "model_loaded",
            model=model_instance.name,
            version=model_instance.version,
            load_time=round(model_instance._load_time, 3),
        )
    except Exception:
        logger.critical("model_load_failed", model=settings.MODEL_NAME, exc_info=True)
        raise

    # Warmup
    try:
        model_instance.warmup()
        logger.info("model_warmup_complete", model=model_instance.name)
    except Exception:
        logger.warning("model_warmup_failed", model=model_instance.name, exc_info=True)

    app.state.model = model_instance

    # Connect Redis
    redis_client = None
    try:
        import redis.asyncio as aioredis

        redis_client = aioredis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=5,
        )
        await redis_client.ping()
        logger.info("redis_connected", url=settings.REDIS_URL)
    except Exception:
        logger.warning(
            "redis_connection_failed — caching disabled",
            url=settings.REDIS_URL,
            exc_info=True,
        )
        redis_client = None

    app.state.redis = redis_client

    yield

    # --- Shutdown --------------------------------------------------------
    if redis_client is not None:
        await redis_client.aclose()
        logger.info("redis_disconnected")

    logger.info("service_stopped")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        description="Production-ready ML inference service built with FastAPI.",
        version=settings.APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
    )

    # --- Routers ---------------------------------------------------------
    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(model.router)

    # --- Middleware (order matters: outermost first) ----------------------
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Prometheus /metrics endpoint ------------------------------------
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    return app


app = create_app()
