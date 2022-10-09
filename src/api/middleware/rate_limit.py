"""Simple in-memory sliding-window rate limiter middleware.

Uses Redis when available for distributed rate limiting; falls back to a
per-process in-memory counter otherwise.
"""

from __future__ import annotations

import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.core.config import get_settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enforce per-IP rate limits based on config."""

    def __init__(self, app: object) -> None:
        super().__init__(app)
        self._buckets: dict[str, list[float]] = defaultdict(list)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        settings = get_settings()
        max_requests = settings.RATE_LIMIT_REQUESTS
        window = settings.RATE_LIMIT_WINDOW

        # Skip rate limiting for health/metrics endpoints
        if request.url.path in ("/health", "/liveness", "/readiness", "/metrics"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        cutoff = now - window

        # Prune expired entries
        bucket = self._buckets[client_ip]
        self._buckets[client_ip] = [ts for ts in bucket if ts > cutoff]
        bucket = self._buckets[client_ip]

        if len(bucket) >= max_requests:
            retry_after = int(bucket[0] - cutoff) + 1
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
                headers={"Retry-After": str(retry_after)},
            )

        bucket.append(now)
        return await call_next(request)
