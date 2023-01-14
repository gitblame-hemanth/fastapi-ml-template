"""Middleware that records Prometheus metrics for every request."""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from src.core.metrics import ACTIVE_REQUESTS, REQUEST_COUNT, REQUEST_LATENCY


class MetricsMiddleware(BaseHTTPMiddleware):
    """Track request count, latency, and active connections."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Skip instrumenting the /metrics endpoint itself
        if request.url.path == "/metrics":
            return await call_next(request)

        method = request.method
        path = request.url.path

        ACTIVE_REQUESTS.inc()
        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            REQUEST_COUNT.labels(method=method, endpoint=path, status_code=500).inc()
            ACTIVE_REQUESTS.dec()
            raise

        elapsed = time.perf_counter() - start
        REQUEST_COUNT.labels(
            method=method, endpoint=path, status_code=response.status_code
        ).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=path).observe(elapsed)
        ACTIVE_REQUESTS.dec()

        return response
