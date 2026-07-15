"""Middleware for request tracking and rate limiting."""

from __future__ import annotations

import time
import uuid
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.core.logging import set_request_id


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Ensure every request has an X-Request-ID header.

    If the client sends one it is preserved; otherwise a UUID4 is generated.
    The value is injected into the logging context and returned in the response.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        set_request_id(request_id)
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory sliding-window rate limiter keyed by client IP.

    Not suitable for multi-process deployments (use Redis-backed limiter instead).
    """

    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        cutoff = now - self.window_seconds

        # Prune old entries
        self._hits[client_ip] = [t for t in self._hits[client_ip] if t > cutoff]

        if len(self._hits[client_ip]) >= self.max_requests:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
            )

        self._hits[client_ip].append(now)
        return await call_next(request)
