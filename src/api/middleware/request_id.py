"""Middleware that assigns a unique request ID to every request."""

from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from src.core.logging import set_request_id

_HEADER = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique request ID to each request and response."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get(_HEADER, str(uuid.uuid4()))
        set_request_id(request_id)
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers[_HEADER] = request_id

        # Clear after response
        set_request_id(None)
        return response
