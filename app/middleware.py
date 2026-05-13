"""Custom FastAPI middleware for Talent-Radar."""
from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

_RATE_LIMIT_WINDOW = 60
_MAX_REQUESTS_PER_WINDOW = 200
_request_counts: dict[str, list[float]] = defaultdict(list)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory sliding-window rate limiter.

    Limits each client IP to MAX_REQUESTS_PER_WINDOW requests per minute.
    """

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Enforce rate limit and pass request through."""
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - _RATE_LIMIT_WINDOW
        timestamps = _request_counts[client_ip]
        _request_counts[client_ip] = [t for t in timestamps if t > window_start]
        if len(_request_counts[client_ip]) >= _MAX_REQUESTS_PER_WINDOW:
            logger.warning("Rate limit exceeded for IP %s", client_ip)
            return Response(
                content='{"detail": "Rate limit exceeded. Try again in a minute."}',
                status_code=429,
                media_type="application/json",
            )
        _request_counts[client_ip].append(now)
        return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs method, path, status code, and latency for every request."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Log request details and pass through."""
        start = time.time()
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        response = await call_next(request)
        latency_ms = round((time.time() - start) * 1000, 1)
        logger.info(
            "%s %s %d %.1fms cid=%s",
            request.method, request.url.path,
            response.status_code, latency_ms, correlation_id,
        )
        return response
