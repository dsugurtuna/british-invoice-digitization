"""Request context, metrics and rate limiting."""

from __future__ import annotations

import time

from collections import deque
from collections.abc import Awaitable, Callable
from uuid import uuid4

import structlog

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from invoice_digitizer.api.metrics import REQUEST_COUNT, REQUEST_LATENCY

logger = structlog.get_logger(__name__)

CallNext = Callable[[Request], Awaitable[Response]]

_UNLIMITED_PREFIXES = ("/health", "/metrics")


def route_template(request: Request) -> str:
    """The matched route's path template, so metric labels stay low-cardinality."""
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) else "unmatched"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Give each request an ID, log it, time it and record metrics.

    The log line carries method, route template, status and timing. It does not
    carry file names or client addresses, which can identify people.
    """

    async def dispatch(self, request: Request, call_next: CallNext) -> Response:
        request_id = str(uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("request_failed", request_id=request_id, method=request.method)
            raise
        elapsed = time.perf_counter() - start
        route = route_template(request)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{elapsed:.4f}"
        REQUEST_COUNT.labels(request.method, route, str(response.status_code)).inc()
        REQUEST_LATENCY.labels(request.method, route).observe(elapsed)
        logger.info(
            "request_completed",
            request_id=request_id,
            method=request.method,
            route=route,
            status_code=response.status_code,
            duration_ms=round(elapsed * 1000, 2),
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding one-minute window per client address, held in memory.

    Limits: the counts live in one process, so N worker processes allow N times the
    limit; behind a reverse proxy every client shares the proxy's address. Use a
    shared store (for example Redis) or the proxy's own limiter for real traffic.
    """

    def __init__(self, app: ASGIApp, requests_per_minute: int = 60) -> None:
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60.0
        self._hits: dict[str, deque[float]] = {}
        self._last_sweep = time.monotonic()

    async def dispatch(self, request: Request, call_next: CallNext) -> Response:
        if request.url.path.startswith(_UNLIMITED_PREFIXES):
            return await call_next(request)

        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        hits = self._hits.setdefault(client, deque())
        while hits and now - hits[0] >= self.window_seconds:
            hits.popleft()

        if len(hits) >= self.requests_per_minute:
            retry_after = max(1, int(self.window_seconds - (now - hits[0])) + 1)
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": f"Limit is {self.requests_per_minute} requests per minute.",
                    }
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                },
            )

        hits.append(now)
        self._forget_idle_clients(now)
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(self.requests_per_minute - len(hits))
        return response

    def _forget_idle_clients(self, now: float) -> None:
        """Once per window, drop idle clients so memory does not grow without bound."""
        if now - self._last_sweep < self.window_seconds:
            return
        self._last_sweep = now
        idle = [
            key
            for key, hits in self._hits.items()
            if not hits or now - hits[-1] >= self.window_seconds
        ]
        for key in idle:
            del self._hits[key]
