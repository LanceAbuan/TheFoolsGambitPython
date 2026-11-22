"""Light per-IP token-bucket rate limit on POST endpoints (FS-9).

Pure stdlib (no new deps): a Starlette-native ASGI middleware granting each
client IP a burst of ``RATE_LIMIT_CAPACITY`` tokens refilled at
``RATE_LIMIT_REFILL``/s. Non-POST requests (GET, WebSocket, lifespan) pass
through untouched; an empty bucket answers 429 with the standard
``{"detail": ...}`` error shape.

Bucket state lives on the event loop only (ASGI middleware is never invoked
concurrently for the same app within one process's loop), so plain dict
operations suffice. The map is bounded: when it grows past ``_MAX_BUCKETS``
the fullest idle buckets are shed.
"""
from __future__ import annotations

import logging
import time

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from . import config

log = logging.getLogger(__name__)

_MAX_BUCKETS = 10_000


class TokenBucketRateLimit:
    """ASGI middleware: token bucket per client IP over POST requests."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._buckets: dict[str, tuple[float, float]] = {}  # ip -> (tokens, last_refill)

    def _take(self, ip: str) -> bool:
        """Consume one token for ``ip``; False when the bucket is empty."""
        now = time.monotonic()
        capacity = float(max(1, config.RATE_LIMIT_CAPACITY))
        rate = max(0.001, config.RATE_LIMIT_REFILL)
        tokens, last = self._buckets.get(ip, (capacity, now))
        tokens = min(capacity, tokens + (now - last) * rate)
        allowed = tokens >= 1.0
        self._buckets[ip] = (tokens - 1.0 if allowed else tokens, now)
        if len(self._buckets) > _MAX_BUCKETS:
            self._shed(capacity)
        return allowed

    def _shed(self, capacity: float) -> None:
        """Drop idle (full) buckets so the map cannot grow without bound."""
        full = sorted(
            (key for key, (tokens, _) in self._buckets.items() if tokens >= capacity - 1e-9),
        )
        for key in full[: len(full) // 2]:
            del self._buckets[key]
        log.debug("rate-limit buckets shed: %d idle entries dropped", len(full) // 2)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return
        client = scope.get("client")
        ip = client[0] if client else "unknown"
        if not self._take(ip):
            log.warning("rate limit hit for %s on %s", ip, scope.get("path", "?"))
            response = JSONResponse({"detail": "rate limit exceeded"}, status_code=429)
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)
