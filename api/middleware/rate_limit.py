import time
import asyncio
from collections import defaultdict

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class InMemoryRateLimiter:
    """Sliding-window per-IP rate limiter for single-process deployments."""

    def __init__(self, max_requests: int = 10, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def is_allowed(self, key: str) -> bool:
        now = time.monotonic()
        async with self._lock:
            bucket = self._buckets[key]
            cutoff = now - self.window_seconds
            while bucket and bucket[0] < cutoff:
                bucket.pop(0)
            if len(bucket) >= self.max_requests:
                return False
            bucket.append(now)
            return True


_limiter = InMemoryRateLimiter(max_requests=10, window_seconds=60.0)


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        key = f"{client_ip}:{request.url.path}"

        if not await _limiter.is_allowed(key):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "detail": "Too many requests. Please wait and retry.",
                    "retry_after_seconds": int(_limiter.window_seconds),
                },
            )

        return await call_next(request)
