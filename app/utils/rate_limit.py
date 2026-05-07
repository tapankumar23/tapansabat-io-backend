import os
import time
from collections import defaultdict, deque

from fastapi import Request

_MAX_REQUESTS: int = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "1"))
_WINDOW_SECONDS: int = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
_EVICTION_INTERVAL: float = 3600.0


class RateLimitError(Exception):
    def __init__(self, retry_after: int) -> None:
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after}s.")


_buckets: dict[str, deque] = defaultdict(deque)
_last_eviction: float = 0.0


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _evict_stale(now: float) -> None:
    global _last_eviction
    if now - _last_eviction < _EVICTION_INTERVAL:
        return
    _last_eviction = now
    window_start = now - _WINDOW_SECONDS
    stale = [ip for ip, b in _buckets.items() if not b or b[-1] < window_start]
    for ip in stale:
        del _buckets[ip]


def enforce(request: Request) -> None:
    now = time.monotonic()
    window_start = now - _WINDOW_SECONDS
    ip = _client_ip(request)
    bucket = _buckets[ip]

    while bucket and bucket[0] < window_start:
        bucket.popleft()

    _evict_stale(now)

    if len(bucket) >= _MAX_REQUESTS:
        retry_after = int(bucket[0] - window_start) + 1
        raise RateLimitError(retry_after)

    bucket.append(now)
