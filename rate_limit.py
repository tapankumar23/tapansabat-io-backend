import os
import time
from collections import defaultdict, deque

from fastapi import Request


class RateLimitError(Exception):
    def __init__(self, retry_after: int) -> None:
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after}s.")


_buckets: dict[str, deque] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce(request: Request) -> None:
    max_requests = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "1"))
    window_seconds = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

    ip = _client_ip(request)
    now = time.monotonic()
    window_start = now - window_seconds
    bucket = _buckets[ip]

    while bucket and bucket[0] < window_start:
        bucket.popleft()

    if len(bucket) >= max_requests:
        retry_after = int(bucket[0] - window_start) + 1
        raise RateLimitError(retry_after)

    bucket.append(now)
