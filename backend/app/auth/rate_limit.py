"""In-process per-user rate limiting (sliding window, no Redis)."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException, status

from app.auth.dependencies import get_current_user
from app.config import settings
from app.db.models import User

_lock = threading.Lock()
# key -> deque[timestamp]
_buckets: dict[str, deque[float]] = defaultdict(deque)


def _prune(bucket: deque[float], *, window_seconds: float, now: float) -> None:
    cutoff = now - window_seconds
    while bucket and bucket[0] <= cutoff:
        bucket.popleft()


def _check_limit(*, key: str, limit: int, window_seconds: float) -> None:
    if limit <= 0:
        return

    now = time.monotonic()
    with _lock:
        bucket = _buckets[key]
        _prune(bucket, window_seconds=window_seconds, now=now)
        if len(bucket) >= limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
            )
        bucket.append(now)


def reset_rate_limits() -> None:
    """Clear all buckets (for tests)."""
    with _lock:
        _buckets.clear()


def rate_limit_upload(user: User = Depends(get_current_user)) -> User:
    _check_limit(
        key=f"upload:{user.id}",
        limit=settings.upload_rate_limit_per_minute,
        window_seconds=60.0,
    )
    return user


def rate_limit_query(user: User = Depends(get_current_user)) -> User:
    _check_limit(
        key=f"query:{user.id}",
        limit=settings.query_rate_limit_per_minute,
        window_seconds=60.0,
    )
    return user
