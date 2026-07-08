"""Unit tests for in-process rate limiter."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.auth.rate_limit import _check_limit, reset_rate_limits


@pytest.fixture(autouse=True)
def _clear():
    reset_rate_limits()
    yield
    reset_rate_limits()


def test_allows_requests_under_limit():
    for _ in range(3):
        _check_limit(key="user:1", limit=3, window_seconds=60.0)


def test_blocks_when_limit_exceeded():
    for _ in range(2):
        _check_limit(key="user:2", limit=2, window_seconds=60.0)

    with pytest.raises(HTTPException) as exc:
        _check_limit(key="user:2", limit=2, window_seconds=60.0)
    assert exc.value.status_code == 429
