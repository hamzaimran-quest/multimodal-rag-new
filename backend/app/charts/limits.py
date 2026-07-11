"""Shared output caps and query-driven slice limits for chart builders."""

from __future__ import annotations

OUTPUT_MAX_LABELS = 12
OUTPUT_MAX_SERIES = 8


def parse_query_slice_limit(query: str, *, max_value: int) -> int | None:
    """First plausible integer token in the query (structural, no regex)."""
    for token in query.split():
        if not token.isdigit():
            continue
        value = int(token)
        if 1 <= value <= max_value:
            return value
    return None
