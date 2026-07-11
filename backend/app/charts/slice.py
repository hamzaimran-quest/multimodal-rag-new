"""Query-time slicing for PDF wide/long chart specs."""

from __future__ import annotations

from typing import Any

from app.charts.limits import OUTPUT_MAX_LABELS, OUTPUT_MAX_SERIES, parse_query_slice_limit


def _token_overlap_score(query: str, text: str) -> float:
    query_tokens = {token.lower() for token in query.split() if token and not token.isdigit()}
    if not query_tokens:
        return 0.0
    text_tokens = {token.lower() for token in text.split() if token}
    return len(query_tokens & text_tokens) / len(query_tokens)


def _trim_series_values(series: list[dict[str, Any]], width: int) -> list[dict[str, Any]]:
    return [
        {"name": entry["name"], "values": list(entry["values"][:width])}
        for entry in series
    ]


def _slice_periods(
    periods: list[str],
    series: list[dict[str, Any]],
    *,
    user_query: str,
    limit: int | None,
) -> tuple[list[str], list[dict[str, Any]]]:
    if not periods:
        return periods, series

    query = user_query.strip()
    selected_indices: list[int] = []

    if query:
        scored = [(_token_overlap_score(query, label), index) for index, label in enumerate(periods)]
        matched = [index for score, index in scored if score > 0]
        if matched:
            selected_indices = sorted(matched)

    if not selected_indices:
        count = limit if limit is not None else min(len(periods), OUTPUT_MAX_LABELS)
        selected_indices = list(range(min(count, len(periods))))

    periods_out = [periods[index] for index in selected_indices]
    series_out = [
        {
            "name": entry["name"],
            "values": [entry["values"][index] for index in selected_indices if index < len(entry["values"])],
        }
        for entry in series
    ]
    return periods_out, series_out


def _slice_series(
    series: list[dict[str, Any]],
    *,
    user_query: str,
    limit: int | None,
) -> list[dict[str, Any]]:
    if not series:
        return series

    query = user_query.strip()
    if query:
        scored = [(_token_overlap_score(query, str(entry.get("name", ""))), entry) for entry in series]
        matched = [entry for score, entry in scored if score > 0]
        if matched:
            series = matched

    cap = limit if limit is not None else OUTPUT_MAX_SERIES
    if len(series) > cap:
        series = series[:cap]
    return series


def slice_pdf_chart_spec(
    periods: list[str],
    series: list[dict[str, Any]],
    *,
    user_query: str = "",
) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Trim periods (x-axis labels) and metric series to query intent and output caps.

    Integer tokens in the query slice the larger dimension (metrics vs periods).
    """
    if not periods or not series:
        return periods, series

    limit = parse_query_slice_limit(user_query, max_value=OUTPUT_MAX_LABELS)
    if limit is not None and len(series) >= len(periods) and len(series) > 1:
        series = _slice_series(series, user_query=user_query, limit=limit)
        width = len(periods)
        if width > OUTPUT_MAX_LABELS:
            periods, series = _slice_periods(periods, series, user_query="", limit=OUTPUT_MAX_LABELS)
        else:
            series = _trim_series_values(series, width)
        return periods, series

    periods, series = _slice_periods(periods, series, user_query=user_query, limit=limit)
    series = _slice_series(series, user_query=user_query, limit=limit)
    return periods, series
