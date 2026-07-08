"""Ingestion-time structural chartability analysis for tables."""

from __future__ import annotations

from typing import Any

from app.charts.columns import (
    classify_long_layout,
    classify_wide_layout,
    extract_long_series,
    extract_wide_series,
    score_long_layout,
    score_wide_layout,
)
from app.charts.units import detect_value_axis_label
from app.ingestion.tables import clean_cell

MIN_PERIODS_BAR = 2
MAX_PERIODS = 5
MIN_PERIODS_LINE = 3
MIN_METRICS = 1
MAX_METRICS = 8

ORIENTATION_CONFIDENCE_DELTA = 0.25


def _normalize_rows(rows: list[list[object | None]]) -> list[list[str]]:
    cleaned = [[clean_cell(c) for c in row] for row in rows if any(clean_cell(c) for c in row)]
    if not cleaned:
        return []
    width = max(len(row) for row in cleaned)
    return [row + [""] * (width - len(row)) for row in cleaned]


def _is_composition_row(values: list[float]) -> bool:
    """Detect a single row whose values sum to ~100 (pie-like; out of scope)."""
    if len(values) < 3:
        return False
    if not all(0 <= value <= 100 for value in values):
        return False
    total = sum(values)
    return 90.0 <= total <= 110.0


def _suggested_chart_type(*, metric_count: int, period_count: int) -> str | None:
    if metric_count < MIN_METRICS or metric_count > MAX_METRICS:
        return None
    if period_count < MIN_PERIODS_BAR or period_count > MAX_PERIODS:
        return None
    if metric_count == 1 and period_count >= MIN_PERIODS_LINE:
        return "line"
    if MIN_METRICS <= metric_count <= MAX_METRICS and MIN_PERIODS_BAR <= period_count <= MAX_PERIODS:
        return "bar"
    return None


def analyze_table_chartability(rows: list[list[object | None]]) -> dict[str, Any] | None:
    """
    Determine whether a table's shape reduces cleanly to metric-by-period series.

    Returns a chart_profile dict for extra_metadata, or None when not chartable.
    Logic uses only structural properties — never document-specific literals.
    """
    normalized = _normalize_rows(rows)
    if len(normalized) < 2:
        return None

    wide_layout = classify_wide_layout(normalized)
    long_layout = classify_long_layout(normalized)
    wide_score = score_wide_layout(normalized, wide_layout) if wide_layout else None
    long_score = score_long_layout(normalized, long_layout) if long_layout else None

    if wide_score is None and long_score is None:
        return None

    if wide_score is not None and long_score is not None:
        if abs(wide_score - long_score) < ORIENTATION_CONFIDENCE_DELTA:
            return None
        orientation = "wide" if wide_score > long_score else "long"
    elif wide_score is not None:
        orientation = "wide"
    else:
        orientation = "long"

    if orientation == "wide":
        assert wide_layout is not None
        extracted = extract_wide_series(normalized, wide_layout)
        if extracted is None:
            return None
        periods, series = extracted
        period_count = len(periods)
        metric_count = len(series)
        period_column_indices = list(wide_layout.period_column_indices)
    else:
        assert long_layout is not None
        extracted = extract_long_series(normalized, long_layout)
        if extracted is None:
            return None
        periods, series = extracted
        period_count = len(periods)
        metric_count = len(series)
        period_column_indices = []

    chart_type = _suggested_chart_type(metric_count=metric_count, period_count=period_count)
    if chart_type is None:
        return None

    if metric_count == 1 and period_count >= 3:
        values = series[0]["values"]
        if _is_composition_row(values):
            return None
    if metric_count > 1 and period_count >= 3:
        for entry in series:
            if _is_composition_row(entry["values"]):
                return None

    profile: dict[str, Any] = {
        "chartable": True,
        "orientation": orientation,
        "period_count": period_count,
        "metric_count": metric_count,
        "suggested_chart_type": chart_type,
        "period_labels": periods,
        "value_axis_label": detect_value_axis_label(
            normalized,
            orientation=orientation,
            period_column_indices=period_column_indices if orientation == "wide" else None,
        ),
    }
    if orientation == "wide":
        profile["period_column_indices"] = period_column_indices
    return profile
