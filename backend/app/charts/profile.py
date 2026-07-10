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
from app.charts.table_parse import parse_numeric_cell
from app.charts.units import detect_value_axis_label
from app.ingestion.tables import clean_cell, is_financial_value

MIN_PERIODS_BAR = 2
MAX_PERIODS = 5
MIN_PERIODS_LINE = 3
MIN_METRICS = 1
MAX_METRICS = 8

ORIENTATION_CONFIDENCE_DELTA = 0.25


def _split_embedded_value_cell(cell: str) -> tuple[str, str] | None:
    """Split a cell that mixes row-label text with a trailing numeric value."""
    parts = cell.split()
    if len(parts) < 2:
        return None

    for split_at in range(len(parts) - 1, 0, -1):
        prefix = " ".join(parts[:split_at])
        suffix = " ".join(parts[split_at:])
        if not prefix.strip() or is_financial_value(prefix):
            continue
        if parse_numeric_cell(suffix) is not None:
            return prefix, suffix
    return None


def _repair_spilled_label_values(rows: list[list[str]]) -> list[list[str]]:
    """Move label fragments out of value columns when extraction merged adjacent cells."""
    if len(rows) < 2:
        return rows

    repaired = [list(rows[0])]
    for row in rows[1:]:
        current = list(row)
        for col_idx in range(1, len(current)):
            cell = current[col_idx]
            if parse_numeric_cell(cell) is not None:
                continue
            split = _split_embedded_value_cell(cell)
            if split is None:
                continue
            prefix, suffix = split
            label = current[0].strip()
            current[0] = f"{label} {prefix}".strip() if label else prefix
            current[col_idx] = suffix
        repaired.append(current)
    return repaired


def _normalize_rows(rows: list[list[object | None]]) -> list[list[str]]:
    cleaned = [[clean_cell(c) for c in row] for row in rows if any(clean_cell(c) for c in row)]
    if not cleaned:
        return []
    width = max(len(row) for row in cleaned)
    padded = [row + [""] * (width - len(row)) for row in cleaned]
    return _repair_spilled_label_values(padded)


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


def normalize_chart_table_rows(rows: list[list[object | None]]) -> list[list[str]]:
    """Normalize cell text and repair label fragments spilled into value columns."""
    return _normalize_rows(rows)


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
