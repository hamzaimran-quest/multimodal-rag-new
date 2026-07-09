"""Pie-chart structural analysis and spec construction."""

from __future__ import annotations

from typing import Any

from app.charts.columns import (
    classify_wide_layout,
    extract_wide_series,
)
from app.charts.period import extract_period_key, period_display_label
from app.charts.table_parse import parse_numeric_cell
from app.ingestion.tables import clean_cell, is_financial_value

MIN_PIE_SLICES = 2
MAX_PIE_SLICES = 12


def _normalize_rows(rows: list[list[object | None]]) -> list[list[str]]:
    cleaned = [[clean_cell(c) for c in row] for row in rows if any(clean_cell(c) for c in row)]
    if not cleaned:
        return []
    width = max(len(row) for row in cleaned)
    return [row + [""] * (width - len(row)) for row in cleaned]


def analyze_category_value_pie(rows: list[list[str]]) -> dict[str, Any] | None:
    """
    Detect a two-column label + value table suitable for a pie chart.

    Example: | Region | Revenue | with one numeric column and text labels.
    """
    if len(rows) < MIN_PIE_SLICES + 1:
        return None
    if len(rows[0]) != 2:
        return None

    categories: list[str] = []
    values: list[float] = []
    value_header = rows[0][1].strip() or "Value"

    for row in rows[1:]:
        label = row[0].strip() if row else ""
        if not label or is_financial_value(label):
            return None
        parsed = parse_numeric_cell(row[1] if len(row) > 1 else "")
        if parsed is None or parsed < 0:
            return None
        categories.append(label)
        values.append(parsed)

    if not (MIN_PIE_SLICES <= len(categories) <= MAX_PIE_SLICES):
        return None
    if sum(values) <= 0:
        return None

    return {
        "chartable": True,
        "layout": "category_value",
        "slice_count": len(categories),
        "suggested_chart_type": "pie",
        "categories": categories,
        "values": values,
        "value_axis_label": value_header,
    }


def analyze_snapshot_pie(rows: list[list[str]]) -> dict[str, Any] | None:
    """Wide table with exactly one period column and multiple metric rows."""
    if len(rows) < MIN_PIE_SLICES + 1:
        return None

    header = rows[0]
    data_rows = rows[1:]
    period_indices = [
        col_idx
        for col_idx in range(1, len(header))
        if extract_period_key(header[col_idx])
    ]
    if len(period_indices) != 1:
        return None

    col_idx = period_indices[0]
    period_label = period_display_label(header[col_idx])
    categories: list[str] = []
    values: list[float] = []

    for row in data_rows:
        label = row[0].strip() if row else ""
        if not label or is_financial_value(label):
            return None
        parsed = parse_numeric_cell(row[col_idx] if col_idx < len(row) else "")
        if parsed is None or parsed < 0:
            return None
        categories.append(label)
        values.append(parsed)

    if not (MIN_PIE_SLICES <= len(categories) <= MAX_PIE_SLICES):
        return None
    if sum(values) <= 0:
        return None

    return {
        "chartable": True,
        "layout": "snapshot",
        "slice_count": len(categories),
        "suggested_chart_type": "pie",
        "categories": categories,
        "values": values,
        "period_label": period_label,
        "value_axis_label": "Value",
    }


def analyze_pie_chartability(rows: list[list[object | None]]) -> dict[str, Any] | None:
    """Return a pie profile when the table reduces to categorical slices."""
    normalized = _normalize_rows(rows)
    if len(normalized) < MIN_PIE_SLICES + 1:
        return None

    category_value = analyze_category_value_pie(normalized)
    if category_value is not None:
        return category_value

    return analyze_snapshot_pie(normalized)


def build_pie_spec_from_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Build a frontend pie spec from a pie profile."""
    categories = list(profile["categories"])
    values = list(profile["values"])
    series_name = profile.get("period_label") or profile.get("value_axis_label") or "Share"
    return {
        "chart_type": "pie",
        "periods": categories,
        "series": [{"name": series_name, "values": values}],
        "value_axis_label": profile.get("value_axis_label", "Value"),
        "period_axis_label": "Category",
        "orientation": profile.get("layout", "category_value"),
        "period_count": len(categories),
        "metric_count": 1,
    }


def build_pie_from_time_series(
    *,
    periods: list[str],
    series: list[dict[str, list[float]]],
    period_label: str | None = None,
    value_axis_label: str = "Value",
) -> dict[str, Any] | None:
    """
    Convert a metric-by-period table into a pie for one selected period.

    Each metric becomes a slice; values come from the chosen period column.
    """
    if not periods or not series:
        return None

    chosen = period_label
    if chosen is None:
        chosen = periods[-1]
    if chosen not in periods:
        normalized = {p.strip().lower(): p for p in periods}
        key = chosen.strip().lower()
        if key not in normalized:
            return None
        chosen = normalized[key]

    period_idx = periods.index(chosen)
    categories: list[str] = []
    values: list[float] = []

    for entry in series:
        name = str(entry.get("name", "")).strip()
        entry_values = entry.get("values") or []
        if not name or period_idx >= len(entry_values):
            return None
        value = entry_values[period_idx]
        if value < 0:
            return None
        categories.append(name)
        values.append(float(value))

    if not (MIN_PIE_SLICES <= len(categories) <= MAX_PIE_SLICES):
        return None
    if sum(values) <= 0:
        return None

    return {
        "chart_type": "pie",
        "periods": categories,
        "series": [{"name": chosen, "values": values}],
        "value_axis_label": value_axis_label,
        "period_axis_label": "Category",
        "orientation": "time_series_slice",
        "period_count": len(categories),
        "metric_count": 1,
    }


def try_build_pie_from_wide_rows(rows: list[list[str]]) -> dict[str, Any] | None:
    """Attempt pie spec from wide rows without going through bar/line profile."""
    profile = analyze_pie_chartability(rows)
    if profile is None:
        layout = classify_wide_layout(rows)
        if layout is None or len(layout.period_column_indices) != 1:
            return None
        extracted = extract_wide_series(rows, layout)
        if extracted is None:
            return None
        _, series = extracted
        periods = list(layout.period_labels)
        return build_pie_from_time_series(
            periods=periods,
            series=series,
            period_label=periods[0] if periods else None,
        )
    return build_pie_spec_from_profile(profile)
