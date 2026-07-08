"""Query-time validation and chart spec construction from table chunks."""

from __future__ import annotations

from typing import Any

from app.charts.columns import (
    classify_long_layout,
    classify_wide_layout,
    extract_long_series,
    extract_wide_series,
)
from app.charts.profile import analyze_table_chartability
from app.charts.table_parse import parse_markdown_table


def validate_and_build_chart_spec(
    markdown: str,
    chart_profile: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Re-validate a retrieved table chunk and build a frontend chart spec.

    Fails closed when structure, completeness, or orientation is ambiguous.
    """
    if not chart_profile.get("chartable"):
        return None

    rows = parse_markdown_table(markdown)
    if not rows:
        return None

    fresh_profile = analyze_table_chartability(rows)
    if fresh_profile is None:
        return None

    if fresh_profile["orientation"] != chart_profile.get("orientation"):
        return None
    if fresh_profile["period_count"] != chart_profile.get("period_count"):
        return None
    if fresh_profile["metric_count"] != chart_profile.get("metric_count"):
        return None

    orientation = fresh_profile["orientation"]
    chart_type = fresh_profile["suggested_chart_type"]

    if orientation == "wide":
        layout = classify_wide_layout(rows)
        if layout is None:
            return None
        extracted = extract_wide_series(rows, layout)
    else:
        layout = classify_long_layout(rows)
        if layout is None:
            return None
        extracted = extract_long_series(rows, layout)

    if extracted is None:
        return None

    periods, series = extracted
    if len(periods) != fresh_profile["period_count"]:
        return None
    if len(series) != fresh_profile["metric_count"]:
        return None

    return {
        "chart_type": chart_type,
        "periods": periods,
        "series": series,
        "orientation": orientation,
        "period_count": len(periods),
        "metric_count": len(series),
        "value_axis_label": fresh_profile.get("value_axis_label", "Value"),
        "period_axis_label": "Period",
    }
