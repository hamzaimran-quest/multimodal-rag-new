"""Unified chart construction for computed and tool-driven workflows."""

from __future__ import annotations

from typing import Any, Literal

from app.charts.pie import (
    analyze_pie_chartability,
    build_pie_from_time_series,
    build_pie_spec_from_profile,
    try_build_pie_from_wide_rows,
)
from app.charts.profile import analyze_table_chartability
from app.charts.spec import validate_and_build_chart_spec
from app.charts.table_parse import parse_markdown_table
from app.retrieval.models import RetrievedChunk

ChartType = Literal["bar", "line", "pie"]
_VALID_CHART_TYPES = {"bar", "line", "pie"}


def _chart_type_compatible(
    profile: dict[str, Any],
    chart_type: ChartType,
) -> bool:
    metric_count = int(profile.get("metric_count", 0))
    period_count = int(profile.get("period_count", 0))

    if chart_type == "line":
        return metric_count == 1 and period_count >= 3
    if chart_type == "bar":
        return metric_count >= 1 and 2 <= period_count <= 5
    if chart_type == "pie":
        return metric_count >= 2 or period_count == 1
    return False


def _not_chartable_reason(
    *,
    chart_type: ChartType | None,
    has_table: bool,
    profile: dict[str, Any] | None,
    pie_profile: dict[str, Any] | None,
) -> str:
    if not has_table:
        return "No table data was found for this query."
    if chart_type == "pie":
        if pie_profile is None and profile is None:
            return (
                "A chart cannot be created for this data: the table is not structured "
                "as categorical slices or a metric-by-period grid suitable for a pie chart."
            )
        if profile is not None and pie_profile is None and chart_type == "pie":
            if not _chart_type_compatible(profile, "pie"):
                return (
                    "A chart cannot be created for this data as a pie chart: "
                    "pie charts need category-value pairs, a single-period snapshot, "
                    "or a multi-metric table with a chosen period."
                )
    if profile is None and pie_profile is None:
        return (
            "A chart cannot be created for this data: the retrieved table does not "
            "have a consistent metric-by-period structure."
        )
    if chart_type and profile and not _chart_type_compatible(profile, chart_type):
        return (
            f"A chart cannot be created for this data as a {chart_type} chart: "
            f"the table has {profile.get('metric_count')} metric(s) and "
            f"{profile.get('period_count')} period(s), which is not compatible."
        )
    return "A chart cannot be created for this data."


def build_chart_spec(
    markdown: str,
    *,
    chart_profile: dict[str, Any] | None = None,
    chart_type: ChartType | None = None,
    period_label: str | None = None,
) -> dict[str, Any] | None:
    """
    Build a chart spec from table markdown.

    Uses ingestion profile when provided; always re-validates structure at query time.
    """
    rows = parse_markdown_table(markdown)
    if not rows:
        return None

    requested = chart_type
    if requested and requested not in _VALID_CHART_TYPES:
        return None

    pie_profile = analyze_pie_chartability(rows)
    series_profile = analyze_table_chartability(rows)

    if requested == "pie":
        if pie_profile is not None:
            return build_pie_spec_from_profile(pie_profile)
        if series_profile is not None:
            from app.charts.columns import (
                classify_long_layout,
                classify_wide_layout,
                extract_long_series,
                extract_wide_series,
            )

            orientation = series_profile["orientation"]
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
            return build_pie_from_time_series(
                periods=periods,
                series=series,
                period_label=period_label,
                value_axis_label=series_profile.get("value_axis_label", "Value"),
            )
        return try_build_pie_from_wide_rows(rows)

    if series_profile is not None:
        effective_profile = dict(series_profile)
        if chart_profile:
            effective_profile.update(
                {
                    k: chart_profile[k]
                    for k in ("orientation", "period_count", "metric_count")
                    if k in chart_profile
                }
            )
        effective_profile["chartable"] = True
        if requested:
            if not _chart_type_compatible(effective_profile, requested):
                return None
            effective_profile["suggested_chart_type"] = requested
        spec = validate_and_build_chart_spec(markdown, effective_profile)
        if spec is None:
            return None
        if requested:
            spec["chart_type"] = requested
        return spec

    if requested == "pie" or requested is None:
        if pie_profile is not None:
            return build_pie_spec_from_profile(pie_profile)

    return None


def chart_payload_from_chunk(
    chunk: RetrievedChunk,
    spec: dict[str, Any],
    *,
    derivation: str = "computed",
    is_secondary: bool = False,
) -> dict[str, Any]:
    """Wrap a chart spec with chunk citation metadata for the API."""
    return {
        **spec,
        "chunk_id": chunk.chunk_id,
        "filename": chunk.filename,
        "page_number": chunk.page_number,
        "doc_id": chunk.doc_id,
        "is_secondary": is_secondary,
        "derivation": derivation,
        "citation": {
            "chunk_id": chunk.chunk_id,
            "filename": chunk.filename,
            "page_number": chunk.page_number,
            "chunk_type": chunk.chunk_type,
        },
    }


def attempt_chart_from_chunk(
    chunk: RetrievedChunk,
    *,
    chart_type: ChartType | None = None,
    period_label: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Try to build a chart from a table chunk.

    Returns (chart_payload, error_message). On success error_message is None.
    """
    if chunk.chunk_type != "table":
        return None, "The selected chunk is not a table."

    extra = chunk.extra_metadata or {}
    chart_profile = extra.get("chart_profile")
    rows = parse_markdown_table(chunk.content)
    pie_profile = analyze_pie_chartability(rows) if rows else None
    series_profile = analyze_table_chartability(rows) if rows else None

    spec = build_chart_spec(
        chunk.content,
        chart_profile=chart_profile,
        chart_type=chart_type,
        period_label=period_label,
    )
    if spec is not None:
        return chart_payload_from_chunk(chunk, spec, derivation="tool"), None

    reason = _not_chartable_reason(
        chart_type=chart_type,
        has_table=bool(rows),
        profile=series_profile,
        pie_profile=pie_profile,
    )
    return None, reason


def merge_chart_outputs(*chart_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge chart lists; later lists override earlier entries for the same chunk_id."""
    merged: dict[str, dict[str, Any]] = {}
    for charts in chart_lists:
        for chart in charts:
            merged[chart["chunk_id"]] = chart
    return list(merged.values())
