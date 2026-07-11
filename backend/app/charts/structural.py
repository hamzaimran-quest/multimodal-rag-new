"""Deterministic chart data extraction from structurally chartable tables."""

from __future__ import annotations

from typing import Any, Literal

from app.charts.llm_config import _validate_chart_data_spec
from app.charts.profile import analyze_table_chartability, normalize_chart_table_rows
from app.charts.spec import validate_and_build_chart_spec
from app.charts.table_parse import parse_markdown_table
from app.ingestion.xlsx_serialize import table_rows_from_chunk_content

ChartTypeHint = Literal["bar", "line"] | None
_VALID_CHART_TYPES = {"bar", "line"}


def _table_rows_from_markdown(markdown: str, extra_metadata: dict[str, Any] | None = None) -> list[list[str]]:
    extra = extra_metadata or {}
    rows = table_rows_from_chunk_content(markdown, extra)
    if rows:
        return [[str(cell) for cell in row] for row in rows]
    return parse_markdown_table(markdown)


def _filter_total_series(
    series: list[dict[str, Any]],
    *,
    user_query: str,
) -> list[dict[str, Any]]:
    if "total" in user_query.strip().lower():
        return series
    return [entry for entry in series if str(entry.get("name", "")).strip().lower() != "total"]


def _token_overlap_score(query: str, text: str) -> float:
    query_tokens = {token.lower() for token in query.split() if token}
    if not query_tokens:
        return 0.0
    text_tokens = {token.lower() for token in text.split() if token}
    return len(query_tokens & text_tokens) / len(query_tokens)


def _select_line_series(
    series: list[dict[str, Any]],
    *,
    user_query: str,
) -> list[dict[str, Any]]:
    """Pick metric rows for a line chart; collapse to one only when the query names it."""
    if len(series) <= 1:
        return series

    query = user_query.strip()
    if not query:
        return series

    scored = [(_token_overlap_score(query, str(entry.get("name", ""))), entry) for entry in series]
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score = scored[0][0]
    if best_score <= 0:
        return series

    top = [entry for score, entry in scored if score == best_score]
    return top if len(top) == 1 else series


def build_chart_data_spec_from_structure(
    markdown: str,
    *,
    user_query: str = "",
    chart_type: ChartTypeHint = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    Build {labels, series} from structural table profiling when the grid is chartable.

    Returns None when the table shape is ambiguous or fails validation.
    """
    rows = _table_rows_from_markdown(markdown, extra_metadata)
    if not rows:
        return None

    rows = normalize_chart_table_rows(rows)

    profile = analyze_table_chartability(rows)
    if profile is None:
        return None

    structural = validate_and_build_chart_spec(markdown, profile, user_query=user_query)
    if structural is None:
        return None

    series = _filter_total_series(structural["series"], user_query=user_query)
    if not series:
        return None

    resolved_type = str(chart_type or structural["chart_type"]).strip().lower()
    if resolved_type not in _VALID_CHART_TYPES:
        resolved_type = str(structural["chart_type"]).strip().lower()
    if resolved_type not in _VALID_CHART_TYPES:
        resolved_type = "bar"

    if resolved_type == "line":
        series = _select_line_series(series, user_query=user_query)

    value_axis = str(structural.get("value_axis_label") or "").strip()
    title = value_axis if value_axis and value_axis != "Value" else "Chart"

    spec = {
        "chart_type": resolved_type,
        "title": title,
        "labels": [str(label) for label in structural["periods"]],
        "series": [
            {"name": str(entry["name"]), "values": [float(value) for value in entry["values"]]}
            for entry in series
        ],
    }
    if _validate_chart_data_spec(spec) is not None:
        return None
    return spec
