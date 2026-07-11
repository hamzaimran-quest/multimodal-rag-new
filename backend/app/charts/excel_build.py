"""Deterministic chart extraction for Excel entity × metric grids (separate from PDF wide/long)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from app.charts.columns import classify_long_layout, classify_wide_layout
from app.charts.llm_config import _validate_chart_data_spec
from app.charts.period import extract_period_key, is_annotation_column
from app.charts.profile import normalize_chart_table_rows
from app.charts.table_parse import parse_numeric_cell
from app.ingestion.xlsx_entity_keys import normalize_query_tokens, row_query_match_score
from app.ingestion.xlsx_serialize import table_rows_from_chunk_content

logger = logging.getLogger(__name__)

ChartTypeHint = Literal["bar", "line"] | None
_VALID_CHART_TYPES = {"bar", "line"}

MIN_ENTITY_GRID_METRICS = 2
OUTPUT_MAX_LABELS = 12
MAX_ENTITY_ROWS_WITHOUT_MATCH = 8

_LABEL_HEADERS = frozenset(
    {
        "country",
        "nation",
        "region",
        "name",
        "entity",
        "year",
        "rank",
        "id",
        "#",
    }
)

@dataclass(frozen=True)
class EntityGridLayout:
    entity_column_index: int
    metric_column_indices: tuple[int, ...]
    metric_labels: tuple[str, ...]


def _token_overlap_score(query: str, text: str) -> float:
    query_tokens = {token.lower() for token in query.split() if token}
    if not query_tokens:
        return 0.0
    text_tokens = {token.lower() for token in text.split() if token}
    return len(query_tokens & text_tokens) / len(query_tokens)


def _parse_metric_limit(query: str) -> int | None:
    """First plausible integer token in the query (structural slice count)."""
    for token in query.split():
        if not token.isdigit():
            continue
        value = int(token)
        if 1 <= value <= OUTPUT_MAX_LABELS:
            return value
    return None


def _column_values(data_rows: list[list[str]], col_idx: int) -> list[str]:
    return [row[col_idx] if col_idx < len(row) else "" for row in data_rows]


def _numeric_ratio(values: list[str]) -> float:
    non_empty = [value for value in values if value.strip()]
    if not non_empty:
        return 0.0
    numeric = sum(1 for value in non_empty if parse_numeric_cell(value) is not None)
    return numeric / len(non_empty)


def _is_label_column(header: str, values: list[str]) -> bool:
    lowered = header.strip().casefold()
    if lowered in _LABEL_HEADERS:
        return True

    non_empty = [value.strip() for value in values if value.strip()]
    if not non_empty:
        return True

    numeric_values = [parse_numeric_cell(value) for value in non_empty]
    parsed = [value for value in numeric_values if value is not None]
    if parsed and len(parsed) == len(non_empty):
        if lowered == "year" or all(1900 <= value <= 2099 for value in parsed):
            return len(set(parsed)) <= max(1, len(parsed) // 2)
        if lowered == "rank" or all(value == int(value) and value < 1000 for value in parsed):
            return True

    text_ratio = sum(1 for value in non_empty if parse_numeric_cell(value) is None) / len(non_empty)
    return text_ratio >= 0.5


def _resolve_entity_column_index(header: list[str], extra_metadata: dict[str, Any]) -> int:
    entity_column = str(extra_metadata.get("entity_key_column") or "").strip()
    if entity_column:
        lowered = [col.casefold() for col in header]
        target = entity_column.casefold()
        if target in lowered:
            return lowered.index(target)
    return 0


def classify_entity_grid_layout(
    rows: list[list[str]],
    extra_metadata: dict[str, Any] | None = None,
) -> EntityGridLayout | None:
    """
    Detect entity-row / metric-column Excel grids.

    Returns None when the table fits the PDF wide/long layouts or is not chart-shaped.
    """
    extra = extra_metadata or {}
    if len(rows) < 2 or len(rows[0]) < 3:
        return None

    if classify_wide_layout(rows) is not None or classify_long_layout(rows) is not None:
        return None

    header = rows[0]
    data_rows = rows[1:]
    if not data_rows:
        return None

    entity_column_index = _resolve_entity_column_index(header, extra)
    metric_indices: list[int] = []

    for col_idx, col_header in enumerate(header):
        if col_idx == entity_column_index:
            continue
        values = _column_values(data_rows, col_idx)
        if extract_period_key(col_header):
            return None
        if is_annotation_column(col_header, values):
            continue
        if _is_label_column(col_header, values):
            continue
        if _numeric_ratio(values) < 0.5:
            continue
        metric_indices.append(col_idx)

    if len(metric_indices) < MIN_ENTITY_GRID_METRICS:
        return None

    metric_labels = tuple(header[idx].strip() for idx in metric_indices)
    if any(not label for label in metric_labels):
        return None

    return EntityGridLayout(
        entity_column_index=entity_column_index,
        metric_column_indices=tuple(metric_indices),
        metric_labels=metric_labels,
    )


def analyze_excel_chartability(
    rows: list[list[str]],
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return an Excel entity-grid chart profile for logging and eligibility checks."""
    normalized = normalize_chart_table_rows(rows)
    layout = classify_entity_grid_layout(normalized, extra_metadata)
    if layout is None:
        return None

    metric_count = len(layout.metric_column_indices)
    return {
        "chartable": True,
        "orientation": "entity_grid",
        "period_count": metric_count,
        "metric_count": 1,
        "suggested_chart_type": "bar",
        "period_labels": list(layout.metric_labels),
        "entity_column_index": layout.entity_column_index,
        "metric_column_indices": list(layout.metric_column_indices),
    }


def _metric_columns_for_query(
    header: list[str],
    layout: EntityGridLayout,
    *,
    user_query: str,
) -> list[tuple[int, str]]:
    include_total = "total" in user_query.strip().lower()
    candidates: list[tuple[int, str]] = []
    for col_idx, label in zip(layout.metric_column_indices, layout.metric_labels, strict=True):
        if not include_total and label.strip().casefold() == "total":
            continue
        candidates.append((col_idx, label))

    if not candidates:
        return []

    limit = _parse_metric_limit(user_query)
    scored = [(_token_overlap_score(user_query, label), col_idx, label) for col_idx, label in candidates]
    named = [item for item in scored if item[0] > 0]
    if named:
        selected = sorted(named, key=lambda item: (-item[0], item[1]))
    else:
        selected = [(0.0, col_idx, label) for col_idx, label in candidates]

    cap = limit if limit is not None else OUTPUT_MAX_LABELS
    selected = selected[:cap]

    return [(col_idx, label) for _, col_idx, label in selected]


def _select_entity_row(
    data_rows: list[list[str]],
    layout: EntityGridLayout,
    *,
    user_query: str,
    extra_metadata: dict[str, Any],
) -> list[str] | None:
    tokens = normalize_query_tokens(user_query)
    best_row: list[str] | None = None
    best_score = -1.0

    row_entity_keys = extra_metadata.get("row_entity_keys") or {}
    sheet_row_map = list(extra_metadata.get("sheet_row_map") or [])

    for row_offset, row in enumerate(data_rows):
        entity_value = row[layout.entity_column_index] if layout.entity_column_index < len(row) else ""
        row_text = " | ".join(str(value) for value in row)
        score = row_query_match_score(row_text, tokens)
        if entity_value and tokens and any(token in entity_value.casefold() for token in tokens):
            score += 0.5

        mapped_key = None
        if row_offset < len(sheet_row_map):
            mapped_key = row_entity_keys.get(str(sheet_row_map[row_offset]))
        if mapped_key and tokens and any(token in mapped_key.casefold() for token in tokens):
            score += 0.5

        if score > best_score:
            best_score = score
            best_row = row

    if best_row is not None and best_score > 0:
        return best_row

    if len(data_rows) == 1:
        return data_rows[0]

    if len(data_rows) <= MAX_ENTITY_ROWS_WITHOUT_MATCH and tokens:
        return None

    if len(data_rows) <= MAX_ENTITY_ROWS_WITHOUT_MATCH:
        return data_rows[0]

    return None


def _entity_title_parts(row: list[str], layout: EntityGridLayout, header: list[str]) -> tuple[str, str]:
    entity_value = row[layout.entity_column_index].strip() if layout.entity_column_index < len(row) else ""
    year_value = ""
    for col_idx, col_name in enumerate(header):
        if col_name.strip().casefold() == "year" and col_idx < len(row):
            year_value = row[col_idx].strip()
            break
    return entity_value, year_value


def build_excel_chart_data_spec(
    rows: list[list[str]],
    *,
    user_query: str = "",
    chart_type: ChartTypeHint = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build a Chart.js-ready data spec from an Excel entity-grid table."""
    extra = extra_metadata or {}
    normalized = normalize_chart_table_rows(rows)
    layout = classify_entity_grid_layout(normalized, extra)
    if layout is None:
        logger.info("EXCEL_CHART_BUILD layout_rejected reason=no_entity_grid")
        return None

    header = normalized[0]
    data_rows = normalized[1:]
    entity_row = _select_entity_row(
        data_rows,
        layout,
        user_query=user_query,
        extra_metadata=extra,
    )
    if entity_row is None:
        logger.info(
            "EXCEL_CHART_BUILD entity_rejected reason=no_matching_row rows=%s query_preview=%r",
            len(data_rows),
            user_query[:120],
        )
        return None

    metric_columns = _metric_columns_for_query(header, layout, user_query=user_query)
    if len(metric_columns) < MIN_ENTITY_GRID_METRICS:
        logger.info("EXCEL_CHART_BUILD metrics_rejected reason=too_few_metrics count=%s", len(metric_columns))
        return None

    labels: list[str] = []
    values: list[float] = []
    for col_idx, label in metric_columns:
        if col_idx >= len(entity_row):
            continue
        parsed = parse_numeric_cell(entity_row[col_idx])
        if parsed is None:
            continue
        labels.append(label)
        values.append(parsed)

    if len(labels) < MIN_ENTITY_GRID_METRICS:
        logger.info("EXCEL_CHART_BUILD metrics_rejected reason=insufficient_numeric_values count=%s", len(labels))
        return None

    resolved_type = str(chart_type or "bar").strip().lower()
    if resolved_type not in _VALID_CHART_TYPES or resolved_type == "line":
        resolved_type = "bar"

    entity_name, year_value = _entity_title_parts(entity_row, layout, header)
    title_parts = [part for part in (entity_name, year_value) if part]
    title = " ".join(title_parts) if title_parts else "Chart"
    series_name = entity_name or "Value"

    spec = {
        "chart_type": resolved_type,
        "title": title,
        "labels": labels,
        "series": [{"name": series_name, "values": values}],
    }
    validation_error = _validate_chart_data_spec(spec)
    if validation_error is not None:
        logger.info("EXCEL_CHART_BUILD spec_rejected reason=%s", validation_error)
        return None

    logger.info(
        "EXCEL_CHART_BUILD spec_ok entity=%r metrics=%s chart_type=%s",
        entity_name,
        len(labels),
        resolved_type,
    )
    return spec


def build_excel_chart_data_spec_from_chunk(
    content: str,
    *,
    user_query: str = "",
    chart_type: ChartTypeHint = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    extra = extra_metadata or {}
    rows = table_rows_from_chunk_content(content, extra)
    if not rows:
        return None
    return build_excel_chart_data_spec(
        [[str(cell) for cell in row] for row in rows],
        user_query=user_query,
        chart_type=chart_type,
        extra_metadata=extra,
    )


def excel_entity_match_score(chunk: Any, query: str) -> float:
    """Boost chart candidate ranking when the query aligns with Excel entity keys."""
    extra = getattr(chunk, "extra_metadata", None) or {}
    if extra.get("source_format") != "xlsx":
        return 0.0

    tokens = normalize_query_tokens(query)
    if not tokens:
        return 0.0

    entity_keys = extra.get("entity_keys") or []
    hits = sum(1 for key in entity_keys if any(token in str(key).casefold() for token in tokens))
    if hits:
        return min(0.75, 0.25 * hits)

    rows = table_rows_from_chunk_content(getattr(chunk, "content", ""), extra)
    data_rows = rows[1:] if len(rows) > 1 else []
    layout = classify_entity_grid_layout(rows, extra)
    if layout is None or not data_rows:
        return 0.0

    best = max(
        row_query_match_score(" | ".join(str(value) for value in row), tokens)
        for row in data_rows
    )
    return best * 0.35
