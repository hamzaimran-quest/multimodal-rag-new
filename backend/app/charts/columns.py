"""Column-role classification for metric-by-period table grids."""

from __future__ import annotations

from dataclasses import dataclass

from app.charts.period import (
    extract_period_key,
    is_annotation_column,
    is_period_label,
    period_display_label,
)
from app.charts.table_parse import parse_numeric_cell
from app.ingestion.tables import is_financial_value

MIN_PERIODS = 2
MAX_PERIODS = 5
MIN_METRICS = 1
MAX_METRICS = 8


@dataclass(frozen=True)
class WideLayout:
    period_column_indices: tuple[int, ...]
    period_labels: tuple[str, ...]
    annotation_column_indices: tuple[int, ...]


@dataclass(frozen=True)
class LongLayout:
    period_row_indices: tuple[int, ...]
    period_labels: tuple[str, ...]
    metric_column_indices: tuple[int, ...]
    metric_labels: tuple[str, ...]
    annotation_column_indices: tuple[int, ...]


def _column_values(data_rows: list[list[str]], col_idx: int) -> list[str]:
    return [row[col_idx] if col_idx < len(row) else "" for row in data_rows]


def _period_sort_key(label: str) -> tuple[int | float, str]:
    key = extract_period_key(label)
    if key and key.isdigit():
        return (0, int(key))
    return (1, label)


def _dedupe_period_columns(
    header: list[str],
    data_rows: list[list[str]],
    period_indices: list[int],
) -> list[int] | None:
    """Collapse duplicate period keys when all metric rows agree on values."""
    grouped: dict[str, list[int]] = {}
    for idx in period_indices:
        key = extract_period_key(header[idx])
        if not key:
            return None
        grouped.setdefault(key, []).append(idx)

    kept: list[int] = []
    for indices in grouped.values():
        if len(indices) == 1:
            kept.append(indices[0])
            continue

        reference = _column_values(data_rows, indices[0])
        if all(_column_values(data_rows, idx) == reference for idx in indices[1:]):
            kept.append(indices[0])
            continue
        return None

    return sorted(kept, key=lambda idx: _period_sort_key(header[idx]))


def classify_wide_layout(rows: list[list[str]]) -> WideLayout | None:
    if len(rows) < 2 or len(rows[0]) < 3:
        return None

    header = rows[0]
    data_rows = rows[1:]
    period_indices: list[int] = []
    annotation_indices: list[int] = []

    for col_idx in range(1, len(header)):
        values = _column_values(data_rows, col_idx)
        if is_annotation_column(header[col_idx], values):
            annotation_indices.append(col_idx)
            continue
        if extract_period_key(header[col_idx]):
            period_indices.append(col_idx)
            continue
        return None

    period_indices = _dedupe_period_columns(header, data_rows, period_indices) or []
    if not (MIN_PERIODS <= len(period_indices) <= MAX_PERIODS):
        return None

    period_labels = tuple(
        period_display_label(header[idx])
        for idx in sorted(period_indices, key=lambda i: _period_sort_key(header[i]))
    )
    if len(set(period_labels)) != len(period_labels):
        return None

    return WideLayout(
        period_column_indices=tuple(sorted(period_indices, key=lambda i: _period_sort_key(header[i]))),
        period_labels=period_labels,
        annotation_column_indices=tuple(annotation_indices),
    )


def classify_long_layout(rows: list[list[str]]) -> LongLayout | None:
    if len(rows) < 2 or len(rows[0]) < 3:
        return None

    header = rows[0]
    data_rows = rows[1:]

    metric_indices: list[int] = []
    annotation_indices: list[int] = []
    for col_idx in range(1, len(header)):
        values = _column_values(data_rows, col_idx)
        if is_annotation_column(header[col_idx], values):
            annotation_indices.append(col_idx)
            continue
        if header[col_idx].strip() and not is_period_label(header[col_idx]):
            metric_indices.append(col_idx)
            continue
        return None

    if not (MIN_METRICS <= len(metric_indices) <= MAX_METRICS):
        return None

    period_row_indices: list[int] = []
    for row_idx, row in enumerate(data_rows):
        label = row[0] if row else ""
        if extract_period_key(label):
            period_row_indices.append(row_idx)
        else:
            return None

    if not (MIN_PERIODS <= len(period_row_indices) <= MAX_PERIODS):
        return None

    period_labels = tuple(
        period_display_label(data_rows[idx][0])
        for idx in sorted(period_row_indices, key=lambda i: _period_sort_key(data_rows[i][0]))
    )
    if len(set(period_labels)) != len(period_labels):
        return None

    sorted_period_rows = tuple(sorted(period_row_indices, key=lambda i: _period_sort_key(data_rows[i][0])))

    metric_labels = tuple(header[idx].strip() for idx in metric_indices)
    if any(not label for label in metric_labels):
        return None
    if len(set(metric_labels)) != len(metric_labels):
        return None

    return LongLayout(
        period_row_indices=sorted_period_rows,
        period_labels=period_labels,
        metric_column_indices=tuple(metric_indices),
        metric_labels=metric_labels,
        annotation_column_indices=tuple(annotation_indices),
    )


def _metric_label_ratio_wide(data_rows: list[list[str]]) -> float:
    labels = [row[0] for row in data_rows]
    if not labels:
        return 0.0
    metric_like = sum(
        1 for label in labels if label.strip() and not is_period_label(label) and not is_financial_value(label)
    )
    return metric_like / len(labels)


def _data_numeric_ratio_for_cells(cells: list[str]) -> float:
    if not cells:
        return 0.0
    numeric = sum(1 for cell in cells if parse_numeric_cell(cell) is not None)
    return numeric / len(cells)


def extract_wide_series(rows: list[list[str]], layout: WideLayout) -> tuple[list[str], list[dict[str, list[float]]]] | None:
    data_rows = rows[1:]
    periods = list(layout.period_labels)
    series: list[dict[str, list[float]]] = []

    for row in data_rows:
        name = row[0].strip() if row else ""
        if not name or is_financial_value(name):
            return None
        values: list[float] = []
        for col_idx in layout.period_column_indices:
            parsed = parse_numeric_cell(row[col_idx] if col_idx < len(row) else "")
            if parsed is None:
                return None
            values.append(parsed)
        series.append({"name": name, "values": values})

    if not (MIN_METRICS <= len(series) <= MAX_METRICS):
        return None

    return periods, series


def extract_long_series(rows: list[list[str]], layout: LongLayout) -> tuple[list[str], list[dict[str, list[float]]]] | None:
    periods = list(layout.period_labels)
    series = [{"name": name, "values": []} for name in layout.metric_labels]

    for row_idx in layout.period_row_indices:
        row = rows[row_idx + 1]
        for series_idx, col_idx in enumerate(layout.metric_column_indices):
            parsed = parse_numeric_cell(row[col_idx] if col_idx < len(row) else "")
            if parsed is None:
                return None
            series[series_idx]["values"].append(parsed)

    for entry in series:
        if len(entry["values"]) != len(periods):
            return None

    return periods, series


def score_wide_layout(rows: list[list[str]], layout: WideLayout) -> float | None:
    data_rows = rows[1:]
    metric_ratio = _metric_label_ratio_wide(data_rows)
    if metric_ratio < 0.6:
        return None

    period_cells = [
        row[col_idx]
        for row in data_rows
        for col_idx in layout.period_column_indices
        if col_idx < len(row)
    ]
    numeric_ratio = _data_numeric_ratio_for_cells(period_cells)
    if numeric_ratio < 0.75:
        return None

    return (metric_ratio + numeric_ratio + 1.0) / 3.0


def score_long_layout(rows: list[list[str]], layout: LongLayout) -> float | None:
    data_rows = rows[1:]
    cells: list[str] = []
    for row_idx in layout.period_row_indices:
        row = data_rows[row_idx]
        for col_idx in layout.metric_column_indices:
            cells.append(row[col_idx] if col_idx < len(row) else "")

    numeric_ratio = _data_numeric_ratio_for_cells(cells)
    if numeric_ratio < 0.75:
        return None

    return (numeric_ratio + 1.0) / 2.0
