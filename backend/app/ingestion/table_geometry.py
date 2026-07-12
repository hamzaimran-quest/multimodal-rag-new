"""Geometry-based table reconstruction from word bounding boxes."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.ingestion.tables import is_financial_value, is_numeric_like, is_year_like, validate_reconstructed_table

if TYPE_CHECKING:
    import pdfplumber

DATA_ROW_NUMERIC_THRESHOLD = 0.5
COLUMN_GAP_RATIO = 0.55
LABEL_BBOX_EXPAND_RATIO = 0.15


def expand_table_bbox_for_labels(
    page: "pdfplumber.page.Page",
    bbox: tuple[float, ...],
    *,
    left_ratio: float = LABEL_BBOX_EXPAND_RATIO,
) -> tuple[float, float, float, float]:
    """Extend a table bbox leftward so row labels outside the pdfplumber grid are in scope."""
    x0, top, x1, bottom = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    expand_left = min(x0 - page.bbox[0], page.width * left_ratio)
    expanded = (max(page.bbox[0], x0 - expand_left), top, x1, bottom)
    px0, ptop, px1, pbottom = page.bbox
    return (
        max(px0, expanded[0]),
        max(ptop, expanded[1]),
        min(px1, expanded[2]),
        min(pbottom, expanded[3]),
    )


@dataclass(frozen=True)
class TableWord:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float

    @property
    def x_center(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def width(self) -> float:
        return self.x1 - self.x0


def words_in_bbox(page: "pdfplumber.page.Page", bbox: tuple[float, ...]) -> list[TableWord]:
    raw = page.within_bbox(bbox).extract_words(use_text_flow=True, keep_blank_chars=False)
    return [
        TableWord(
            text=str(w.get("text", "")).strip(),
            x0=float(w["x0"]),
            x1=float(w["x1"]),
            top=float(w["top"]),
            bottom=float(w["bottom"]),
        )
        for w in raw
        if str(w.get("text", "")).strip()
    ]


def _median_word_height(words: list[TableWord]) -> float:
    heights = [w.bottom - w.top for w in words if w.bottom > w.top]
    return statistics.median(heights) if heights else 8.0


def cluster_row_bands(words: list[TableWord], y_tolerance: float | None = None) -> list[list[TableWord]]:
    if not words:
        return []
    tol = y_tolerance if y_tolerance is not None else _median_word_height(words) * 0.65
    sorted_words = sorted(words, key=lambda w: (w.top, w.x0))
    bands: list[list[TableWord]] = []
    for word in sorted_words:
        placed = False
        for band in bands:
            if abs(word.top - band[0].top) <= tol:
                band.append(word)
                placed = True
                break
        if not placed:
            bands.append([word])
    return [sorted(band, key=lambda w: w.x0) for band in bands]


def _row_numeric_ratio(row: list[TableWord]) -> float:
    if not row:
        return 0.0
    financial = sum(1 for w in row if is_financial_value(w.text))
    return financial / len(row)


def _row_has_metric_label(row: list[TableWord]) -> bool:
    return any(not is_numeric_like(w.text) for w in row)


def find_first_data_row_index(row_bands: list[list[TableWord]]) -> int | None:
    for idx, row in enumerate(row_bands):
        financial_count = sum(1 for w in row if is_financial_value(w.text))
        if _row_has_metric_label(row) and financial_count >= 2:
            return idx
        if financial_count >= max(2, int(len(row) * DATA_ROW_NUMERIC_THRESHOLD)):
            return idx
    return None


def _cluster_positions(centers: list[float], gap_threshold: float) -> list[tuple[float, float]]:
    if not centers:
        return []
    sorted_centers = sorted(centers)
    clusters: list[list[float]] = [[sorted_centers[0]]]
    for center in sorted_centers[1:]:
        cluster_max = max(clusters[-1])
        if center - cluster_max <= gap_threshold:
            clusters[-1].append(center)
        else:
            clusters.append([center])
    bands: list[tuple[float, float]] = []
    for cluster in clusters:
        half_pad = gap_threshold * 0.35
        center = statistics.mean(cluster)
        bands.append((center - half_pad, center + half_pad))
    return bands


def _reference_data_rows(data_rows: list[list[TableWord]]) -> list[list[TableWord]]:
    counts = [sum(1 for w in row if is_financial_value(w.text)) for row in data_rows]
    if not counts:
        return []
    max_count = max(counts)
    return [row for row in data_rows if sum(1 for w in row if is_financial_value(w.text)) >= max_count]


def _merge_multiline_label_rows(row_bands: list[list[TableWord]]) -> list[list[TableWord]]:
    """Merge rows where labels wrap across lines but values sit on adjacent rows."""
    merged: list[list[TableWord]] = []
    idx = 0
    while idx < len(row_bands):
        row = row_bands[idx]
        financial = [w for w in row if is_financial_value(w.text)]
        label_words = [w for w in row if not is_financial_value(w.text) and not is_year_like(w.text)]

        if financial and not label_words and merged:
            prev = merged[-1]
            prev_financial = [w for w in prev if is_financial_value(w.text)]
            prev_labels = [w for w in prev if not is_financial_value(w.text) and not is_year_like(w.text)]
            if prev_labels and not prev_financial:
                merged[-1] = prev + row
                idx += 1
                continue

        if label_words and not financial:
            combined = list(row)
            look_ahead = idx + 1
            while look_ahead < len(row_bands):
                nxt = row_bands[look_ahead]
                nxt_financial = [w for w in nxt if is_financial_value(w.text)]
                nxt_labels = [w for w in nxt if not is_financial_value(w.text) and not is_year_like(w.text)]
                if nxt_financial:
                    combined.extend(nxt)
                    look_ahead += 1
                    while look_ahead < len(row_bands):
                        trailing = row_bands[look_ahead]
                        trailing_financial = [w for w in trailing if is_financial_value(w.text)]
                        trailing_labels = [
                            w for w in trailing if not is_financial_value(w.text) and not is_year_like(w.text)
                        ]
                        if trailing_labels and not trailing_financial:
                            combined.extend(trailing)
                            look_ahead += 1
                            continue
                        break
                    break
                if nxt_labels and not nxt_financial:
                    combined.extend(nxt)
                    look_ahead += 1
                    continue
                break
            merged.append(combined)
            idx = look_ahead
            continue

        merged.append(row)
        idx += 1
    return merged


def _infer_data_column_bands(data_rows: list[list[TableWord]]) -> list[tuple[float, float]]:
    reference_rows = _reference_data_rows(data_rows)
    if not reference_rows:
        return []

    template_row = reference_rows[0]
    financial_words = sorted(
        [w for w in template_row if is_financial_value(w.text)],
        key=lambda w: w.x0,
    )
    if not financial_words:
        return []

    pad = statistics.median([max(w.width, 1.0) for w in financial_words]) * 0.3
    return [(word.x0 - pad, word.x1 + pad) for word in financial_words]


def _label_cutoff_x(data_rows: list[list[TableWord]], data_bands: list[tuple[float, float]]) -> float:
    if not data_bands:
        return float("-inf")
    first_band_x0 = data_bands[0][0]
    label_words = [
        w for row in data_rows for w in row if not is_financial_value(w.text) and w.x1 <= first_band_x0 + 2
    ]
    if label_words:
        return max(w.x1 for w in label_words)
    return first_band_x0 - 1.0


def _band_index_for_word(word: TableWord, bands: list[tuple[float, float]]) -> int | None:
    for idx, (x0, x1) in enumerate(bands):
        if word.x0 <= x1 and word.x1 >= x0:
            return idx
    if not bands:
        return None
    return min(range(len(bands)), key=lambda i: abs(word.x_center - statistics.mean(bands[i])))


def _words_for_band(row: list[TableWord], band: tuple[float, float]) -> list[TableWord]:
    return [w for w in row if w.x0 <= band[1] and w.x1 >= band[0]]


def _header_words_for_band_with_spanning(
    header_row: list[TableWord],
    band_idx: int,
    bands: list[tuple[float, float]],
) -> list[TableWord]:
    if not header_row:
        return []

    sorted_words = sorted(header_row, key=lambda w: w.x0)
    direct = [word for word in sorted_words if _band_index_for_word(word, bands) == band_idx]
    if direct:
        return direct

    for word_idx, word in enumerate(sorted_words):
        start_band = _band_index_for_word(word, bands)
        if start_band is None:
            continue
        if word_idx + 1 < len(sorted_words):
            next_band = _band_index_for_word(sorted_words[word_idx + 1], bands)
            end_band = (next_band - 1) if next_band is not None else len(bands) - 1
        else:
            end_band = len(bands) - 1
        if start_band <= band_idx <= end_band:
            return [word]
    return []


def _group_adjacent_words(row: list[TableWord], gap_tolerance: float) -> list[list[TableWord]]:
    groups: list[list[TableWord]] = []
    for word in sorted(row, key=lambda w: w.x0):
        if not groups or word.x0 - groups[-1][-1].x1 > gap_tolerance:
            groups.append([word])
        else:
            groups[-1].append(word)
    return groups


def _header_unit_groups(header_rows: list[list[TableWord]]) -> list[tuple[str, float]]:
  if len(header_rows) < 2:
    return []
  groups: list[tuple[str, float]] = []
  for row in header_rows[1:]:
    if not row:
      continue
    gap = statistics.median([max(w.width, 1.0) for w in row]) * 0.8
    for group in _group_adjacent_words(row, gap):
      text = " ".join(w.text for w in group).strip()
      if not text or is_year_like(text):
        continue
      groups.append((text, statistics.mean(w.x_center for w in group)))
  return groups


def _assign_unit_groups_to_bands(
    unit_groups: list[tuple[str, float]],
    bands: list[tuple[float, float]],
) -> dict[int, list[str]]:
    assignments: dict[int, list[str]] = {i: [] for i in range(len(bands))}
    candidates: list[tuple[float, str, float, int]] = []
    for text, center in unit_groups:
        for band_idx, band in enumerate(bands):
            distance = abs(statistics.mean(band) - center)
            candidates.append((distance, text, center, band_idx))
    candidates.sort()

    assigned_bands: set[int] = set()
    assigned_groups: set[tuple[str, float]] = set()
    for _, text, center, band_idx in candidates:
        group_key = (text, center)
        if band_idx in assigned_bands or group_key in assigned_groups:
            continue
        assignments[band_idx].append(text)
        assigned_bands.add(band_idx)
        assigned_groups.add(group_key)
    return assignments


def _compose_header_for_band(
    header_rows: list[list[TableWord]],
    band_idx: int,
    bands: list[tuple[float, float]],
    unit_assignments: dict[int, list[str]],
) -> str:
    parts: list[str] = []
    if header_rows:
        year_words = _header_words_for_band_with_spanning(header_rows[0], band_idx, bands)
        if year_words:
            parts.append(" ".join(w.text for w in year_words))

    for unit_text in unit_assignments.get(band_idx, []):
        if unit_text not in parts:
            parts.append(unit_text)

    return " ".join(part for part in parts if part).strip()


def _build_rows_from_words(
    header_rows: list[list[TableWord]],
    data_rows: list[list[TableWord]],
    data_bands: list[tuple[float, float]],
) -> list[list[str]] | None:
    if not data_bands:
        return None

    label_cutoff = _label_cutoff_x(data_rows, data_bands)
    unit_assignments = _assign_unit_groups_to_bands(_header_unit_groups(header_rows), data_bands)
    headers = ["Metric"]
    for band_idx in range(len(data_bands)):
        header_text = _compose_header_for_band(header_rows, band_idx, data_bands, unit_assignments)
        headers.append(header_text or f"Column {band_idx + 1}")

    body: list[list[str]] = []
    for row in data_rows:
        label_words = [w for w in row if w.x1 <= label_cutoff + 1 and not is_financial_value(w.text)]
        if not label_words:
            label_words = [w for w in row if w.x1 <= label_cutoff + 1]
        label = " ".join(w.text for w in label_words).strip()
        values = [" ".join(w.text for w in _words_for_band(row, band)).strip() for band in data_bands]
        body.append([label, *values])

    if not body:
        return None
    return [headers, *body]


def reconstruct_table_geometry(
    page: "pdfplumber.page.Page",
    bbox: tuple[float, ...],
) -> list[list[str]] | None:
    words = words_in_bbox(page, bbox)
    if not words:
        return None

    row_bands = cluster_row_bands(words)
    data_start = find_first_data_row_index(row_bands)
    if data_start is None or data_start < 1:
        return None

    header_rows = row_bands[:data_start]
    data_rows = _merge_multiline_label_rows(row_bands[data_start:])
    return _build_rows_from_words(header_rows, data_rows, _infer_data_column_bands(data_rows))


def reconstruct_table_text_lines(
    page: "pdfplumber.page.Page",
    bbox: tuple[float, ...],
) -> list[list[str]] | None:
    """Text-line fallback with looser row clustering."""
    words = words_in_bbox(page, bbox)
    if not words:
        return None

    row_bands = cluster_row_bands(words, y_tolerance=_median_word_height(words) * 0.9)
    data_start = find_first_data_row_index(row_bands)
    if data_start is None or data_start < 1:
        return None

    data_rows = row_bands[data_start:]
    data_rows = _merge_multiline_label_rows(data_rows)
    return _build_rows_from_words(row_bands[:data_start], data_rows, _infer_data_column_bands(data_rows))


__all__ = [
    "expand_table_bbox_for_labels",
    "reconstruct_table_geometry",
    "reconstruct_table_text_lines",
    "validate_reconstructed_table",
]
