"""Shared helpers for XLSX foreign-key entity keys on row bands."""

from __future__ import annotations

import re
from typing import Any

from app.ingestion.models import ExtractedChunk
from app.ingestion.xlsx_serialize import table_rows_from_chunk_content

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def normalize_query_tokens(query: str) -> list[str]:
    tokens = [token.casefold() for token in _TOKEN_RE.findall(query or "")]
    return [token for token in tokens if len(token) >= 3]


def row_query_match_score(row_text: str, tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    haystack = row_text.casefold()
    hits = sum(1 for token in tokens if token in haystack)
    return hits / len(tokens)


def entity_key_from_row_values(
    row_values: list[str],
    *,
    key_col_index: int | None,
    key_column: str | None,
    headers: list[str],
) -> str | None:
    index = key_col_index
    if index is None and key_column and headers:
        lowered = [header.casefold() for header in headers]
        target = key_column.casefold()
        if target in lowered:
            index = lowered.index(target)
    if index is None or index >= len(row_values):
        return None
    value = row_values[index].strip()
    return value or None


def annotate_chunk_entity_keys(
    chunk: ExtractedChunk,
    *,
    key_col_index: int,
    key_column: str,
) -> None:
    extra = chunk.extra_metadata
    headers = list(extra.get("table_headers") or [])
    sheet_row_map = list(extra.get("sheet_row_map") or [])
    rows = table_rows_from_chunk_content(chunk.content, extra)
    data_rows = rows[1:] if len(rows) > 1 else []

    entity_keys: list[str] = []
    row_entity_keys: dict[str, str] = {}
    for row_offset, row_values in enumerate(data_rows):
        key_value = entity_key_from_row_values(
            row_values,
            key_col_index=key_col_index,
            key_column=key_column,
            headers=headers,
        )
        if not key_value:
            continue
        entity_keys.append(key_value)
        if row_offset < len(sheet_row_map):
            row_entity_keys[str(sheet_row_map[row_offset])] = key_value

    extra["entity_key_column"] = key_column
    extra["entity_keys"] = sorted(set(entity_keys))
    extra["row_entity_keys"] = row_entity_keys


def resolve_anchor_keys_from_chunk(
    chunk: Any,
    query: str,
    *,
    min_score: float = 0.34,
) -> list[tuple[str, float]]:
    """Return [(entity_key, row_match_score), ...] for query-aligned rows in a chunk."""
    extra = chunk.extra_metadata or {}
    if extra.get("source_format") != "xlsx":
        return []

    tokens = normalize_query_tokens(query)
    if not tokens:
        return []

    headers = list(extra.get("table_headers") or [])
    sheet_row_map = list(extra.get("sheet_row_map") or [])
    row_entity_keys = extra.get("row_entity_keys") or {}
    key_column = extra.get("entity_key_column")
    rows = table_rows_from_chunk_content(chunk.content, extra)
    data_rows = rows[1:] if len(rows) > 1 else []

    anchors: list[tuple[str, float]] = []
    for row_offset, row_values in enumerate(data_rows):
        row_text = " | ".join(str(value) for value in row_values)
        score = row_query_match_score(row_text, tokens)
        if score < min_score:
            continue

        key_value: str | None = None
        if row_offset < len(sheet_row_map):
            key_value = row_entity_keys.get(str(sheet_row_map[row_offset]))
        if not key_value:
            key_value = entity_key_from_row_values(
                row_values,
                key_col_index=None,
                key_column=key_column,
                headers=headers,
            )
        if key_value:
            anchors.append((key_value, score))

    return anchors
