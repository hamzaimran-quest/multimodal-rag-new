"""Match spreadsheet query sources to specific sheet rows for viewer highlighting."""

from __future__ import annotations

import logging
from typing import Any

from app.ingestion.xlsx_entity_keys import row_query_match_score
from app.ingestion.xlsx_serialize import table_rows_from_chunk_content
from app.retrieval.models import RetrievedChunk
from app.retrieval.query_phrases import QueryMatchProfile, build_query_match_profile

logger = logging.getLogger(__name__)

_MIN_ROW_MATCH_SCORE = 0.34


def _is_primary_table_sheet(extra: dict[str, Any]) -> bool:
    return str(extra.get("sheet_role") or "").casefold() == "primary"


def _data_rows_from_chunk(content: str, extra: dict[str, Any]) -> list[list[str]]:
    rows = table_rows_from_chunk_content(content, extra)
    if not rows:
        return []
    if len(rows) == 1:
        return rows
    return rows[1:]


def _sheet_rows_for_data_rows(extra: dict[str, Any], data_row_count: int) -> list[int]:
    sheet_row_map = [int(value) for value in (extra.get("sheet_row_map") or []) if value is not None]
    if sheet_row_map and len(sheet_row_map) == data_row_count:
        return sheet_row_map
    if sheet_row_map and len(sheet_row_map) == data_row_count + 1:
        return sheet_row_map[1 : 1 + data_row_count]
    if sheet_row_map and len(sheet_row_map) > data_row_count:
        return sheet_row_map[:data_row_count]

    row_range = extra.get("row_range")
    if isinstance(row_range, list) and len(row_range) == 2:
        start = int(row_range[0])
        if extra.get("table_headers") and start == 1:
            start = 2
        return [start + offset for offset in range(data_row_count)]

    return [2 + offset for offset in range(data_row_count)]


def _aligned_sheet_rows(content: str, extra: dict[str, Any]) -> list[tuple[int, list[str]]]:
    data_rows = _data_rows_from_chunk(content, extra)
    if not data_rows:
        return []
    sheet_rows = _sheet_rows_for_data_rows(extra, len(data_rows))
    return list(zip(sheet_rows, data_rows))


_LABEL_COLUMN_HINTS = ("title", "name", "label", "show", "movie", "series", "program")


def _label_column_index(headers: list[str]) -> int | None:
    lowered = [header.casefold() for header in headers]
    for hint in _LABEL_COLUMN_HINTS:
        for index, header in enumerate(lowered):
            if header == hint or header.endswith(f"_{hint}") or header.startswith(f"{hint}_"):
                return index
    return None


def _entity_column_index(headers: list[str], extra: dict[str, Any]) -> int | None:
    key_column = extra.get("entity_key_column")
    if key_column and headers:
        lowered = [header.casefold() for header in headers]
        target = str(key_column).casefold()
        if target in lowered:
            return lowered.index(target)
    return _label_column_index(headers)


def _sheet_row_for_anchor(extra: dict[str, Any], anchor_key: str | None) -> int | None:
    if not anchor_key:
        return None
    row_entity_keys = extra.get("row_entity_keys") or {}
    for sheet_row, value in row_entity_keys.items():
        if str(value) == str(anchor_key):
            return int(sheet_row)
    return None


def _score_row(
    row_values: list[str],
    *,
    headers: list[str],
    entity_col_index: int | None,
    query_profile: QueryMatchProfile,
    answer_profile: QueryMatchProfile | None,
) -> float:
    row_text = " | ".join(str(value) for value in row_values)
    score = row_query_match_score(row_text, profile=query_profile)
    if answer_profile is not None:
        score = max(score, row_query_match_score(row_text, profile=answer_profile))

    if entity_col_index is not None and entity_col_index < len(row_values):
        entity_value = str(row_values[entity_col_index]).strip()
        if entity_value:
            entity_folded = entity_value.casefold()
            for profile in (query_profile, answer_profile):
                if profile is None:
                    continue
                for phrase in profile.phrases:
                    if phrase.casefold() in entity_folded:
                        score = max(score, 1.0)

    return score


def _row_range_from_sheet_rows(sheet_rows: list[int]) -> list[int] | None:
    if not sheet_rows:
        return None
    ordered = sorted(sheet_rows)
    return [ordered[0], ordered[-1]]


def match_xlsx_highlight_row_range(
    *,
    content: str,
    extra_metadata: dict[str, Any] | None,
    row_range: list[int] | None,
    query: str,
    answer: str = "",
) -> list[int] | None:
    """Narrow a chunk row band to the sheet row(s) that best match the query/answer."""
    extra = extra_metadata or {}
    if not row_range or len(row_range) != 2:
        return row_range

    aligned_rows = _aligned_sheet_rows(content, extra)
    if not aligned_rows:
        return row_range

    anchor_row = _sheet_row_for_anchor(extra, extra.get("xlsx_anchor_key"))
    if anchor_row is not None:
        logger.info(
            "XLSX_HIGHLIGHT match anchor chunk_sheet=%s anchor_key=%s row=%s",
            extra.get("sheet_name"),
            extra.get("xlsx_anchor_key"),
            anchor_row,
        )
        return [anchor_row, anchor_row]

    if len(aligned_rows) == 1:
        only_row = aligned_rows[0][0]
        logger.info(
            "XLSX_HIGHLIGHT match single_row chunk_sheet=%s row=%s",
            extra.get("sheet_name"),
            only_row,
        )
        return [only_row, only_row]

    full_start, full_end = int(row_range[0]), int(row_range[1])
    if full_end - full_start <= 1:
        logger.info(
            "XLSX_HIGHLIGHT match keep_small_band chunk_sheet=%s row_range=%s",
            extra.get("sheet_name"),
            row_range,
        )
        return row_range

    headers = list(extra.get("table_headers") or [])
    entity_col_index = _entity_column_index(headers, extra)
    query_profile = build_query_match_profile(query)
    answer_profile = build_query_match_profile(answer) if (answer or "").strip() else None

    scored: list[tuple[float, int]] = []
    for sheet_row, row_values in aligned_rows:
        score = _score_row(
            row_values,
            headers=headers,
            entity_col_index=entity_col_index,
            query_profile=query_profile,
            answer_profile=answer_profile,
        )
        scored.append((score, sheet_row))

    best_score = max(score for score, _ in scored)
    if best_score < _MIN_ROW_MATCH_SCORE:
        logger.info(
            "XLSX_HIGHLIGHT match fallback chunk_sheet=%s row_range=%s "
            "aligned_rows=%s best_score=%.3f min_score=%.3f query_preview=%r",
            extra.get("sheet_name"),
            row_range,
            len(aligned_rows),
            best_score,
            _MIN_ROW_MATCH_SCORE,
            (query or "")[:80],
        )
        return row_range

    winners = [sheet_row for score, sheet_row in scored if score >= best_score - 1e-6]
    narrowed = _row_range_from_sheet_rows(winners)
    logger.info(
        "XLSX_HIGHLIGHT match narrowed chunk_sheet=%s from=%s to=%s "
        "best_score=%.3f winners=%s query_preview=%r answer_preview=%r",
        extra.get("sheet_name"),
        row_range,
        narrowed,
        best_score,
        winners[:6],
        (query or "")[:80],
        (answer or "")[:80],
    )
    return narrowed or row_range


def apply_xlsx_highlights_to_sources(
    sources: list[dict],
    chunks: list[RetrievedChunk],
    *,
    query: str,
    answer: str = "",
) -> int:
    """Attach query-specific sheet row ranges to xlsx sources for the viewer."""
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    xlsx_sources = 0
    updated = 0
    missing_chunk = 0

    for source in sources:
        if source.get("source_format") != "xlsx":
            continue
        xlsx_sources += 1
        chunk_id = str(source.get("chunk_id") or "")
        chunk = chunk_by_id.get(chunk_id)
        if chunk is None:
            missing_chunk += 1
            logger.warning(
                "XLSX_HIGHLIGHT chunk_missing chunk_id=%s sheet=%s "
                "known_chunk_ids=%s",
                chunk_id,
                source.get("sheet_name"),
                sorted(chunk_by_id.keys())[:12],
            )
            continue

        extra = chunk.extra_metadata or {}
        if not _is_primary_table_sheet(extra):
            logger.info(
                "XLSX_HIGHLIGHT skip_non_primary chunk_id=%s sheet=%s sheet_role=%s",
                chunk_id,
                source.get("sheet_name") or extra.get("sheet_name"),
                extra.get("sheet_role"),
            )
            continue

        current_range = source.get("row_range") or extra.get("row_range")
        logger.info(
            "XLSX_HIGHLIGHT before chunk_id=%s sheet=%s row_range=%s "
            "sheet_row_map_len=%s content_rows=%s",
            chunk_id,
            source.get("sheet_name") or extra.get("sheet_name"),
            current_range,
            len(extra.get("sheet_row_map") or []),
            len(_data_rows_from_chunk(chunk.content, extra)),
        )
        refined = match_xlsx_highlight_row_range(
            content=chunk.content,
            extra_metadata=extra,
            row_range=current_range if isinstance(current_range, list) else None,
            query=query,
            answer=answer,
        )
        if refined:
            source["row_range"] = refined
            if refined[0] == refined[1]:
                source["highlight_row"] = refined[0]
            updated += 1
            logger.info(
                "XLSX_HIGHLIGHT after chunk_id=%s sheet=%s row_range=%s highlight_row=%s",
                chunk_id,
                source.get("sheet_name") or extra.get("sheet_name"),
                source.get("row_range"),
                source.get("highlight_row"),
            )

    logger.info(
        "XLSX_HIGHLIGHT summary xlsx_sources=%s updated=%s missing_chunk=%s "
        "query_preview=%r answer_chars=%s",
        xlsx_sources,
        updated,
        missing_chunk,
        (query or "")[:80],
        len(answer or ""),
    )
    return updated
