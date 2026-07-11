"""Tests for XLSX source row highlighting."""

from __future__ import annotations

from app.ingestion.xlsx_highlight import apply_xlsx_highlights_to_sources, match_xlsx_highlight_row_range
from app.retrieval.models import RetrievedChunk


def _slim_chunk(
    *,
    chunk_id: str,
    content: str,
    sheet_row_map: list[int],
    row_range: list[int],
    table_headers: list[str],
    entity_key_column: str | None = None,
    row_entity_keys: dict[str, str] | None = None,
    xlsx_anchor_key: str | None = None,
) -> RetrievedChunk:
    extra = {
        "source_format": "xlsx",
        "content_format": "slim_rows",
        "sheet_name": "Titles",
        "sheet_index": 1,
        "sheet_role": "primary",
        "table_headers": table_headers,
        "sheet_row_map": sheet_row_map,
        "row_range": row_range,
        "col_range": [1, 6],
    }
    if entity_key_column:
        extra["entity_key_column"] = entity_key_column
    if row_entity_keys:
        extra["row_entity_keys"] = row_entity_keys
    if xlsx_anchor_key:
        extra["xlsx_anchor_key"] = xlsx_anchor_key
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="d1",
        filename="catalog.xlsx",
        page_number=1,
        chunk_type="table",
        content=content,
        score=0.9,
        extra_metadata=extra,
    )


def test_match_xlsx_highlight_prefers_answer_date_over_query_noise() -> None:
    content = "Ben 10 | TV Show | 2012-03-01 | s10\n6 Years | Movie | 2015-06-19 | s99"
    row_range = match_xlsx_highlight_row_range(
        content=content,
        extra_metadata={
            "content_format": "slim_rows",
            "table_headers": ["title", "type", "date_added", "show_id"],
            "sheet_row_map": [4, 5],
            "row_range": [2, 43],
        },
        row_range=[2, 43],
        query="when was it released",
        answer="6 Years was released on 2015-06-19.",
    )
    assert row_range == [5, 5]


def test_match_xlsx_highlight_uses_answer_name_when_query_is_vague() -> None:
    content = "Ben 10 | TV Show | 2012-03-01 | s10\n6 Years | Movie | 2015-06-19 | s99"
    row_range = match_xlsx_highlight_row_range(
        content=content,
        extra_metadata={
            "content_format": "slim_rows",
            "table_headers": ["title", "type", "date_added", "show_id"],
            "sheet_row_map": [4, 5],
            "row_range": [2, 43],
        },
        row_range=[2, 43],
        query="tell me more",
        answer="Ben 10 is a TV Show added on 2012-03-01.",
    )
    assert row_range == [4, 4]


def test_match_xlsx_highlight_keeps_small_ranges() -> None:
    content = "Ben 10 | TV Show | 2012-03-01 | s10\n6 Years | Movie | 2015-06-19 | s99"
    row_range = match_xlsx_highlight_row_range(
        content=content,
        extra_metadata={
            "content_format": "slim_rows",
            "table_headers": ["title", "type", "date_added", "show_id"],
            "sheet_row_map": [4, 5],
            "row_range": [4, 5],
        },
        row_range=[4, 5],
        query="compare ben 10 and 6 years",
        answer="Ben 10 and 6 Years are different titles.",
    )
    assert row_range == [4, 5]


def test_match_xlsx_highlight_falls_back_without_match() -> None:
    row_range = match_xlsx_highlight_row_range(
        content="Alpha | 1\nBeta | 2",
        extra_metadata={
            "content_format": "slim_rows",
            "table_headers": ["title", "value"],
            "sheet_row_map": [2, 3],
            "row_range": [2, 43],
        },
        row_range=[2, 43],
        query="summarize the workbook",
        answer="The workbook contains many entries.",
    )
    assert row_range == [2, 43]


def test_match_xlsx_highlight_uses_anchor_key() -> None:
    row_range = match_xlsx_highlight_row_range(
        content="Ben 10 | TV Show | 2012-03-01 | s10\n6 Years | Movie | 2015-06-19 | s99",
        extra_metadata={
            "content_format": "slim_rows",
            "table_headers": ["title", "type", "date_added", "show_id"],
            "sheet_row_map": [4, 5],
            "row_range": [2, 43],
            "row_entity_keys": {"4": "s10", "5": "s99"},
            "xlsx_anchor_key": "s99",
        },
        row_range=[2, 43],
        query="tell me about 6 years",
        answer="",
    )
    assert row_range == [5, 5]


def test_match_xlsx_highlight_uses_title_column_when_entity_key_is_id() -> None:
    content = "Ben 10 | TV Show | 2012-03-01 | s10\n6 Years | Movie | 2015-06-19 | s99"
    row_range = match_xlsx_highlight_row_range(
        content=content,
        extra_metadata={
            "content_format": "slim_rows",
            "table_headers": ["title", "type", "date_added", "show_id"],
            "entity_key_column": "show_id",
            "sheet_row_map": [4, 5],
            "row_range": [2, 43],
        },
        row_range=[2, 43],
        query="tell me about ben 10",
        answer="",
    )
    assert row_range == [4, 4]


def test_apply_xlsx_highlights_to_sources_updates_matching_source() -> None:
    chunk = _slim_chunk(
        chunk_id="c1",
        content="Ben 10 | TV Show | 2012-03-01 | s10\n6 Years | Movie | 2015-06-19 | s99",
        sheet_row_map=[4, 5, 6, 7, 8],
        row_range=[2, 43],
        table_headers=["title", "type", "date_added", "show_id"],
        entity_key_column="title",
        row_entity_keys={"4": "s10", "5": "s99", "6": "s11", "7": "s12", "8": "s13"},
    )
    sources = [
        {
            "chunk_id": "c1",
            "source_format": "xlsx",
            "row_range": [2, 43],
            "sheet_name": "Titles",
        }
    ]
    apply_xlsx_highlights_to_sources(
        sources,
        [chunk],
        query="tell me about ben 10",
        answer="Ben 10 is a TV Show.",
    )
    assert sources[0]["row_range"] == [4, 4]
    assert sources[0]["highlight_row"] == 4


def test_apply_xlsx_highlights_skips_non_primary_sheets() -> None:
    chunk = _slim_chunk(
        chunk_id="c2",
        content="Actor A | s10",
        sheet_row_map=[2],
        row_range=[2, 10],
        table_headers=["cast", "show_id"],
        entity_key_column="show_id",
    )
    chunk.extra_metadata["sheet_role"] = "satellite"
    chunk.extra_metadata["sheet_name"] = "Cast"
    sources = [
        {
            "chunk_id": "c2",
            "source_format": "xlsx",
            "row_range": [2, 10],
            "sheet_name": "Cast",
        }
    ]
    apply_xlsx_highlights_to_sources(
        sources,
        [chunk],
        query="tell me about ben 10",
        answer="Ben 10 cast includes Actor A.",
    )
    assert sources[0]["row_range"] == [2, 10]
    assert "highlight_row" not in sources[0]
