"""Tests for XLSX anchor entity key helpers."""

from __future__ import annotations

from app.ingestion.models import ExtractedChunk
from app.ingestion.xlsx_entity_keys import (
    annotate_chunk_entity_keys,
    normalize_query_tokens,
    resolve_anchor_keys_from_chunk,
)
from app.retrieval.models import RetrievedChunk


def test_normalize_query_tokens_filters_short_words() -> None:
    tokens = normalize_query_tokens("Tell me the country of Transformers Prime")
    assert "transformers" in tokens
    assert "prime" in tokens
    assert "me" not in tokens


def test_resolve_anchor_keys_from_matching_primary_row() -> None:
    chunk = RetrievedChunk(
        chunk_id="c1",
        doc_id="d1",
        filename="shows.xlsx",
        page_number=1,
        chunk_type="table",
        content="Transformers Prime | 2013 | 70234439",
        score=0.9,
        extra_metadata={
            "source_format": "xlsx",
            "content_format": "slim_rows",
            "table_headers": ["title", "year", "show_id"],
            "sheet_row_map": [2],
            "entity_key_column": "show_id",
            "row_entity_keys": {"2": "70234439"},
        },
    )

    anchors = resolve_anchor_keys_from_chunk(
        chunk,
        "country of Transformers Prime",
    )
    assert anchors and anchors[0][0] == "70234439"
    assert anchors[0][1] >= 0.5


def test_annotate_chunk_entity_keys_on_standalone_band() -> None:
    chunk = ExtractedChunk(
        content="70234439 | United States\n999 | Canada",
        page_number=2,
        chunk_type="table",
        extraction_method="xlsx_native",
        extra_metadata={
            "source_format": "xlsx",
            "content_format": "slim_rows",
            "table_headers": ["show_id", "country"],
            "sheet_row_map": [2, 3],
            "sheet_role": "standalone",
        },
    )
    annotate_chunk_entity_keys(chunk, key_col_index=0, key_column="show_id")
    assert chunk.extra_metadata["entity_keys"] == ["70234439", "999"]
    assert chunk.extra_metadata["row_entity_keys"]["2"] == "70234439"
