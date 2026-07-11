"""Tests for XLSX anchor entity key helpers."""

from __future__ import annotations

from app.ingestion.models import ExtractedChunk
from app.ingestion.xlsx_entity_keys import (
    annotate_chunk_entity_keys,
    normalize_query_tokens,
    resolve_anchor_keys_from_chunk,
    row_query_match_score,
)
from app.retrieval.query_phrases import build_query_match_profile
from app.retrieval.models import RetrievedChunk


def test_normalize_query_tokens_filters_short_words() -> None:
    tokens = normalize_query_tokens("Tell me the country of Transformers Prime")
    assert "transformers" in tokens
    assert "prime" in tokens
    assert "me" not in tokens


def test_build_query_match_profile_keeps_short_title_tokens() -> None:
    profile = build_query_match_profile("tell me about ben 10")
    assert "ben 10" in [phrase.casefold() for phrase in profile.phrases]
    assert "ben" in profile.tokens
    assert "10" in profile.tokens


def test_row_query_match_score_prefers_full_phrase() -> None:
    profile = build_query_match_profile("tell me about ben 10")
    title_row = row_query_match_score("Ben 10 | Movie | 2012 | s1", profile=profile)
    cast_row = row_query_match_score("Jane Doe | Ben Smith | Actor", profile=profile)
    assert title_row == 1.0
    assert cast_row < 1.0


def test_resolve_anchor_keys_from_digit_word_title() -> None:
    chunk = RetrievedChunk(
        chunk_id="c2",
        doc_id="d1",
        filename="titles.xlsx",
        page_number=1,
        chunk_type="table",
        content="6 Years | Movie | 2015 | s99",
        score=0.8,
        extra_metadata={
            "source_format": "xlsx",
            "content_format": "slim_rows",
            "table_headers": ["title", "type", "year", "show_id"],
            "sheet_row_map": [2],
            "entity_key_column": "show_id",
            "row_entity_keys": {"2": "s99"},
        },
    )

    anchors = resolve_anchor_keys_from_chunk(chunk, "when was 6 years released")
    assert anchors and anchors[0][0] == "s99"
    assert anchors[0][1] >= 0.5


def test_resolve_anchor_keys_from_ben_10_title() -> None:
    chunk = RetrievedChunk(
        chunk_id="c3",
        doc_id="d1",
        filename="titles.xlsx",
        page_number=1,
        chunk_type="table",
        content="Ben 10 | TV Show | 2012 | s10",
        score=0.8,
        extra_metadata={
            "source_format": "xlsx",
            "content_format": "slim_rows",
            "table_headers": ["title", "type", "year", "show_id"],
            "sheet_row_map": [2],
            "entity_key_column": "show_id",
            "row_entity_keys": {"2": "s10"},
        },
    )

    anchors = resolve_anchor_keys_from_chunk(chunk, "tell me about ben 10")
    assert anchors and anchors[0][0] == "s10"
    assert anchors[0][1] >= 0.5


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
