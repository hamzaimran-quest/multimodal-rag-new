"""Tests for table-slot merge in document search."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.retrieval.models import RetrievedChunk
from app.retrieval.table_slot import merge_with_table_slot


def _chunk(
    chunk_id: str,
    *,
    chunk_type: str = "text",
    score: float = 1.0,
    content: str = "body",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        filename="report.pdf",
        page_number=1,
        chunk_type=chunk_type,
        content=content,
        score=score,
    )


def test_merge_with_table_slot_skips_when_not_pdf_scope():
    primary = [_chunk("a", score=0.9)]
    merged = merge_with_table_slot(
        MagicMock(),
        "revenue by region",
        None,
        primary,
        user_id=1,
        top_k=5,
        pdf_scope=False,
    )
    assert merged == primary


def test_merge_with_table_slot_keeps_top_three_tables(monkeypatch):
    highlights = _chunk(
        "highlights",
        chunk_type="table",
        score=0.9,
        content="| Metric | 2025 | 2024 |\n| --- | --- | --- |\n| Revenue | 880,941 | 862,072 |",
    )
    region = _chunk(
        "region",
        chunk_type="table",
        score=0.5,
        content="By region\n\n| Metric | 2025 | 2024 |\n| --- | --- | --- |\n| China | 616,249 | 615,264 |",
    )
    segment = _chunk(
        "segment",
        chunk_type="table",
        score=0.4,
        content="| Metric | 2025 | 2024 |\n| --- | --- | --- |\n| Consumer | 344,473 | 339,006 |",
    )
    primary = [
        _chunk("text", score=0.95, content="intro paragraph"),
        highlights,
        region,
        _chunk("body", score=0.3),
    ]

    def fake_hybrid_search(client, **kwargs):
        source = segment.model_dump()
        source.pop("score", None)
        return {"hits": {"hits": [{"_source": source, "_score": 0.4}]}}

    def fake_parse(hit):
        return RetrievedChunk(**hit["_source"], score=float(hit["_score"]))

    monkeypatch.setattr("app.retrieval.table_slot.hybrid_search", fake_hybrid_search)
    monkeypatch.setattr("app.retrieval.table_slot.parse_search_hit", fake_parse)

    merged = merge_with_table_slot(
        MagicMock(),
        "revenue by region 2024 and 2025",
        None,
        primary,
        user_id=1,
        top_k=7,
        pdf_scope=True,
        table_slots=3,
    )
    table_ids = [chunk.chunk_id for chunk in merged if chunk.chunk_type == "table"]
    assert table_ids == ["highlights", "region", "segment"]
    assert len(merged) == 5
    assert sum(1 for chunk in merged if chunk.chunk_type == "text") == 2


def test_merge_with_table_slot_injects_tables_when_primary_has_none(monkeypatch):
    primary = [_chunk("a", score=0.9), _chunk("b", score=0.8)]
    table_hit = _chunk(
        "tbl",
        chunk_type="table",
        score=0.6,
        content="| Year | Revenue |\n| --- | --- |\n| 2020 | 100 |",
    )

    def fake_hybrid_search(client, **kwargs):
        assert kwargs.get("chunk_type") == "table"
        source = table_hit.model_dump()
        source.pop("score", None)
        return {"hits": {"hits": [{"_source": source, "_score": 0.6}]}}

    def fake_parse(hit):
        return RetrievedChunk(**hit["_source"], score=float(hit["_score"]))

    monkeypatch.setattr("app.retrieval.table_slot.hybrid_search", fake_hybrid_search)
    monkeypatch.setattr("app.retrieval.table_slot.parse_search_hit", fake_parse)

    merged = merge_with_table_slot(
        MagicMock(),
        "five year financial highlights",
        None,
        primary,
        user_id=1,
        top_k=5,
        pdf_scope=True,
        table_slots=1,
    )
    assert any(chunk.chunk_type == "table" for chunk in merged)
