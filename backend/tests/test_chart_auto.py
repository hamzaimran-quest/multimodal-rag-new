"""Tests for automatic chart creation from retrieved tables."""

from __future__ import annotations

import json

from app.charts.auto import best_table_chunk, chart_requested, try_auto_chart_from_retrieval
from app.retrieval.models import RetrievedChunk


def _table_chunk(*, chunk_id: str, content: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="d1",
        filename="report.pdf",
        page_number=4,
        chunk_type="table",
        content=content,
        score=score,
    )


def test_chart_requested_detects_plot_queries():
    assert chart_requested("plot a chart of the finances")
    assert not chart_requested("what was revenue in 2024")


def test_best_table_chunk_prefers_highest_score():
    finance = _table_chunk(chunk_id="finance", content="| A | B |\n| --- | --- |\n| 1 | 2 |", score=0.9)
    region = _table_chunk(chunk_id="region", content="| A | B |\n| --- | --- |\n| 3 | 4 |", score=0.5)
    text = RetrievedChunk(
        chunk_id="text",
        doc_id="d1",
        filename="report.pdf",
        page_number=1,
        chunk_type="text",
        content="plain",
        score=1.0,
    )
    assert best_table_chunk([text, region, finance]).chunk_id == "finance"


def test_try_auto_chart_from_retrieval_uses_top_table(monkeypatch):
    content = (
        "| Metric | 2022 | 2023 | 2024 |\n"
        "| --- | --- | --- | --- |\n"
        "| Series A | 100 | 110 | 120 |"
    )
    finance = _table_chunk(chunk_id="finance", content=content, score=0.95)
    region = _table_chunk(
        chunk_id="region",
        content="| Region | 2024 |\n| --- | --- |\n| China | 100 |",
        score=0.4,
    )

    def fake_create_chart(client, **kwargs):
        assert kwargs["chunk_id"] == "finance"
        chart = {
            "chart_type": "bar",
            "chunk_id": "finance",
            "filename": "report.pdf",
            "page_number": 4,
            "derivation": "tool",
        }
        return json.dumps({"status": "created"}), [chart], [finance]

    monkeypatch.setattr("app.charts.auto.execute_create_chart", fake_create_chart)

    charts, note = try_auto_chart_from_retrieval(
        object(),
        user_id=1,
        user_query="plot a chart of the finances",
        retrieved_chunks=[region, finance],
    )
    assert charts
    assert note is None
