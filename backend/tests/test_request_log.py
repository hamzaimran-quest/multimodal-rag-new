"""Tests for per-request retrieval logging helpers."""

from __future__ import annotations

from app.retrieval.models import RetrievedChunk
from app.retrieval.request_log import build_chart_eligibility_records, build_request_summary


def _table_chunk(*, chunk_id: str, content: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="d1",
        filename="report.pdf",
        page_number=2,
        chunk_type="table",
        content=content,
        score=0.81,
    )


def test_chart_eligibility_not_chartable():
    chunk = _table_chunk(
        chunk_id="t1",
        content="| A | B |\n| --- | --- |\n| 1 | 2 |",
    )
    records = build_chart_eligibility_records([chunk])
    assert len(records) == 1
    assert records[0]["runtime_chartable"] is False
    assert records[0]["validation_outcome"] == "not_chartable_at_runtime"
    assert records[0]["chart_offered"] is False


def test_chart_eligibility_offered():
    content = (
        "| Metric | 2022 | 2023 | 2024 |\n"
        "| --- | --- | --- | --- |\n"
        "| Series A | 100 | 110 | 120 |"
    )
    records = build_chart_eligibility_records([_table_chunk(chunk_id="t2", content=content)])
    assert records[0]["chart_offered"] is True
    assert records[0]["validation_outcome"] == "offered"
    assert records[0]["chart_type"] == "line"


def test_request_summary_includes_chunks_and_chart_counts():
    text = RetrievedChunk(
        chunk_id="x1",
        doc_id="d1",
        filename="report.pdf",
        page_number=1,
        chunk_type="text",
        content="hello",
        score=0.5,
    )
    table_content = (
        "| Metric | 2022 | 2023 | 2024 |\n"
        "| --- | --- | --- | --- |\n"
        "| Series A | 100 | 110 | 120 |\n"
        "| Series B | 10 | 12 | 15 |"
    )
    table = _table_chunk(chunk_id="t3", content=table_content)
    eligibility = build_chart_eligibility_records([table])
    charts = [{"chunk_id": "t3", "derivation": "tool"}]

    summary = build_request_summary(
        endpoint="/query/stream",
        query="sample query",
        top_k=8,
        doc_id=None,
        chunks=[text, table],
        charts=charts,
        chart_eligibility=eligibility,
    )

    assert summary["retrieved_total"] == 2
    assert summary["table_chunks_retrieved"] == 1
    assert summary["table_chunks_runtime_chartable"] == 1
    assert summary["charts_offered"] == 1
    assert summary["tool_charts_emitted"] == 1
    assert summary["chunk_type_counts"]["text"] == 1
    assert len(summary["chunks"]) == 2
    assert summary["chunks"][1]["chart"]["chart_offered"] is True
