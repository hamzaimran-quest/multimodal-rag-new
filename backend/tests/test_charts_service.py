"""Tests for computed chart assembly from retrieved chunks."""

from __future__ import annotations

from app.charts.service import build_computed_charts
from app.retrieval.models import RetrievedChunk


def _table_chunk(*, chunk_id: str = "t1", content: str, profile: dict) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="d1",
        filename="report.pdf",
        page_number=4,
        chunk_type="table",
        content=content,
        score=0.8,
        extra_metadata={"chart_profile": profile},
    )


def test_build_computed_charts_from_chartable_table():
    content = (
        "| Metric | 2022 | 2023 | 2024 |\n"
        "| --- | --- | --- | --- |\n"
        "| Series A | 100 | 110 | 120 |\n"
        "| Series B | 10 | 12 | 15 |"
    )
    profile = {
        "chartable": True,
        "orientation": "wide",
        "period_count": 3,
        "metric_count": 2,
        "suggested_chart_type": "bar",
    }
    charts = build_computed_charts([_table_chunk(content=content, profile=profile)])
    assert len(charts) == 1
    assert charts[0]["chart_type"] == "bar"
    assert charts[0]["derivation"] == "computed"
    assert charts[0]["is_secondary"] is False
    assert charts[0]["citation"]["page_number"] == 4


def test_computed_chart_marked_secondary_when_image_retrieved():
    content = (
        "| Metric | 2022 | 2023 | 2024 |\n"
        "| --- | --- | --- | --- |\n"
        "| Series A | 100 | 110 | 120 |"
    )
    profile = {
        "chartable": True,
        "orientation": "wide",
        "period_count": 3,
        "metric_count": 1,
        "suggested_chart_type": "line",
    }
    table = _table_chunk(content=content, profile=profile)
    image = RetrievedChunk(
        chunk_id="i1",
        doc_id="d1",
        filename="report.pdf",
        page_number=4,
        chunk_type="image",
        content="image context",
        score=0.7,
        image_url="/images/d1/page4_img0.png",
    )
    charts = build_computed_charts([table, image])
    assert len(charts) == 1
    assert charts[0]["is_secondary"] is True


def test_skips_non_chartable_or_non_table_chunks():
    text = RetrievedChunk(
        chunk_id="x1",
        doc_id="d1",
        filename="report.pdf",
        page_number=1,
        chunk_type="text",
        content="plain text",
        score=0.5,
    )
    assert build_computed_charts([text]) == []

    table = _table_chunk(
        content="| A | B |\n| --- | --- |\n| 1 | 2 |",
        profile={"chartable": False},
    )
    assert build_computed_charts([table]) == []
