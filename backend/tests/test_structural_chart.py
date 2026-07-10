"""Tests for structural chart data extraction."""

from __future__ import annotations

from app.charts.build import attempt_chart_from_chunk
from app.charts.structural import build_chart_data_spec_from_structure
from app.retrieval.models import RetrievedChunk

_REGIONAL_MARKDOWN = (
    "| Metric | (CNY Million) 2025 | 2024 | YoY |\n"
    "| --- | --- | --- | --- |\n"
    "| China | 616,249 | 615,264 | 0.2% |\n"
    "| EMEA | 161,356 | 148,355 | 8.8% |\n"
    "| Asia Pacific | 50,113 | 43,306 | 15.7% |\n"
    "| Americas | 37,184 | 36,301 | 2.4% |\n"
    "| Other | 16,039 | 18,846 | (14.9)% |\n"
    "| Total | 880,941 | 862,072 | 2.2% |"
)


_SEGMENT_MARKDOWN = (
    "| Metric | (CNY Million) 2025 | 2024 | YoY |\n"
    "| --- | --- | --- | --- |\n"
    "| ICT Infrastructure | 375,014 | 365,424 | 2.6% |\n"
    "| Consumer | 344,473 | 339,006 | 1.6% |\n"
    "| Cloud Computing | 32,161 | 33,325 | (3.5)% |\n"
    "| Digital Power | 77,312 | 68,607 | 12.7% |\n"
    "| Intelligent Solution | Automotive 45,018 | 26,158 | 72.1% |\n"
    "| Other | 6,963 | 29,552 | (76.4)% |\n"
    "| Total | 880,941 | 862,072 | 2.2% |"
)


def test_build_chart_data_spec_from_structure_segment_table_with_spilled_label():
    spec = build_chart_data_spec_from_structure(
        _SEGMENT_MARKDOWN,
        user_query="business segment revenue chart",
        chart_type="bar",
    )
    assert spec is not None
    assert spec["chart_type"] == "bar"
    assert len(spec["series"]) == 6
    names = {entry["name"] for entry in spec["series"]}
    assert "Intelligent Solution Automotive" in names
    spilled = next(entry for entry in spec["series"] if "Automotive" in entry["name"])
    assert spilled["values"] == [26158.0, 45018.0]


def test_build_chart_data_spec_from_structure_regional_table():
    spec = build_chart_data_spec_from_structure(
        _REGIONAL_MARKDOWN,
        user_query="regional revenue chart",
        chart_type="bar",
    )
    assert spec is not None
    assert spec["chart_type"] == "bar"
    assert spec["labels"] == ["2024", "2025"]
    assert len(spec["series"]) == 5
    assert spec["series"][0]["name"] == "China"
    assert spec["series"][0]["values"] == [615264.0, 616249.0]
    assert all(entry["name"] != "Total" for entry in spec["series"])


def test_build_chart_data_spec_from_structure_includes_total_when_requested():
    spec = build_chart_data_spec_from_structure(
        _REGIONAL_MARKDOWN,
        user_query="chart including total revenue",
        chart_type="bar",
    )
    assert spec is not None
    assert any(entry["name"] == "Total" for entry in spec["series"])


def test_attempt_chart_from_chunk_prefers_structural_over_llm(monkeypatch):
    chunk = RetrievedChunk(
        chunk_id="regional-t1",
        doc_id="d1",
        filename="report.pdf",
        page_number=23,
        chunk_type="table",
        content=_REGIONAL_MARKDOWN,
        score=0.9,
    )

    def fail_llm(*args, **kwargs):
        raise AssertionError("LLM should not be called when structural path succeeds")

    monkeypatch.setattr("app.charts.build.extract_chart_data_spec", fail_llm)
    monkeypatch.setattr(
        "app.charts.build.build_quickchart_url",
        lambda config: "https://quickchart.io/chart?c=structural",
    )

    chart, error = attempt_chart_from_chunk(chunk, user_query="regional revenue", chart_type="bar")
    assert error is None
    assert chart is not None
    assert chart["chart_url"] == "https://quickchart.io/chart?c=structural"
    assert chart["periods"] == ["2024", "2025"]
    assert len(chart["series"]) == 5
