"""Tests for QuickChart chart building."""

from __future__ import annotations

from app.charts.build import attempt_chart_from_chunk
from app.charts.quickchart import build_quickchart_url
from app.retrieval.models import RetrievedChunk


def test_build_quickchart_url_from_config():
    url = build_quickchart_url(
        {
            "type": "bar",
            "data": {
                "labels": ["2024", "2025"],
                "datasets": [{"label": "China", "data": [100.0, 110.0]}],
            },
        }
    )
    assert url.startswith("https://quickchart.io/chart?")
    assert "China" in url or "2024" in url


def test_build_quickchart_url_from_line_config():
    url = build_quickchart_url(
        {
            "type": "line",
            "data": {
                "labels": ["2020", "2021", "2022"],
                "datasets": [
                    {
                        "label": "Revenue",
                        "data": [50.0, 60.0, 70.0],
                        "fill": False,
                        "borderColor": "rgb(54, 162, 235)",
                    }
                ],
            },
            "options": {"elements": {"line": {"tension": 0}}},
        }
    )
    assert url.startswith("https://quickchart.io/chart?")


def test_attempt_chart_from_chunk_uses_structural_path(monkeypatch):
    chunk = RetrievedChunk(
        chunk_id="t1",
        doc_id="d1",
        filename="report.pdf",
        page_number=4,
        chunk_type="table",
        content="| Region | 2024 | 2025 |\n| --- | --- | --- |\n| China | 100 | 110 |",
        score=0.9,
    )

    def fail_llm(*args, **kwargs):
        raise AssertionError("LLM should not run when structural profiling succeeds")

    monkeypatch.setattr("app.charts.build.extract_chart_data_spec", fail_llm)
    monkeypatch.setattr(
        "app.charts.build.build_quickchart_url",
        lambda config: "https://quickchart.io/chart?c=test",
    )

    chart, error = attempt_chart_from_chunk(chunk, user_query="chart china revenue", chart_type="bar")
    assert error is None
    assert chart is not None
    assert chart["chart_url"] == "https://quickchart.io/chart?c=test"
    assert chart["chart_type"] == "bar"
    assert chart["periods"] == ["2024", "2025"]
    assert chart["series"][0]["name"] == "China"
