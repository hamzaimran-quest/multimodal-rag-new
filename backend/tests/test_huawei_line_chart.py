"""Regression tests for multi-year financial statement line charts."""

from __future__ import annotations

from app.charts.build import attempt_chart_from_chunk
from app.charts.structural import build_chart_data_spec_from_structure
from app.retrieval.models import RetrievedChunk

_HUAWEI_FINANCIAL_MARKDOWN = (
    "| Metric | 2025 (USD Million) (CNY Million) | 2025 | 2024 | 2023 (CNY Million) | 2022 | 2021 |\n"
    "| --- | --- | --- | --- | --- | --- | --- |\n"
    "| Revenue | 126,018 | 880,941 | 862,072 | 704,174 | 642,338 | 636,807 |\n"
    "| Operating profit | 13,867 | 96,937 | 79,361 | 104,401 | 42,216 | 121,412 |\n"
    "| Operating margin | 11.0% | 11.0% | 9.2% | 14.8% | 6.6% | 19.1% |\n"
    "| Net profit | 9,732 | 68,036 | 62,574 | 86,950 | 35,562 | 113,718 |\n"
    "| Cash flow from operating activities Cash and short-term | 18,222 | 127,384 | 88,417 | 69,807 | 17,797 | 59,670 |\n"
    "|  | 51,702 | 361,426 | 372,232 | 475,317 | 373,452 | 416,334 |\n"
    "| investments Working capital | 43,523 | 304,252 | 319,178 | 421,662 | 344,938 | 376,923 |\n"
    "| Total assets | 190,961 | 1,334,930 | 1,290,149 | 1,263,597 | 1,063,804 | 982,971 |\n"
    "| Total borrowings | 34,229 | 239,284 | 264,871 | 308,414 | 197,144 | 175,100 |\n"
    "| Equity | 85,847 | 600,120 | 544,619 | 507,568 | 437,076 | 414,652 |\n"
    "| Liability ratio | 55.0% | 55.0% | 57.8% | 59.8% | 58.9% | 57.8% |"
)


def test_build_chart_data_spec_huawei_revenue_line_chart():
    spec = build_chart_data_spec_from_structure(
        _HUAWEI_FINANCIAL_MARKDOWN,
        user_query="Huawei revenue 2021-2025 line graph",
        chart_type="line",
    )
    assert spec is not None
    assert spec["chart_type"] == "line"
    assert spec["labels"] == ["2021", "2022", "2023", "2024", "2025"]
    assert len(spec["series"]) == 1
    assert spec["series"][0]["name"] == "Revenue"
    assert spec["series"][0]["values"] == [636807.0, 642338.0, 704174.0, 862072.0, 880941.0]


def test_attempt_chart_from_chunk_huawei_line_emits_quickchart(monkeypatch):
    chunk = RetrievedChunk(
        chunk_id="huawei-financial",
        doc_id="d1",
        filename="huawei.pdf",
        page_number=9,
        chunk_type="table",
        content=_HUAWEI_FINANCIAL_MARKDOWN,
        score=0.9,
    )
    captured: dict[str, object] = {}

    def capture_config(config):
        captured["config"] = config
        return "https://quickchart.io/chart?c=huawei-line"

    def fail_llm(*args, **kwargs):
        raise AssertionError("LLM should not run when structural profiling succeeds")

    monkeypatch.setattr("app.charts.build.extract_chart_data_spec", fail_llm)
    monkeypatch.setattr("app.charts.build.build_quickchart_url", capture_config)

    chart, error = attempt_chart_from_chunk(
        chunk,
        user_query="Huawei revenue 2021-2025 line graph",
        chart_type="line",
    )
    assert error is None
    assert chart is not None
    assert chart["chart_url"] == "https://quickchart.io/chart?c=huawei-line"
    assert chart["chart_type"] == "line"

    config = captured["config"]
    assert config["type"] == "line"
    dataset = config["data"]["datasets"][0]
    assert dataset["label"] == "Revenue"
    assert dataset["fill"] is False
    assert "borderColor" in dataset
