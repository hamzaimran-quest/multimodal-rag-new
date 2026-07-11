"""Tests for the Excel entity-grid chart pipeline."""

from __future__ import annotations

from app.charts.build import attempt_chart_from_chunk
from app.charts.excel_build import (
    analyze_excel_chartability,
    build_excel_chart_data_spec,
    classify_entity_grid_layout,
)
from app.retrieval.models import RetrievedChunk

_FSI_HEADERS = [
    "Country",
    "Year",
    "Rank",
    "Total",
    "S1: Demographic Pressures",
    "S2: Refugees and IDPs",
    "C3: Group Grievance",
    "E3: Human Flight and Brain Drain",
    "E2: Economic Inequality",
    "E1: Economy",
    "P1: State Legitimacy",
]

_SOMALIA_ROW = [
    "Somalia",
    "2023",
    "5",
    "109.8",
    "9.8",
    "9.5",
    "8.6",
    "9.1",
    "7.5",
    "8.2",
    "9.4",
]

_KENYA_ROW = [
    "Kenya",
    "2023",
    "35",
    "98.4",
    "7.8",
    "6.9",
    "7.1",
    "6.5",
    "6.8",
    "7.0",
    "8.1",
]


def _fsi_rows(*data_rows: list[str]) -> list[list[str]]:
    return [_FSI_HEADERS, *data_rows]


def _xlsx_chunk(
    *,
    chunk_id: str = "fsi-1",
    content: str,
    headers: list[str] | None = None,
    entity_keys: list[str] | None = None,
) -> RetrievedChunk:
    extra = {
        "source_format": "xlsx",
        "content_format": "slim_rows",
        "table_headers": headers or _FSI_HEADERS,
        "entity_key_column": "Country",
        "entity_keys": entity_keys or ["Somalia"],
        "row_entity_keys": {"2": "Somalia"},
        "sheet_name": "2023",
    }
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="d-fsi",
        filename="FSI-2023-DOWNLOAD.xlsx",
        page_number=1,
        chunk_type="table",
        content=content,
        score=0.9,
        extra_metadata=extra,
    )


def test_classify_entity_grid_layout_accepts_many_metric_columns():
    headers = _FSI_HEADERS + [
        "P2: Public Services",
        "P3: Human Rights",
        "C1: Security Apparatus",
        "C2: Factionalized Elites",
        "X1: External Intervention",
    ]
    row = _SOMALIA_ROW + ["9.0", "9.5", "9.6", "9.2", "9.1"]
    layout = classify_entity_grid_layout([headers, row])
    assert layout is not None
    assert len(layout.metric_column_indices) >= 12


def test_build_excel_chart_data_spec_slices_many_metrics_to_requested_count():
    headers = _FSI_HEADERS + [
        "P2: Public Services",
        "P3: Human Rights",
        "C1: Security Apparatus",
        "C2: Factionalized Elites",
        "X1: External Intervention",
    ]
    row = _SOMALIA_ROW + ["9.0", "9.5", "9.6", "9.2", "9.1"]
    spec = build_excel_chart_data_spec(
        [headers, row],
        user_query="Somalia plot 5 metrics",
        extra_metadata={"entity_key_column": "Country"},
    )
    assert spec is not None
    assert len(spec["labels"]) == 5


def test_classify_entity_grid_layout_detects_fsi_shape():
    layout = classify_entity_grid_layout(_fsi_rows(_SOMALIA_ROW))
    assert layout is not None
    assert layout.entity_column_index == 0
    assert "S1: Demographic Pressures" in layout.metric_labels
    assert "Total" in layout.metric_labels


def test_analyze_excel_chartability_marks_entity_grid():
    profile = analyze_excel_chartability(_fsi_rows(_SOMALIA_ROW))
    assert profile is not None
    assert profile["orientation"] == "entity_grid"
    assert profile["suggested_chart_type"] == "bar"
    assert profile["metric_count"] == 1
    assert profile["period_count"] >= 5


def test_build_excel_chart_data_spec_first_five_metrics():
    spec = build_excel_chart_data_spec(
        _fsi_rows(_SOMALIA_ROW),
        user_query="Somalia 2023 plot the first 5 metrics",
        extra_metadata={"entity_key_column": "Country"},
    )
    assert spec is not None
    assert spec["chart_type"] == "bar"
    assert len(spec["labels"]) == 5
    assert spec["labels"][0] == "S1: Demographic Pressures"
    assert spec["series"][0]["name"] == "Somalia"
    assert spec["series"][0]["values"] == [9.8, 9.5, 8.6, 9.1, 7.5]
    assert "Somalia" in spec["title"]
    assert "2023" in spec["title"]


def test_build_excel_chart_data_spec_selects_matching_entity_row():
    spec = build_excel_chart_data_spec(
        _fsi_rows(_SOMALIA_ROW, _KENYA_ROW),
        user_query="Kenya 2023 indicators",
        extra_metadata={"entity_key_column": "Country"},
    )
    assert spec is not None
    assert spec["series"][0]["name"] == "Kenya"
    assert spec["series"][0]["values"][0] == 7.8


def test_build_excel_chart_data_spec_rejects_ambiguous_multi_entity_without_match():
    spec = build_excel_chart_data_spec(
        _fsi_rows(_SOMALIA_ROW, _KENYA_ROW),
        user_query="plot metrics",
        extra_metadata={"entity_key_column": "Country"},
    )
    assert spec is None


def test_attempt_excel_chart_skips_pdf_llm_path(monkeypatch):
    content = "Somalia | 2023 | 5 | 109.8 | 9.8 | 9.5 | 8.6 | 9.1 | 7.5 | 8.2 | 9.4"
    chunk = _xlsx_chunk(content=content)

    def fail_llm(*args, **kwargs):
        raise AssertionError("PDF/LLM chart path must not run for Excel chunks")

    def fail_pdf_structural(*args, **kwargs):
        raise AssertionError("PDF structural chart path must not run for Excel chunks")

    monkeypatch.setattr("app.charts.build.extract_chart_data_spec", fail_llm)
    monkeypatch.setattr("app.charts.build.build_chart_data_spec_from_structure", fail_pdf_structural)
    monkeypatch.setattr(
        "app.charts.build.build_quickchart_url",
        lambda config: "https://quickchart.io/chart?c=excel",
    )

    chart, error = attempt_chart_from_chunk(
        chunk,
        user_query="Somalia plot the first 5 metrics",
        chart_type="bar",
    )
    assert error is None
    assert chart is not None
    assert chart["chart_url"] == "https://quickchart.io/chart?c=excel"
    assert len(chart["periods"]) == 5
    assert chart["series"][0]["name"] == "Somalia"


def test_attempt_pdf_chart_still_uses_structural_path(monkeypatch):
    markdown = (
        "| Metric | 2020 | 2021 | 2022 | 2023 |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| Revenue | 50 | 60 | 70 | 80 |"
    )
    chunk = RetrievedChunk(
        chunk_id="pdf-1",
        doc_id="d1",
        filename="report.pdf",
        page_number=2,
        chunk_type="table",
        content=markdown,
        score=0.9,
        extra_metadata={"source_format": "pdf"},
    )

    def fail_excel(*args, **kwargs):
        raise AssertionError("Excel chart path must not run for PDF chunks")

    monkeypatch.setattr("app.charts.build.build_excel_chart_data_spec_from_chunk", fail_excel)
    monkeypatch.setattr(
        "app.charts.build.build_quickchart_url",
        lambda config: "https://quickchart.io/chart?c=pdf",
    )

    chart, error = attempt_chart_from_chunk(chunk, user_query="revenue trend", chart_type="line")
    assert error is None
    assert chart is not None
    assert chart["chart_type"] == "line"
