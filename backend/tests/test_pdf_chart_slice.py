"""Tests for PDF chart spec slicing."""

from __future__ import annotations

from app.charts.slice import slice_pdf_chart_spec
from app.charts.structural import build_chart_data_spec_from_structure


def test_slice_pdf_chart_spec_limits_many_periods():
    periods = [str(year) for year in range(2016, 2029)]
    values = [float(index) for index in range(len(periods))]
    series = [{"name": "Revenue", "values": values}]

    periods_out, series_out = slice_pdf_chart_spec(periods, series, user_query="plot trend")
    assert len(periods_out) == 12
    assert len(series_out[0]["values"]) == 12


def test_slice_pdf_chart_spec_honors_query_integer_for_periods():
    periods = [str(year) for year in range(2016, 2026)]
    values = [float(index) for index in range(len(periods))]
    series = [{"name": "Revenue", "values": values}]

    periods_out, series_out = slice_pdf_chart_spec(periods, series, user_query="plot 5 periods")
    assert periods_out == ["2016", "2017", "2018", "2019", "2020"]
    assert series_out[0]["values"] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_build_chart_data_spec_slices_many_metric_rows():
    markdown = (
        "| Metric | 2024 | 2025 |\n"
        "| --- | --- | --- |\n"
        "| Alpha | 1 | 2 |\n"
        "| Beta | 3 | 4 |\n"
        "| Gamma | 5 | 6 |\n"
        "| Delta | 7 | 8 |\n"
        "| Epsilon | 9 | 10 |\n"
        "| Zeta | 11 | 12 |"
    )
    spec = build_chart_data_spec_from_structure(markdown, user_query="plot 3 metrics", chart_type="bar")
    assert spec is not None
    assert len(spec["series"]) == 3
    assert spec["series"][0]["name"] == "Alpha"
