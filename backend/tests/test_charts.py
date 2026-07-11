"""Tests for structural table chartability analysis."""

from __future__ import annotations

from app.charts.profile import analyze_table_chartability
from app.charts.spec import validate_and_build_chart_spec
from app.charts.table_parse import parse_markdown_table
from app.ingestion.tables import table_to_markdown


def test_wide_table_bar_chart_profile():
    rows = [
        ["Metric", "2022", "2023", "2024"],
        ["Series A", "100", "110", "120"],
        ["Series B", "10", "12", "15"],
    ]
    profile = analyze_table_chartability(rows)
    assert profile is not None
    assert profile["chartable"] is True
    assert profile["orientation"] == "wide"
    assert profile["period_count"] == 3
    assert profile["metric_count"] == 2
    assert profile["suggested_chart_type"] == "bar"


def test_single_metric_line_chart_profile():
    rows = [
        ["Metric", "2020", "2021", "2022", "2023"],
        ["Series A", "50", "60", "70", "80"],
    ]
    profile = analyze_table_chartability(rows)
    assert profile is not None
    assert profile["suggested_chart_type"] == "line"
    assert profile["metric_count"] == 1
    assert profile["period_count"] == 4


def test_long_orientation_table_profile():
    rows = [
        ["Period", "Alpha", "Beta"],
        ["2022", "100", "10"],
        ["2023", "110", "12"],
        ["2024", "120", "15"],
    ]
    profile = analyze_table_chartability(rows)
    assert profile is not None
    assert profile["orientation"] == "long"
    assert profile["suggested_chart_type"] == "bar"


def test_rejects_non_period_numeric_grid():
    rows = [
        ["Col1", "Col2", "Col3"],
        ["10", "20", "30"],
        ["40", "50", "60"],
    ]
    assert analyze_table_chartability(rows) is None


def test_rejects_incomplete_numeric_grid():
    rows = [
        ["Metric", "2022", "2023", "2024"],
        ["Series A", "100", "", "120"],
    ]
    assert analyze_table_chartability(rows) is None


def test_rejects_too_many_periods():
    rows = [
        ["Metric", "2016", "2017", "2018", "2019", "2020", "2021", "2022"],
        ["Series A", "1", "2", "3", "4", "5", "6", "7"],
    ]
    profile = analyze_table_chartability(rows)
    assert profile is not None
    assert profile["period_count"] == 7


def test_many_periods_slice_to_query_limit():
    from app.charts.structural import build_chart_data_spec_from_structure

    rows = [
        ["Metric", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026", "2027"],
        ["Revenue", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
    ]
    markdown = table_to_markdown(rows)
    spec = build_chart_data_spec_from_structure(markdown, user_query="revenue 5", chart_type="line")
    assert spec is not None
    assert len(spec["labels"]) == 5
    assert len(spec["series"][0]["values"]) == 5


def test_rejects_composition_row_summing_to_100():
    rows = [
        ["Metric", "2022", "2023", "2024"],
        ["Share", "30", "35", "35"],
    ]
    assert analyze_table_chartability(rows) is None


def test_validate_and_build_spec_from_markdown():
    rows = [
        ["Metric", "2022", "2023", "2024"],
        ["Series A", "100", "110", "120"],
        ["Series B", "10", "12", "15"],
    ]
    profile = analyze_table_chartability(rows)
    markdown = table_to_markdown(rows)
    spec = validate_and_build_chart_spec(markdown, profile)
    assert spec is not None
    assert spec["chart_type"] == "bar"
    assert spec["periods"] == ["2022", "2023", "2024"]
    assert spec["value_axis_label"] == "Value"
    assert spec["period_axis_label"] == "Period"
    assert len(spec["series"]) == 2
    assert spec["series"][0]["values"] == [100.0, 110.0, 120.0]


def test_parse_markdown_table_roundtrip():
    rows = [
        ["Metric", "2022", "2023"],
        ["Series A", "1", "2"],
    ]
    markdown = table_to_markdown(rows)
    parsed = parse_markdown_table(markdown)
    assert parsed == rows


def test_embedded_year_header_with_annotation_column():
    rows = [
        ["Metric", "(CNY Million) 2025", "2024", "YoY"],
        ["Segment A", "375,014", "365,424", "2.6%"],
        ["Segment B", "100,000", "95,000", "5.3%"],
    ]
    profile = analyze_table_chartability(rows)
    assert profile is not None
    assert profile["orientation"] == "wide"
    assert profile["period_count"] == 2
    assert profile["metric_count"] == 2
    assert profile["period_labels"] == ["2024", "2025"]
    assert profile["value_axis_label"] == "CNY Million"
    assert profile["suggested_chart_type"] == "bar"


def test_resolves_duplicate_period_columns_with_conflicting_values():
    rows = [
        ["Metric", "2025 (USD Million)", "2025", "2024"],
        ["Segment A", "100", "200", "90"],
        ["Segment B", "10", "20", "8"],
    ]
    profile = analyze_table_chartability(rows)
    assert profile is not None
    assert profile["period_count"] == 2
    assert profile["period_labels"] == ["2024", "2025"]

    markdown = table_to_markdown(rows)
    spec = validate_and_build_chart_spec(markdown, profile)
    assert spec is not None
    assert spec["series"][0]["values"] == [90.0, 200.0]
    assert spec["series"][1]["values"] == [8.0, 20.0]


def test_extract_period_key_from_embedded_year():
    from app.charts.period import extract_period_key

    assert extract_period_key("(CNY Million) 2025") == "2025"
    assert extract_period_key("2023 (CNY Million)") == "2023"
    assert extract_period_key("YoY") is None


def test_detect_value_axis_label_from_period_headers():
    from app.charts.units import detect_value_axis_label

    rows = [
        ["Metric", "(CNY Million) 2025", "2024", "YoY"],
        ["Segment A", "100", "90", "11%"],
    ]
    label = detect_value_axis_label(rows, orientation="wide", period_column_indices=[1, 2])
    assert label == "CNY Million"


def test_detect_value_axis_label_fails_closed_on_conflict():
    from app.charts.units import detect_value_axis_label

    rows = [
        ["Metric", "(CNY Million) 2025", "(USD Million) 2024"],
        ["Segment A", "100", "90"],
    ]
    label = detect_value_axis_label(rows, orientation="wide", period_column_indices=[1, 2])
    assert label == "Value"
