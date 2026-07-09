"""Tests for pie chart analysis and spec construction."""

from __future__ import annotations

from app.charts.build import build_chart_spec
from app.charts.pie import analyze_pie_chartability, build_pie_from_time_series


def test_category_value_pie_profile():
    rows = [
        ["Region", "Revenue"],
        ["China", "375"],
        ["EMEA", "100"],
        ["Americas", "50"],
    ]
    profile = analyze_pie_chartability(rows)
    assert profile is not None
    assert profile["suggested_chart_type"] == "pie"
    assert profile["slice_count"] == 3


def test_build_pie_spec_from_category_value_table():
    markdown = (
        "| Region | Revenue |\n"
        "| --- | --- |\n"
        "| China | 375 |\n"
        "| EMEA | 100 |\n"
        "| Americas | 50 |"
    )
    spec = build_chart_spec(markdown, chart_type="pie")
    assert spec is not None
    assert spec["chart_type"] == "pie"
    assert spec["periods"] == ["China", "EMEA", "Americas"]
    assert spec["series"][0]["values"] == [375.0, 100.0, 50.0]


def test_build_pie_from_multi_period_table_uses_selected_period():
    periods = ["2023", "2024"]
    series = [
        {"name": "China", "values": [300.0, 375.0]},
        {"name": "EMEA", "values": [90.0, 100.0]},
        {"name": "Americas", "values": [40.0, 50.0]},
    ]
    spec = build_pie_from_time_series(
        periods=periods,
        series=series,
        period_label="2024",
        value_axis_label="CNY Million",
    )
    assert spec is not None
    assert spec["chart_type"] == "pie"
    assert spec["periods"] == ["China", "EMEA", "Americas"]
    assert spec["series"][0]["values"] == [375.0, 100.0, 50.0]


def test_rejects_pie_for_non_positive_values():
    rows = [
        ["Region", "Revenue"],
        ["China", "-10"],
        ["EMEA", "100"],
    ]
    assert analyze_pie_chartability(rows) is None
