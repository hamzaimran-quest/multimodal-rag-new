"""Tests for chart output merging."""

from __future__ import annotations

from app.charts.build import merge_chart_outputs


def test_merge_chart_outputs_prefers_later_tool_chart():
    first = [
        {
            "chunk_id": "t1",
            "chart_type": "bar",
            "derivation": "tool",
        }
    ]
    second = [
        {
            "chunk_id": "t1",
            "chart_type": "line",
            "derivation": "tool",
        }
    ]
    merged = merge_chart_outputs(first, second)
    assert len(merged) == 1
    assert merged[0]["chart_type"] == "line"
