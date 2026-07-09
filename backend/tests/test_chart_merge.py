"""Tests for chart output merging."""

from __future__ import annotations

from app.charts.build import merge_chart_outputs


def test_merge_chart_outputs_prefers_later_tool_chart():
    computed = [
        {
            "chunk_id": "t1",
            "chart_type": "bar",
            "derivation": "computed",
        }
    ]
    tool = [
        {
            "chunk_id": "t1",
            "chart_type": "pie",
            "derivation": "tool",
        }
    ]
    merged = merge_chart_outputs(computed, tool)
    assert len(merged) == 1
    assert merged[0]["chart_type"] == "pie"
    assert merged[0]["derivation"] == "tool"
