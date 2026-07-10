"""Tests for chart data spec LLM helper and Chart.js builder."""

from __future__ import annotations

import json

from app.charts.llm_config import _validate_chart_data_spec, extract_chart_data_spec
from app.charts.quickchart import chartjs_config_from_data_spec


def test_validate_chart_data_spec_accepts_aligned_series():
    spec = {
        "chart_type": "bar",
        "title": "Regional revenue",
        "labels": ["2024", "2025"],
        "series": [
            {"name": "China", "values": [615264.0, 616249.0]},
            {"name": "EMEA", "values": [148355.0, 161356.0]},
        ],
    }
    assert _validate_chart_data_spec(spec) is None


def test_validate_chart_data_spec_rejects_mismatched_lengths():
    spec = {
        "chart_type": "bar",
        "labels": ["2024", "2025"],
        "series": [{"name": "China", "values": [100.0]}],
    }
    assert _validate_chart_data_spec(spec) is not None


def test_extract_chart_data_spec_parses_llm_json(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    monkeypatch.setattr(settings, "chart_llm_model", "openai/gpt-oss-20b")
    monkeypatch.setattr(settings, "chart_table_max_chars", 4000)

    llm_json = {
        "chart_type": "bar",
        "title": "Regional revenue",
        "labels": ["2024", "2025"],
        "series": [{"name": "EMEA", "values": [148355, 161356]}],
    }

    class _Resp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"choices": [{"message": {"content": json.dumps(llm_json)}}]}

    class _Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, *args, **kwargs):
            return _Resp()

    monkeypatch.setattr("app.charts.llm_config.httpx.Client", _Client)

    spec, error = extract_chart_data_spec(
        "chart regional revenue",
        "| Region | 2024 | 2025 |\n| --- | --- | --- |\n| EMEA | 148355 | 161356 |",
        chart_type="bar",
    )
    assert error is None
    assert spec is not None
    assert spec["chart_type"] == "bar"
    assert len(spec["series"]) == 1


def test_chartjs_config_from_data_spec_builds_valid_chartjs():
    spec = {
        "chart_type": "bar",
        "title": "China revenue",
        "labels": ["2024", "2025"],
        "series": [{"name": "China", "values": [615264, 616249]}],
    }
    config = chartjs_config_from_data_spec(spec)
    assert config["type"] == "bar"
    assert config["data"]["labels"] == ["2024", "2025"]
    assert config["data"]["datasets"][0]["data"] == [615264.0, 616249.0]
    assert config["options"]["plugins"]["title"]["text"] == "China revenue"
