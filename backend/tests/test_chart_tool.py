"""Tests for the create_chart agent tool."""

from __future__ import annotations

import json

import pytest

from app.llm.tools import execute_create_chart
from app.retrieval.models import RetrievedChunk


def _table_chunk(*, chunk_id: str = "t1", content: str, profile: dict | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="d1",
        filename="report.pdf",
        page_number=4,
        chunk_type="table",
        content=content,
        score=0.9,
        extra_metadata={"chart_profile": profile} if profile else None,
    )


def test_create_chart_returns_bar_chart(monkeypatch):
    content = (
        "| Metric | 2022 | 2023 | 2024 |\n"
        "| --- | --- | --- | --- |\n"
        "| Series A | 100 | 110 | 120 |\n"
        "| Series B | 10 | 12 | 15 |"
    )
    profile = {
        "chartable": True,
        "orientation": "wide",
        "period_count": 3,
        "metric_count": 2,
        "suggested_chart_type": "bar",
    }
    chunk = _table_chunk(content=content, profile=profile)

    def fake_retrieve(client, query, user_id, top_k=None, doc_id=None):
        from app.retrieval.models import SearchResponse

        return SearchResponse(query=query, top_k=top_k or 8, doc_id=doc_id, total=1, results=[chunk])

    monkeypatch.setattr("app.llm.tools.hybrid_retrieve", fake_retrieve)

    payload_json, charts = execute_create_chart(
        object(),
        user_id=1,
        query="series revenue",
        chart_type="bar",
    )
    payload = json.loads(payload_json)
    assert payload["status"] == "created"
    assert len(charts) == 1
    assert charts[0]["chart_type"] == "bar"
    assert charts[0]["derivation"] == "tool"


def test_create_chart_reports_not_chartable(monkeypatch):
    text_chunk = RetrievedChunk(
        chunk_id="x1",
        doc_id="d1",
        filename="report.pdf",
        page_number=1,
        chunk_type="text",
        content="plain text",
        score=0.5,
    )

    def fake_retrieve(client, query, user_id, top_k=None, doc_id=None):
        from app.retrieval.models import SearchResponse

        return SearchResponse(query=query, top_k=top_k or 8, doc_id=doc_id, total=1, results=[text_chunk])

    monkeypatch.setattr("app.llm.tools.hybrid_retrieve", fake_retrieve)

    payload_json, charts = execute_create_chart(
        object(),
        user_id=1,
        query="anything",
        chart_type="bar",
    )
    payload = json.loads(payload_json)
    assert payload["status"] == "not_chartable"
    assert "cannot be created" in payload["message"].lower()
    assert charts == []


def test_create_chart_pie_from_category_value(monkeypatch):
    content = (
        "| Region | Revenue |\n"
        "| --- | --- |\n"
        "| China | 375 |\n"
        "| EMEA | 100 |\n"
        "| Americas | 50 |"
    )
    chunk = _table_chunk(content=content)

    def fake_retrieve(client, query, user_id, top_k=None, doc_id=None):
        from app.retrieval.models import SearchResponse

        return SearchResponse(query=query, top_k=top_k or 8, doc_id=doc_id, total=1, results=[chunk])

    monkeypatch.setattr("app.llm.tools.hybrid_retrieve", fake_retrieve)

    payload_json, charts = execute_create_chart(
        object(),
        user_id=1,
        query="region revenue",
        chart_type="pie",
    )
    payload = json.loads(payload_json)
    assert payload["status"] == "created"
    assert charts[0]["chart_type"] == "pie"


@pytest.mark.asyncio
async def test_agent_create_chart_tool(monkeypatch) -> None:
    from app.llm.agent import iter_agent_turn
    from tests.test_agent import _router_then_stop

    chart_round = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_chart",
                            "type": "function",
                            "function": {
                                "name": "create_chart",
                                "arguments": json.dumps(
                                    {"query": "segment revenue", "chart_type": "bar"}
                                ),
                            },
                        }
                    ],
                }
            }
        ]
    }

    def fake_create_chart(
        client,
        user_id,
        query,
        chart_type=None,
        doc_id=None,
        default_doc_id=None,
        chunk_id=None,
        period_label=None,
        top_k=None,
    ):
        chart = {
            "chart_type": "bar",
            "chunk_id": "t1",
            "filename": "report.pdf",
            "page_number": 4,
            "periods": ["2024"],
            "series": [{"name": "A", "values": [1.0]}],
            "derivation": "tool",
        }
        return json.dumps({"status": "created", "chart_type": "bar"}), [chart]

    monkeypatch.setattr("app.llm.agent.groq_chat_completion", _router_then_stop(chart_round))
    monkeypatch.setattr("app.llm.agent.execute_create_chart", fake_create_chart)

    turn = None
    async for event in iter_agent_turn(
        client=object(),
        user_id=1,
        user_query="create a bar chart of segment revenue",
    ):
        if event["type"] == "complete":
            turn = event["result"]

    assert turn is not None
    assert "create_chart" in turn.tools_used
    assert len(turn.tool_charts) == 1
    assert turn.tool_charts[0]["chart_type"] == "bar"
