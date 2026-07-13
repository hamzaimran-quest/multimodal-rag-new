"""Agent router and tool execution tests."""

from __future__ import annotations

import json

import pytest

from typing import Any

from app.llm.agent import (
    PHASE_A_TOOLS,
    _chart_routing_hint,
    _is_clarification_reply,
    _merge_retrieved_chunks,
    _recover_tool_calls_from_failed_generation,
    iter_agent_turn,
    run_agent_turn,
)
from app.retrieval.models import RetrievedChunk


def test_chart_routing_hint_injected_for_plot_queries():
    hint = _chart_routing_hint("draw a chart of revenue")
    assert "create_chart" in hint
    assert _chart_routing_hint("what was revenue in 2024") == ""


def _router_then_stop(*round_responses: dict) -> Any:
    """Build a fake groq completion that stops after the given tool rounds."""
    call_idx = {"n": 0}

    async def fake_completion(*, messages, tools=None, model=None, temperature=0.1, **kwargs):
        idx = call_idx["n"]
        call_idx["n"] += 1
        if idx < len(round_responses):
            return round_responses[idx]
        return {"choices": [{"message": {"content": "", "tool_calls": []}}]}

    return fake_completion


def _chunk(chunk_id: str, content: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="d1",
        filename="huawei.pdf",
        page_number=2,
        chunk_type="text",
        content=content,
        score=score,
    )


@pytest.mark.asyncio
async def test_agent_direct_reply_skips_tools(monkeypatch) -> None:
    async def fake_completion(*, messages, tools=None, model=None, temperature=0.1, **kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": "Hello! I can help you search your uploaded documents.",
                        "tool_calls": [],
                    }
                }
            ]
        }

    monkeypatch.setattr("app.llm.agent.groq_chat_completion", fake_completion)

    result = await run_agent_turn(
        client=object(),
        user_id=1,
        user_query="hi there",
        tools=PHASE_A_TOOLS,
    )
    assert result.direct_answer is not None
    assert result.tools_used == []
    assert result.retrieved_chunks == []


@pytest.mark.asyncio
async def test_agent_search_documents_tool(monkeypatch) -> None:
    search_round = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search_documents",
                                "arguments": json.dumps({"query": "revenue 2024"}),
                            },
                        }
                    ],
                }
            }
        ]
    }

    def fake_search(client, user_id, query, top_k=None, scope_doc_ids=None, **kwargs):
        return (
            json.dumps({"total": 1, "chunks": []}),
            [_chunk("c1", "Revenue grew in 2024.")],
        )

    monkeypatch.setattr("app.llm.agent.groq_chat_completion", _router_then_stop(search_round))
    monkeypatch.setattr("app.llm.agent.execute_search_documents", fake_search)

    result = await run_agent_turn(
        client=object(),
        user_id=1,
        user_query="what was revenue in 2024?",
        tools=PHASE_A_TOOLS,
    )
    assert result.direct_answer is None
    assert result.tools_used == ["search_documents"]
    assert len(result.retrieved_chunks) == 1


@pytest.mark.asyncio
async def test_agent_forces_search_when_router_skips_tools(monkeypatch) -> None:
    async def fake_completion(*, messages, tools=None, model=None, temperature=0.1, **kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": "Sun Yafang is the chairwoman of Huawei.",
                        "tool_calls": [],
                    }
                }
            ]
        }

    def fake_search(client, user_id, query, top_k=None, scope_doc_ids=None, **kwargs):
        return (
            json.dumps({"total": 0, "chunks": []}),
            [],
        )

    monkeypatch.setattr("app.llm.agent.groq_chat_completion", fake_completion)
    monkeypatch.setattr("app.llm.agent.execute_search_documents", fake_search)

    result = await run_agent_turn(
        client=object(),
        user_id=1,
        user_query="who's the chairwoman",
        tools=PHASE_A_TOOLS,
    )
    assert result.direct_answer is None
    assert result.tools_used == ["search_documents"]
    assert result.retrieved_chunks == []


@pytest.mark.asyncio
async def test_agent_list_documents_tool(monkeypatch) -> None:
    list_round = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "list_documents",
                                "arguments": "{}",
                            },
                        }
                    ],
                }
            }
        ]
    }

    monkeypatch.setattr("app.llm.agent.groq_chat_completion", _router_then_stop(list_round))
    monkeypatch.setattr(
        "app.llm.agent.execute_list_documents",
        lambda client, user_id, scope_doc_ids=None: json.dumps({"documents": [{"filename": "a.pdf"}], "total": 1}),
    )

    result = await run_agent_turn(
        client=object(),
        user_id=1,
        user_query="what documents do I have?",
        tools=PHASE_A_TOOLS,
    )
    assert result.tools_used == ["list_documents"]
    assert result.retrieved_chunks == []


@pytest.mark.asyncio
async def test_agent_multi_round_list_then_search(monkeypatch) -> None:
    calls = {"n": 0}

    async def fake_completion(*, messages, tools=None, model=None, temperature=0.1, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_list",
                                    "type": "function",
                                    "function": {"name": "list_documents", "arguments": "{}"},
                                }
                            ],
                        }
                    }
                ]
            }
        if calls["n"] == 2:
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_search",
                                    "type": "function",
                                    "function": {
                                        "name": "search_documents",
                                        "arguments": json.dumps(
                                            {"query": "chairwoman board", "doc_id": "d1"}
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {"choices": [{"message": {"content": "", "tool_calls": []}}]}

    def fake_search(client, user_id, query, top_k=None, scope_doc_ids=None, **kwargs):
        return (
            json.dumps({"total": 1, "chunks": []}),
            [_chunk("c-chair", "Liang Hua serves as Chairman.")],
        )

    monkeypatch.setattr("app.llm.agent.groq_chat_completion", fake_completion)
    monkeypatch.setattr(
        "app.llm.agent.execute_list_documents",
        lambda client, user_id, scope_doc_ids=None: json.dumps(
            {"documents": [{"doc_id": "d1", "filename": "huawei.pdf"}], "total": 1}
        ),
    )
    monkeypatch.setattr("app.llm.agent.execute_search_documents", fake_search)

    monkeypatch.setattr("app.llm.agent.settings.agent_max_rounds", 3)

    result = await run_agent_turn(
        client=object(),
        user_id=1,
        user_query="who is the chairwoman in my huawei file?",
        tools=PHASE_A_TOOLS,
    )
    assert result.tools_used == ["list_documents", "search_documents"]
    assert result.rounds_used == 3
    assert len(result.retrieved_chunks) == 1


@pytest.mark.asyncio
async def test_agent_scoped_clarification_forces_search(monkeypatch) -> None:
    async def fake_completion(*, messages, tools=None, model=None, temperature=0.1, **kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            "Could you clarify what you'd like to know about lists? "
                            "For example, programming or document formatting?"
                        ),
                        "tool_calls": [],
                    }
                }
            ]
        }

    def fake_search(client, user_id, query, top_k=None, scope_doc_ids=None, **kwargs):
        assert scope_doc_ids == ["doc-demo"]
        return (
            json.dumps({"total": 1, "chunks": []}),
            [_chunk("c-list", "All types of lists are supported by the conversion.")],
        )

    monkeypatch.setattr("app.llm.agent.groq_chat_completion", fake_completion)
    monkeypatch.setattr("app.llm.agent.execute_search_documents", fake_search)

    result = await run_agent_turn(
        client=object(),
        user_id=1,
        user_query="tell me about lists",
        scope_doc_ids=["doc-demo"],
        scoped_filenames=["demo.docx"],
        tools=PHASE_A_TOOLS,
    )
    assert result.direct_answer is None
    assert result.is_clarification is False
    assert result.tools_used == ["search_documents"]
    assert len(result.retrieved_chunks) == 1


@pytest.mark.asyncio
async def test_agent_clarification_direct_reply(monkeypatch) -> None:
    async def fake_completion(*, messages, tools=None, model=None, temperature=0.1, **kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": "Could you clarify which fiscal year you mean?",
                        "tool_calls": [],
                    }
                }
            ]
        }

    monkeypatch.setattr("app.llm.agent.groq_chat_completion", fake_completion)

    result = await run_agent_turn(
        client=object(),
        user_id=1,
        user_query="what was revenue last year?",
        tools=PHASE_A_TOOLS,
    )
    assert result.direct_answer is not None
    assert result.is_clarification is True
    assert result.tools_used == []


@pytest.mark.asyncio
async def test_agent_iter_yields_tool_events(monkeypatch) -> None:
    search_round = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search_documents",
                                "arguments": json.dumps({"query": "revenue"}),
                            },
                        }
                    ],
                }
            }
        ]
    }

    monkeypatch.setattr("app.llm.agent.groq_chat_completion", _router_then_stop(search_round))
    monkeypatch.setattr(
        "app.llm.agent.execute_search_documents",
        lambda *a, **k: (json.dumps({"total": 0}), []),
    )

    events: list[dict] = []
    async for event in iter_agent_turn(
        client=object(),
        user_id=1,
        user_query="revenue?",
        tools=PHASE_A_TOOLS,
    ):
        events.append(event)

    tool_events = [e for e in events if e["type"] == "tool"]
    assert [e["status"] for e in tool_events] == ["running", "complete"]
    assert events[-1]["type"] == "complete"


@pytest.mark.asyncio
async def test_agent_router_receives_rewritten_query_only(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_rewrite(user_query, prior_queries, last_assistant_reply=None):
        return "standalone rewritten query"

    async def fake_completion(*, messages, tools=None, model=None, temperature=0.1, **kwargs):
        if "messages" not in captured:
            captured["messages"] = [dict(m) for m in messages]
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "search_documents",
                                    "arguments": '{"query":"standalone rewritten query"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

    def fake_search(client, user_id, query, top_k=None, scope_doc_ids=None, **kwargs):
        return (json.dumps({"total": 0}), [])

    monkeypatch.setattr("app.llm.agent.rewrite_query_for_retrieval", fake_rewrite)
    monkeypatch.setattr("app.llm.agent.groq_chat_completion", fake_completion)
    monkeypatch.setattr("app.llm.agent.execute_search_documents", fake_search)

    await run_agent_turn(
        client=object(),
        user_id=1,
        user_query="what about him?",
        prior_queries=["who is the chairman"],
        tools=PHASE_A_TOOLS,
    )

    messages = captured["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "standalone rewritten query"}


def test_merge_retrieved_chunks_dedupes_and_keeps_best_score() -> None:
    merged = _merge_retrieved_chunks(
        [_chunk("c1", "old", score=0.5)],
        [_chunk("c1", "new", score=0.9), _chunk("c2", "other", score=0.7)],
    )
    assert len(merged) == 2
    by_id = {c.chunk_id: c for c in merged}
    assert by_id["c1"].content == "new"
    assert by_id["c1"].score == 0.9


def test_is_clarification_reply() -> None:
    assert not _is_clarification_reply("Which document should I search?")
    assert _is_clarification_reply("Could you clarify what time period you mean?")
    assert not _is_clarification_reply("Revenue was 100 billion.")


def test_recover_tool_calls_from_groq_failed_generation() -> None:
    failed = '<function=search_documents{"query": "chairwoman", "top_k": 10}</function>'
    recovered = _recover_tool_calls_from_failed_generation(failed)
    assert recovered is not None
    assert len(recovered) == 1
    assert recovered[0]["function"]["name"] == "search_documents"
    assert json.loads(recovered[0]["function"]["arguments"]) == {
        "query": "chairwoman",
        "top_k": 10,
    }


def test_recover_tool_calls_returns_none_for_unparseable_text() -> None:
    assert _recover_tool_calls_from_failed_generation("plain text answer") is None
