"""Tests for turn-level source intent resolution and gating."""

from __future__ import annotations

import json

import pytest

from app.llm.agent import (
    AGENT_TOOLS,
    SqlToolContext,
    _needs_sql_supplement,
    build_agent_tools,
    build_query_database_tool,
    iter_agent_turn,
    should_stop_after_tool_round,
    user_restricted_to_documents_only,
)
from app.llm.source_intent import (
    SourceIntent,
    SourceIntentMethod,
    SourceIntentResult,
    docs_allowed_for_intent,
    match_source_intent_regex,
    resolve_source_intent,
    sql_allowed_for_intent,
)
from app.retrieval.models import RetrievedChunk


def _stub_agent_scope(monkeypatch: pytest.MonkeyPatch, *, indexed: int = 1) -> None:
    monkeypatch.setattr(
        "app.llm.agent.count_indexed_documents_in_scope",
        lambda *args, **kwargs: indexed,
    )
    monkeypatch.setattr(
        "app.llm.agent.sources_hint_for_agent",
        lambda *args, **kwargs: f"\n\n## Available sources\nDocuments: {indexed} indexed.",
    )


def test_regex_matches_from_the_doc() -> None:
    assert match_source_intent_regex("tell me the revenue from 2024 from the doc") == SourceIntent.DOCS_ONLY
    assert user_restricted_to_documents_only("tell me the revenue from 2024 from the doc")


def test_regex_matches_in_the_excerpt() -> None:
    assert (
        match_source_intent_regex("what did the chairman state in the excerpt")
        == SourceIntent.DOCS_ONLY
    )


def test_regex_matches_database_only() -> None:
    assert match_source_intent_regex("compare revenue from the database only") == SourceIntent.DB_ONLY


def test_regex_skips_ambiguous_factual() -> None:
    assert match_source_intent_regex("compare revenue 2024 and 2025") is None


@pytest.mark.asyncio
async def test_resolve_falls_back_when_classifier_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.llm.source_intent.settings.source_intent_classifier_enabled",
        False,
    )
    result = await resolve_source_intent("what was total revenue in 2025?")
    assert result.intent == SourceIntent.AMBIGUOUS
    assert result.method == SourceIntentMethod.FALLBACK


@pytest.mark.asyncio
async def test_resolve_uses_llm_when_regex_misses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.llm.source_intent.settings.source_intent_classifier_enabled", True)
    monkeypatch.setattr("app.llm.source_intent.settings.groq_api_key", "test-key")

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"intent": "docs_only", "reason": "asked for uploaded reports"}
                            )
                        }
                    }
                ]
            }

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            return _Resp()

    monkeypatch.setattr("app.llm.source_intent.httpx.AsyncClient", lambda **kwargs: _Client())
    result = await resolve_source_intent("use my uploaded reports for revenue")
    assert result.intent == SourceIntent.DOCS_ONLY
    assert result.method == SourceIntentMethod.LLM


def test_sql_docs_allowed_helpers() -> None:
    assert not sql_allowed_for_intent(SourceIntent.DOCS_ONLY, sql_active=True)
    assert sql_allowed_for_intent(SourceIntent.AMBIGUOUS, sql_active=True)
    assert not docs_allowed_for_intent(SourceIntent.DB_ONLY)


def test_needs_sql_supplement_blocked_when_sql_not_allowed() -> None:
    query = "tell me the revenue from 2024 and 2025 from the doc"
    assert not _needs_sql_supplement(
        sql_active=True,
        sql_context=SqlToolContext(
            connection_url="postgresql://localhost/huawei",
            description="Huawei report DB",
        ),
        user_query=query,
        router_query=query,
        tools_used=["search_documents"],
        sql_allowed=False,
    )


def test_should_stop_when_sql_not_allowed_after_docs() -> None:
    assert should_stop_after_tool_round(
        tools_used=["search_documents"],
        sql_active=True,
        indexed_doc_count=4,
        user_query="x",
        router_query="x",
        sql_allowed=False,
        docs_allowed=True,
    )


def test_build_agent_tools_omits_sql_when_include_sql_false() -> None:
    tools = build_agent_tools(
        sql_active=True,
        sql_display_name="Huawei DB",
        include_sql=False,
    )
    names = {(t.get("function") or {}).get("name") for t in tools}
    assert "query_database" not in names
    assert "search_documents" in names


@pytest.mark.asyncio
async def test_from_the_doc_does_not_force_sql(monkeypatch: pytest.MonkeyPatch) -> None:
    query = "tell me the revenue from 2024 and 2025 from the doc"
    force_calls = {"n": 0}
    _stub_agent_scope(monkeypatch, indexed=4)

    search_round = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "search_documents",
                                "arguments": json.dumps({"query": query, "top_k": 5}),
                            },
                        }
                    ],
                }
            }
        ]
    }

    call_idx = {"n": 0}

    async def fake_completion(*, messages, tools=None, model=None, temperature=0.1, **kwargs):
        idx = call_idx["n"]
        call_idx["n"] += 1
        if idx == 0:
            names = {(t.get("function") or {}).get("name") for t in (tools or [])}
            assert "query_database" not in names
            return search_round
        return {"choices": [{"message": {"content": "", "tool_calls": []}}]}

    def fake_search(*args, **kwargs):
        chunk = RetrievedChunk(
            chunk_id="c1",
            doc_id="d1",
            filename="huawei.pdf",
            page_number=9,
            chunk_type="table",
            content="Revenue 2024 2025",
            score=1.0,
        )
        return json.dumps({"chunks": [{"chunk_id": "c1"}]}), [chunk]

    async def fake_force(**kwargs):
        force_calls["n"] += 1
        if False:  # pragma: no cover
            yield {}
        return
        yield

    monkeypatch.setattr("app.llm.agent.groq_chat_completion", fake_completion)
    monkeypatch.setattr("app.llm.agent.execute_search_documents", fake_search)
    monkeypatch.setattr("app.llm.agent._force_query_database", fake_force)
    monkeypatch.setattr("app.llm.agent.settings.agent_max_rounds", 2)

    async def resolve_docs(user_query: str, router_query: str | None = None):
        return SourceIntentResult(
            intent=SourceIntent.DOCS_ONLY,
            method=SourceIntentMethod.REGEX,
            reason="from_the_doc",
        )

    monkeypatch.setattr("app.llm.agent.resolve_source_intent", resolve_docs)

    tools = list(AGENT_TOOLS) + [
        build_query_database_tool(display_name="Huawei DB", description="report")
    ]
    result = None
    async for event in iter_agent_turn(
        object(),
        user_id=1,
        user_query=query,
        tools=tools,
        sql_active=True,
        sql_display_name="Huawei DB",
        sql_description="report",
        sql_context=SqlToolContext(
            connection_url="postgresql://localhost/huawei",
            description="report",
        ),
    ):
        if event["type"] == "complete":
            result = event["result"]

    assert result is not None
    assert "search_documents" in result.tools_used
    assert "query_database" not in result.tools_used
    assert force_calls["n"] == 0


@pytest.mark.asyncio
async def test_execute_refuses_query_database_when_docs_only(monkeypatch: pytest.MonkeyPatch) -> None:
    query = "from the doc only: revenue"
    _stub_agent_scope(monkeypatch, indexed=1)

    sql_round = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "query_database",
                                "arguments": json.dumps({"query": "revenue"}),
                            },
                        }
                    ],
                }
            }
        ]
    }

    async def fake_completion(*, messages, tools=None, model=None, temperature=0.1, **kwargs):
        if fake_completion.n == 0:
            fake_completion.n += 1
            return sql_round
        return {"choices": [{"message": {"content": "", "tool_calls": []}}]}

    fake_completion.n = 0
    run_calls = {"n": 0}

    async def fake_run(**kwargs):
        run_calls["n"] += 1
        if False:  # pragma: no cover
            yield {}
        return
        yield

    monkeypatch.setattr("app.llm.agent.groq_chat_completion", fake_completion)
    monkeypatch.setattr("app.llm.agent._run_query_database", fake_run)
    monkeypatch.setattr("app.llm.agent.settings.agent_max_rounds", 2)

    async def resolve_docs(user_query: str, router_query: str | None = None):
        return SourceIntentResult(
            intent=SourceIntent.DOCS_ONLY,
            method=SourceIntentMethod.REGEX,
            reason="test",
        )

    monkeypatch.setattr("app.llm.agent.resolve_source_intent", resolve_docs)
    monkeypatch.setattr(
        "app.llm.agent._filter_tools_for_intent",
        lambda tools, **kwargs: tools,
    )
    monkeypatch.setattr(
        "app.llm.agent._needs_doc_search_supplement",
        lambda **kwargs: False,
    )

    tools = list(AGENT_TOOLS) + [build_query_database_tool(display_name="Huawei DB")]
    result = None
    async for event in iter_agent_turn(
        object(),
        user_id=1,
        user_query=query,
        tools=tools,
        sql_active=True,
        sql_display_name="Huawei DB",
        sql_context=SqlToolContext(
            connection_url="postgresql://localhost/huawei",
            description="report",
        ),
    ):
        if event["type"] == "complete":
            result = event["result"]

    assert result is not None
    assert run_calls["n"] == 0
    assert "query_database" in result.tools_used
