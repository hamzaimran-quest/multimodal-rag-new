"""Router helpers for SQL agent integration."""

from __future__ import annotations

from app.llm.agent import AgentTurnResult, build_agent_tools, resolve_route_mode
from app.retrieval.models import RetrievedChunk


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        doc_id="d1",
        filename="report.pdf",
        page_number=1,
        chunk_type="text",
        content="hello",
        score=0.9,
    )


def test_build_agent_tools_includes_query_database_when_active() -> None:
    names = {
        tool["function"]["name"]
        for tool in build_agent_tools(sql_active=True, sql_display_name="DVD Rental")
    }
    assert "query_database" in names
    assert "search_documents" in names


def test_build_agent_tools_omits_query_database_when_inactive() -> None:
    names = {tool["function"]["name"] for tool in build_agent_tools(sql_active=False)}
    assert "query_database" not in names


def test_resolve_route_mode_sql_only() -> None:
    turn = AgentTurnResult(
        tools_used=["query_database"],
        sql_query="count users",
        sql_result_text="42 users",
    )
    assert resolve_route_mode(turn) == "sql"


def test_resolve_route_mode_hybrid() -> None:
    turn = AgentTurnResult(
        tools_used=["query_database", "search_documents"],
        retrieved_chunks=[_chunk()],
        sql_query="headcount",
    )
    assert resolve_route_mode(turn) == "hybrid"


def test_resolve_route_mode_rag_default() -> None:
    turn = AgentTurnResult(tools_used=["search_documents"], retrieved_chunks=[_chunk()])
    assert resolve_route_mode(turn) == "rag"
