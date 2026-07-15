"""Tests for SQL agent answer cleanup and query extraction."""

from __future__ import annotations

from types import SimpleNamespace

from app.sql_agent.agent import (
    _extract_queries_from_steps,
    clean_sql_answer_text,
    finalize_sql_agent_answer,
    message_requests_tools,
)
from app.sql_agent.prompts import build_sql_agent_prefix


def _step(tool: str, query: str, observation: str = "[]") -> tuple[SimpleNamespace, str]:
    return SimpleNamespace(tool=tool, tool_input={"query": query}), observation


def test_extract_queries_ignores_checker_tool() -> None:
    query = "SELECT title FROM film WHERE title = 'Ace Goldfinger';"
    steps = [
        _step("sql_db_query_checker", query),
        _step("sql_db_query", query),
    ]
    assert _extract_queries_from_steps(steps) == [query]


def test_extract_queries_dedupes_repeated_execute_steps() -> None:
    query = "SELECT title FROM film;"
    steps = [
        _step("sql_db_query", query),
        _step("sql_db_query", query),
    ]
    assert _extract_queries_from_steps(steps) == [query]


def test_clean_sql_answer_text_strips_leading_select() -> None:
    raw = (
        "SELECT film_id, title FROM film WHERE title = 'Ace Goldfinger';"
        "Film: Ace Goldfinger\n\nDescription: A sample film."
    )
    cleaned = clean_sql_answer_text(raw)
    assert cleaned.startswith("Film: Ace Goldfinger")
    assert "SELECT" not in cleaned


def test_message_requests_tools_detects_tool_calls() -> None:
    assert message_requests_tools(SimpleNamespace(tool_calls=[{"name": "sql_db_query"}]))
    assert not message_requests_tools(SimpleNamespace(tool_calls=[], content="Done."))


def test_finalize_discards_prose_without_execute() -> None:
    answer, queries = finalize_sql_agent_answer(
        prose_answer="Revenue was 881 billion based on knowledge.",
        steps=[],
        tool_calls_seen=["sql_db_query_checker"],
    )
    assert queries == []
    assert "Not found" in answer
    assert "881" not in answer


def test_finalize_keeps_prose_after_execute() -> None:
    query = "SELECT sum(revenue) FROM financial_highlights;"
    answer, queries = finalize_sql_agent_answer(
        prose_answer="Total revenue is 881.",
        steps=[_step("sql_db_query", query, "[(881,)]")],
    )
    assert queries == [query]
    assert answer == "Total revenue is 881."


def test_finalize_falls_back_to_observations_when_prose_empty() -> None:
    query = "SELECT segment, growth FROM business_segments;"
    answer, queries = finalize_sql_agent_answer(
        prose_answer="SELECT * FROM business_segments;",
        steps=[_step("sql_db_query", query, "[('ICT', 12.5)]")],
    )
    assert queries == [query]
    assert "ICT" in answer


def test_sql_agent_prefix_is_markdown_and_requires_execute() -> None:
    prefix = build_sql_agent_prefix("Huawei financials", schema_digest="tables: t1")
    assert prefix.startswith("# PostgreSQL")
    assert "## Required workflow" in prefix
    assert "`sql_db_query`" in prefix
    assert "CREATE" in prefix
    assert "Cached schema" in prefix


def test_sql_agent_prefix_omits_checker_by_default(monkeypatch) -> None:
    monkeypatch.setattr("app.sql_agent.prompts.settings.sql_agent_query_checker_enabled", False)
    prefix = build_sql_agent_prefix("Huawei financials", schema_digest="tables: t1")
    assert "sql_db_query_checker" not in prefix
    assert "checker optional" not in prefix


def test_sql_agent_prefix_includes_checker_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr("app.sql_agent.prompts.settings.sql_agent_query_checker_enabled", True)
    prefix = build_sql_agent_prefix("Huawei financials", schema_digest="tables: t1")
    assert "sql_db_query_checker" in prefix
    assert "checker optional" in prefix


def test_sql_agent_tool_names_excludes_checker_by_default(monkeypatch) -> None:
    from app.sql_agent.agent import sql_agent_tool_names

    monkeypatch.setattr("app.sql_agent.agent.settings.sql_agent_query_checker_enabled", False)
    assert sql_agent_tool_names() == {"sql_db_query"}


def test_sql_agent_tool_names_includes_checker_when_enabled(monkeypatch) -> None:
    from app.sql_agent.agent import sql_agent_tool_names

    monkeypatch.setattr("app.sql_agent.agent.settings.sql_agent_query_checker_enabled", True)
    assert sql_agent_tool_names() == {"sql_db_query", "sql_db_query_checker"}
