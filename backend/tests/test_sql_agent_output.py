"""Tests for SQL agent answer cleanup and query extraction."""

from __future__ import annotations

from types import SimpleNamespace

from app.sql_agent.agent import (
    _extract_queries_from_steps,
    clean_sql_answer_text,
    message_requests_tools,
)


def _step(tool: str, query: str) -> tuple[SimpleNamespace, str]:
    return SimpleNamespace(tool=tool, tool_input={"query": query}), "[]"


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
    from types import SimpleNamespace

    assert message_requests_tools(SimpleNamespace(tool_calls=[{"name": "sql_db_query"}]))
    assert not message_requests_tools(SimpleNamespace(tool_calls=[], content="Done."))
