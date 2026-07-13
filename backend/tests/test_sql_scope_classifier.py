"""Tests for SQL scope classifier heuristics."""

from __future__ import annotations

import pytest

from app.sql_agent.scope_classifier import ScopeClassification, _heuristic_scope, classify_query_scope


def test_heuristic_film_query_routes_to_sql() -> None:
    result = _heuristic_scope("tell me about the film Ace Goldfinger", ["film", "actor"])
    assert result is not None
    assert result.decision == "sql"
    assert "film" in result.matched_tables


def test_heuristic_document_keywords_route_to_rag() -> None:
    result = _heuristic_scope("summarize the uploaded pdf report", ["film"])
    assert result is not None
    assert result.decision == "rag"


@pytest.mark.asyncio
async def test_classify_query_scope_uses_heuristic_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr("app.sql_agent.scope_classifier.settings.sql_scope_classifier_enabled", False)
    result = await classify_query_scope(
        query="how many films are there?",
        schema_digest="film(id, title)",
        tables=["film"],
        display_name="DVD",
        description="dvd rental",
    )
    assert isinstance(result, ScopeClassification)
    assert result.decision == "sql"
