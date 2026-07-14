"""Tests for lightweight SQL tool fallback heuristics."""

from __future__ import annotations

from app.sql_agent.sql_fallback import looks_like_hybrid_question, looks_like_sql_query


def test_sql_fallback_detects_aggregate_queries() -> None:
    assert looks_like_sql_query("how many films are there?", ["film", "actor"]) == "sql"


def test_sql_fallback_detects_hybrid_compare() -> None:
    decision = looks_like_sql_query(
        "compare revenue in the database with the uploaded spreadsheet",
        ["orders"],
    )
    assert decision == "hybrid"


def test_sql_fallback_skips_document_questions() -> None:
    assert looks_like_sql_query("what does page 3 of the pdf say?", ["film"]) is None


def test_sql_fallback_without_active_tables() -> None:
    assert looks_like_sql_query("how many users?", []) == "sql"


def test_looks_like_hybrid_question_revenue_and_chairwoman() -> None:
    query = (
        "Show the 2025 revenue and growth rates for our core business segments "
        "and also tell me what the rotating chairwoman states"
    )
    assert looks_like_hybrid_question(query)
    assert looks_like_sql_query(query, ["business_segments"]) == "hybrid"


def test_looks_like_hybrid_question_rejects_pure_sql() -> None:
    assert not looks_like_hybrid_question("total revenue by segment for 2025")
    assert looks_like_sql_query("how many films are there?", ["film"]) == "sql"
