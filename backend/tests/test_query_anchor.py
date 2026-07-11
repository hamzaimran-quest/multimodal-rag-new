"""Tests for retrieval query anchor phrase preservation."""

from __future__ import annotations

from app.retrieval.query_anchor import extract_named_phrases, merge_retrieval_anchor_phrases


def test_extract_named_phrases_finds_colon_title() -> None:
    phrases = extract_named_phrases(
        "tell me about Jandino: Whatever it Takes, its category and country"
    )
    assert any("Jandino: Whatever it Takes" in phrase for phrase in phrases)


def test_merge_retrieval_anchor_phrases_restores_missing_title() -> None:
    merged = merge_retrieval_anchor_phrases(
        "Whatever it Takes category country cast title",
        fallback_queries=[
            "tell me about Jandino: Whatever it Takes, its category, its country and a cast title"
        ],
    )
    assert "Jandino: Whatever it Takes".casefold() in merged.casefold()


def test_merge_retrieval_anchor_phrases_keeps_existing_title() -> None:
    original = "Jandino: Whatever it Takes category country"
    assert merge_retrieval_anchor_phrases(original, fallback_queries=[original]) == original
