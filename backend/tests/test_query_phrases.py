"""Tests for lookup phrase extraction."""

from __future__ import annotations

from app.retrieval.query_phrases import extract_lookup_phrases


def test_extract_lookup_phrases_preserves_digit_word_titles() -> None:
    phrases = extract_lookup_phrases("tell me about ben 10")
    folded = [phrase.casefold() for phrase in phrases]
    assert "ben 10" in folded


def test_extract_lookup_phrases_preserves_leading_digit_titles() -> None:
    phrases = extract_lookup_phrases("when was 6 years released")
    folded = [phrase.casefold() for phrase in phrases]
    assert "6 years" in folded


def test_extract_lookup_phrases_keeps_multi_word_names() -> None:
    phrases = extract_lookup_phrases("country of Transformers Prime")
    folded = [phrase.casefold() for phrase in phrases]
    assert "transformers prime" in folded
