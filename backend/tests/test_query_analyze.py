"""Tests for history-aware query rewriting."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.llm import query_analyze


def test_format_history_skips_empty_turns():
    history = [
        {"role": "user", "content": "compare finances of huawei"},
        {"role": "assistant", "content": "Revenue in 2025 was CNY880,941 million."},
        {"role": "user", "content": ""},
    ]
    formatted = query_analyze._format_history(history)
    assert "compare finances of huawei" in formatted
    assert "880,941" in formatted
    assert formatted.count("user:") == 1


def test_parse_analysis_response_fallback_on_bad_intent():
    result = query_analyze._parse_analysis_response(
        {"standalone_query": "Huawei revenue 2024 vs 2025", "visual_intent": "bogus"},
        original_query="compare these with 2024",
    )
    assert result["standalone_query"] == "Huawei revenue 2024 vs 2025"
    assert result["visual_intent"] == "none"


@pytest.mark.asyncio
async def test_analyze_query_disabled_returns_original(monkeypatch):
    monkeypatch.setattr(query_analyze.settings, "query_rewrite_enabled", False)
    result = await query_analyze.analyze_query([], "compare these with 2024")
    assert result["standalone_query"] == "compare these with 2024"
    assert result["visual_intent"] == "none"


@pytest.mark.asyncio
async def test_analyze_query_rewrites_finance_follow_up(monkeypatch):
    monkeypatch.setattr(query_analyze.settings, "query_rewrite_enabled", True)
    monkeypatch.setattr(query_analyze.settings, "groq_api_key", "test-key")
    monkeypatch.setattr(query_analyze.settings, "query_rewrite_model", "llama-3.1-8b-instant")
    monkeypatch.setattr(query_analyze.settings, "chat_history_turns", 6)

    history = [
        {"role": "user", "content": "compare finances of huawei"},
        {
            "role": "assistant",
            "content": "Huawei's annual revenue in 2025 was CNY880,941 million.",
        },
    ]
    groq_body = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "standalone_query": (
                                "compare Huawei annual revenue 2024 vs 2025 financial highlights"
                            ),
                            "visual_intent": "none",
                        }
                    )
                }
            }
        ]
    }

    mock_response = AsyncMock()
    mock_response.raise_for_status = lambda: None
    mock_response.json = lambda: groq_body

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("app.llm.query_analyze.httpx.AsyncClient", return_value=mock_client):
        result = await query_analyze.analyze_query(history, "compare these with 2024")

    assert "2024" in result["standalone_query"]
    assert "revenue" in result["standalone_query"].lower()
    assert result["visual_intent"] == "none"
    payload = mock_client.post.call_args.kwargs["json"]
    user_content = payload["messages"][1]["content"]
    assert "compare finances of huawei" in user_content
    assert "compare these with 2024" in user_content


@pytest.mark.asyncio
async def test_analyze_query_rewrites_pronoun_image_follow_up(monkeypatch):
    monkeypatch.setattr(query_analyze.settings, "query_rewrite_enabled", True)
    monkeypatch.setattr(query_analyze.settings, "groq_api_key", "test-key")

    history = [
        {"role": "user", "content": "who is the chairman of huawei"},
        {"role": "assistant", "content": "The Chairman of the Board of Huawei is Liang Hua."},
    ]
    groq_body = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "standalone_query": "image of Huawei chairman Liang Hua",
                            "visual_intent": "required",
                        }
                    )
                }
            }
        ]
    }

    mock_response = AsyncMock()
    mock_response.raise_for_status = lambda: None
    mock_response.json = lambda: groq_body

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("app.llm.query_analyze.httpx.AsyncClient", return_value=mock_client):
        result = await query_analyze.analyze_query(history, "show an image of him")

    assert "Liang Hua" in result["standalone_query"]
    assert result["visual_intent"] == "required"
