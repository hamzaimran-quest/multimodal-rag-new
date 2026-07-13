"""Tests for SQL agent LLM provider selection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.config import settings
from app.sql_agent import llm as sql_llm


def test_sql_agent_prefers_openrouter_when_key_set(monkeypatch) -> None:
    monkeypatch.setattr(settings, "sql_agent_openrouter_api_key", "or-test-key")
    monkeypatch.setattr(settings, "sql_agent_model", "openai/gpt-oss-120b")
    monkeypatch.setattr(settings, "groq_api_key", "groq-test-key")

    chat_openai = MagicMock(name="ChatOpenAI")
    with patch.dict("sys.modules", {"langchain_openai": MagicMock(ChatOpenAI=chat_openai)}):
        sql_llm.build_sql_agent_llm()

    chat_openai.assert_called_once_with(
        model="openai/gpt-oss-120b",
        api_key="or-test-key",
        base_url=settings.sql_agent_openrouter_base_url,
        temperature=0,
        streaming=True,
    )


def test_sql_agent_falls_back_to_groq_without_openrouter(monkeypatch) -> None:
    monkeypatch.setattr(settings, "sql_agent_openrouter_api_key", None)
    monkeypatch.setattr(settings, "sql_agent_model", "openai/gpt-oss-20b")
    monkeypatch.setattr(settings, "groq_api_key", "groq-test-key")

    chat_groq = MagicMock(name="ChatGroq")
    with patch.dict("sys.modules", {"langchain_groq": MagicMock(ChatGroq=chat_groq)}):
        sql_llm.build_sql_agent_llm()

    chat_groq.assert_called_once_with(
        model="openai/gpt-oss-20b",
        groq_api_key="groq-test-key",
        temperature=0,
        streaming=True,
    )


def test_sql_agent_llm_requires_provider(monkeypatch) -> None:
    monkeypatch.setattr(settings, "sql_agent_openrouter_api_key", None)
    monkeypatch.setattr(settings, "groq_api_key", None)

    with pytest.raises(RuntimeError, match="SQL agent LLM is not configured"):
        sql_llm.build_sql_agent_llm()
