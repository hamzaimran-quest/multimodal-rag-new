"""Tests for the visual-intent classifier (parsing + fail-closed behavior)."""

from __future__ import annotations

import pytest

from app.llm import intent as intent_module


@pytest.mark.asyncio
async def test_intent_disabled_returns_none(monkeypatch) -> None:
    monkeypatch.setattr("app.llm.intent.settings.image_intent_enabled", False)
    result = await intent_module.classify_visual_intent("show me the chart")
    assert result == {"visual_intent": "none", "confidence": 0.0}


@pytest.mark.asyncio
async def test_intent_missing_api_key_returns_none(monkeypatch) -> None:
    monkeypatch.setattr("app.llm.intent.settings.image_intent_enabled", True)
    monkeypatch.setattr("app.llm.intent.settings.groq_api_key", "your_groq_api_key_here")
    result = await intent_module.classify_visual_intent("show me the chart")
    assert result["visual_intent"] == "none"


@pytest.mark.asyncio
async def test_intent_parses_and_clamps(monkeypatch) -> None:
    monkeypatch.setattr("app.llm.intent.settings.image_intent_enabled", True)
    monkeypatch.setattr("app.llm.intent.settings.groq_api_key", "gsk_test")

    class _Resp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {
                "choices": [
                    {"message": {"content": '{"visual_intent": "required", "confidence": 1.7}'}}
                ]
            }

    class _Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def post(self, *args, **kwargs):
            return _Resp()

    monkeypatch.setattr("app.llm.intent.httpx.AsyncClient", _Client)
    result = await intent_module.classify_visual_intent("what does the chairman look like")
    assert result["visual_intent"] == "required"
    assert result["confidence"] == 1.0


@pytest.mark.asyncio
async def test_intent_invalid_enum_defaults_none(monkeypatch) -> None:
    monkeypatch.setattr("app.llm.intent.settings.image_intent_enabled", True)
    monkeypatch.setattr("app.llm.intent.settings.groq_api_key", "gsk_test")

    class _Resp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"choices": [{"message": {"content": '{"visual_intent": "banana"}'}}]}

    class _Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def post(self, *args, **kwargs):
            return _Resp()

    monkeypatch.setattr("app.llm.intent.httpx.AsyncClient", _Client)
    result = await intent_module.classify_visual_intent("who is the chairman")
    assert result["visual_intent"] == "none"
