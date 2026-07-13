"""Tests for standalone query rewriting."""

from __future__ import annotations

import pytest

from app.llm.query_rewrite import (
    _rewrite_over_expanded,
    _trim_over_expanded_rewrite,
    rewrite_query_for_retrieval,
)


@pytest.mark.asyncio
async def test_rewrite_skips_without_prior_context(monkeypatch) -> None:
    monkeypatch.setattr("app.llm.query_rewrite.settings.query_rewrite_enabled", True)

    async def fail_post(*args, **kwargs):
        raise AssertionError("should not call Groq when there is no prior context")

    monkeypatch.setattr("httpx.AsyncClient.post", fail_post)

    result = await rewrite_query_for_retrieval("who is the chairman?", [])
    assert result == "who is the chairman?"


@pytest.mark.asyncio
async def test_rewrite_uses_prior_queries(monkeypatch) -> None:
    monkeypatch.setattr("app.llm.query_rewrite.settings.query_rewrite_enabled", True)
    monkeypatch.setattr("app.llm.query_rewrite.settings.groq_api_key", "test-key")

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {"message": {"content": "Ren Zhengfei portrait photo"}}
                ]
            }

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: FakeClient())

    rewritten = await rewrite_query_for_retrieval(
        "show an image of him",
        ["who founded huawei"],
    )
    assert rewritten == "Ren Zhengfei portrait photo"


@pytest.mark.asyncio
async def test_rewrite_restores_colon_title_stripped_by_model(monkeypatch) -> None:
    monkeypatch.setattr("app.llm.query_rewrite.settings.query_rewrite_enabled", True)
    monkeypatch.setattr("app.llm.query_rewrite.settings.groq_api_key", "test-key")

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Whatever it Takes category country cast title"
                        }
                    }
                ]
            }

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: FakeClient())

    rewritten = await rewrite_query_for_retrieval(
        "tell me about Jandino: Whatever it Takes, its category, its country and a cast title",
        ["tell me about Jandino: Whatever it Takes, its category, its country and a cast title"],
    )
    assert "Jandino: Whatever it Takes".casefold() in rewritten.casefold()


@pytest.mark.asyncio
async def test_rewrite_uses_last_assistant_reply_only(monkeypatch) -> None:
    monkeypatch.setattr("app.llm.query_rewrite.settings.query_rewrite_enabled", True)
    monkeypatch.setattr("app.llm.query_rewrite.settings.groq_api_key", "test-key")

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "Meng Wanzhou rotating chairwoman "
                                "Message from the Rotating Chairwoman portrait photo"
                            )
                        }
                    }
                ]
            }

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: FakeClient())

    rewritten = await rewrite_query_for_retrieval(
        "show her image",
        ["who is the chairwoman"],
        last_assistant_reply="Meng Wanzhou is the rotating chairwoman.",
    )
    assert "Meng Wanzhou" in rewritten
    assert "rotating chairwoman" in rewritten.casefold()
    assert "portrait" in rewritten.casefold()


@pytest.mark.asyncio
async def test_rewrite_disabled_returns_original(monkeypatch) -> None:
    monkeypatch.setattr("app.llm.query_rewrite.settings.query_rewrite_enabled", False)

    result = await rewrite_query_for_retrieval(
        "show an image of him",
        ["who founded huawei"],
    )
    assert result == "show an image of him"


def test_rewrite_over_expanded_detects_appended_entity_lists() -> None:
    original = "compare that"
    bloated = (
        "compare Huawei revenue 2024 2025 Consumer Business Enterprise Business "
        "Carrier Business Cloud Computing Digital Power Smart Vehicle"
    )
    assert _rewrite_over_expanded(original, bloated) is True


def test_rewrite_over_expanded_allows_focused_resolution() -> None:
    original = "show her image"
    focused = "Meng Wanzhou rotating chairwoman portrait photo"
    assert _rewrite_over_expanded(original, focused) is False


def test_trim_over_expanded_rewrite_trims_segment_lists() -> None:
    original = "compare that"
    bloated = (
        "compare Huawei revenue 2024 2025 Consumer Business Enterprise Business "
        "Carrier Business Cloud Computing Digital Power Smart Vehicle"
    )
    assert _trim_over_expanded_rewrite(original, bloated) == "compare Huawei revenue 2024 2025"


@pytest.mark.asyncio
async def test_rewrite_rejects_over_expanded_model_output(monkeypatch) -> None:
    monkeypatch.setattr("app.llm.query_rewrite.settings.query_rewrite_enabled", True)
    monkeypatch.setattr("app.llm.query_rewrite.settings.groq_api_key", "test-key")

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "compare Huawei revenue 2024 2025 Consumer Business "
                                "Enterprise Business Carrier Business Cloud Computing "
                                "Digital Power Smart Vehicle"
                            )
                        }
                    }
                ]
            }

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: FakeClient())

    rewritten = await rewrite_query_for_retrieval(
        "compare that",
        ["what is Huawei segment revenue by region"],
        last_assistant_reply=(
            "Consumer Business, Enterprise Business, Carrier Business, "
            "Cloud Computing, Digital Power, and Smart Vehicle segments..."
        ),
    )
    assert rewritten == "compare Huawei revenue 2024 2025"
