"""Tests for standalone query rewriting."""

from __future__ import annotations

import pytest

from app.llm.query_rewrite import rewrite_query_for_retrieval


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
async def test_rewrite_uses_last_assistant_reply_only(monkeypatch) -> None:
    monkeypatch.setattr("app.llm.query_rewrite.settings.query_rewrite_enabled", True)
    monkeypatch.setattr("app.llm.query_rewrite.settings.groq_api_key", "test-key")

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {"message": {"content": "Meng Wanzhou portrait photo"}}
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
        [],
        last_assistant_reply="Meng Wanzhou is the rotating chairwoman.",
    )
    assert rewritten == "Meng Wanzhou portrait photo"


@pytest.mark.asyncio
async def test_rewrite_disabled_returns_original(monkeypatch) -> None:
    monkeypatch.setattr("app.llm.query_rewrite.settings.query_rewrite_enabled", False)

    result = await rewrite_query_for_retrieval(
        "show an image of him",
        ["who founded huawei"],
    )
    assert result == "show an image of him"
