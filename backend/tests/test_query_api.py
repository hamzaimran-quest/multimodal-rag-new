"""Phase 5 query streaming API tests."""

from __future__ import annotations

import pytest

from app.retrieval.models import RetrievedChunk, SearchResponse


@pytest.mark.asyncio
async def test_query_stream_sends_tokens_and_sources(api_client_with_opensearch, monkeypatch):
    async def fake_stream_groq_answer(*, query: str, context: str, model: str = "llama-3.3-70b-versatile"):
        assert query == "financial highlights"
        assert "Document: huawei.pdf" in context
        for token in ["Revenue ", "grew ", "year-over-year."]:
            yield token

    def fake_hybrid_retrieve(
        client, query: str, *, user_id: int, top_k: int | None = None, doc_id: str | None = None
    ):
        return SearchResponse(
            query=query,
            top_k=top_k or 8,
            doc_id=doc_id,
            total=2,
            results=[
                RetrievedChunk(
                    chunk_id="c1",
                    doc_id="d1",
                    filename="huawei.pdf",
                    page_number=12,
                    chunk_type="text",
                    content="Five-Year Financial Highlights Revenue CNY Million",
                    score=0.9,
                ),
                RetrievedChunk(
                    chunk_id="c2",
                    doc_id="d1",
                    filename="huawei.pdf",
                    page_number=12,
                    chunk_type="image",
                    content="Chart OCR keywords: revenue CNY million",
                    image_url="/images/d1/page12_img0.png",
                    score=0.7,
                ),
            ],
        )

    monkeypatch.setattr("app.api.query.stream_groq_answer", fake_stream_groq_answer)
    monkeypatch.setattr("app.api.query.hybrid_retrieve", fake_hybrid_retrieve)

    async with api_client_with_opensearch.stream(
        "POST",
        "/query/stream",
        json={"query": "financial highlights", "top_k": 8},
    ) as response:
        assert response.status_code == 200
        body = await response.aread()

    text = body.decode("utf-8")
    assert "event: meta" in text
    assert "\"session_id\"" in text
    assert "event: token" in text
    assert "Revenue " in text
    assert "event: sources" in text
    assert "\"chunk_type\": \"image\"" in text
    assert "\"image_url\": \"/images/d1/page12_img0.png\"" in text
    assert "event: done" in text


@pytest.mark.asyncio
async def test_query_stream_emits_error_event(api_client_with_opensearch, monkeypatch):
    async def fake_stream_fail(*args, **kwargs):
        raise RuntimeError("groq unavailable")
        yield  # pragma: no cover

    def fake_hybrid_retrieve(
        client, query: str, *, user_id: int, top_k: int | None = None, doc_id: str | None = None
    ):
        return SearchResponse(query=query, top_k=top_k or 8, doc_id=doc_id, total=0, results=[])

    monkeypatch.setattr("app.api.query.stream_groq_answer", fake_stream_fail)
    monkeypatch.setattr("app.api.query.hybrid_retrieve", fake_hybrid_retrieve)

    async with api_client_with_opensearch.stream(
        "POST",
        "/query/stream",
        json={"query": "financial highlights"},
    ) as response:
        assert response.status_code == 200
        body = await response.aread()

    text = body.decode("utf-8")
    assert "event: error" in text
    assert "groq unavailable" in text
    assert "\"ok\": false" in text


@pytest.mark.asyncio
async def test_query_stream_emits_charts_for_chartable_table(api_client_with_opensearch, monkeypatch):
    async def fake_stream_groq_answer(*, query: str, context: str, model: str = "llama-3.3-70b-versatile"):
        yield "Answer."

    table_content = (
        "| Metric | 2022 | 2023 | 2024 |\n"
        "| --- | --- | --- | --- |\n"
        "| Series A | 100 | 110 | 120 |\n"
        "| Series B | 10 | 12 | 15 |"
    )

    def fake_hybrid_retrieve(
        client, query: str, *, user_id: int, top_k: int | None = None, doc_id: str | None = None
    ):
        return SearchResponse(
            query=query,
            top_k=top_k or 8,
            doc_id=doc_id,
            total=1,
            results=[
                RetrievedChunk(
                    chunk_id="t1",
                    doc_id="d1",
                    filename="report.pdf",
                    page_number=3,
                    chunk_type="table",
                    content=table_content,
                    score=0.9,
                    extra_metadata={
                        "chart_profile": {
                            "chartable": True,
                            "orientation": "wide",
                            "period_count": 3,
                            "metric_count": 2,
                            "suggested_chart_type": "bar",
                        }
                    },
                ),
            ],
        )

    monkeypatch.setattr("app.api.query.stream_groq_answer", fake_stream_groq_answer)
    monkeypatch.setattr("app.api.query.hybrid_retrieve", fake_hybrid_retrieve)

    async with api_client_with_opensearch.stream(
        "POST",
        "/query/stream",
        json={"query": "anything"},
    ) as response:
        assert response.status_code == 200
        body = await response.aread()

    text = body.decode("utf-8")
    assert "event: charts" in text
    assert "\"derivation\": \"computed\"" in text
    assert "\"chart_type\": \"bar\"" in text


@pytest.mark.asyncio
async def test_query_stream_persists_messages_to_session(api_client_with_opensearch, monkeypatch):
    async def fake_stream_groq_answer(*, query: str, context: str, model: str = "llama-3.3-70b-versatile"):
        yield "Answer text."

    def fake_hybrid_retrieve(
        client, query: str, *, user_id: int, top_k: int | None = None, doc_id: str | None = None
    ):
        return SearchResponse(query=query, top_k=top_k or 8, doc_id=doc_id, total=0, results=[])

    monkeypatch.setattr("app.api.query.stream_groq_answer", fake_stream_groq_answer)
    monkeypatch.setattr("app.api.query.hybrid_retrieve", fake_hybrid_retrieve)

    async with api_client_with_opensearch.stream(
        "POST",
        "/query/stream",
        json={"query": "financial highlights"},
    ) as response:
        assert response.status_code == 200
        body = await response.aread()

    text = body.decode("utf-8")
    assert "event: done" in text
    assert "\"ok\": true" in text

    listed = await api_client_with_opensearch.get("/chats")
    sessions = listed.json()["sessions"]
    assert len(sessions) >= 1

    session_id = sessions[0]["id"]
    detail = await api_client_with_opensearch.get(f"/chats/{session_id}")
    messages = detail.json()["messages"]
    assert len(messages) == 2
    assert messages[0]["content"] == "financial highlights"
    assert messages[1]["content"] == "Answer text."


def test_build_visual_note_includes_caption() -> None:
    from app.api.query import _build_visual_note

    note = _build_visual_note([{"caption": "Rotating Chairwoman"}])
    assert note is not None
    assert "Rotating Chairwoman" in note
    assert "not found" in note.lower()
    assert _build_visual_note([]) is None


def test_build_user_prompt_appends_visual_note() -> None:
    from app.llm.groq import build_user_prompt

    prompt = build_user_prompt("show the chart", "context", visual_note="Image is shown in UI.")
    assert "UI note:" in prompt
    assert "Image is shown in UI." in prompt


@pytest.mark.asyncio
async def test_agent_query_stream_emits_tool_events(api_client_with_opensearch, monkeypatch):
    from app.llm.agent import AgentTurnResult

    async def fake_run_agent_turn(*args, **kwargs):
        return AgentTurnResult(
            direct_answer="Hi there!",
            tools_used=[],
        )

    async def fake_legacy(*args, **kwargs):
        raise AssertionError("legacy path should not run when agent is enabled")

    monkeypatch.setattr(settings, "agent_enabled", True)
    monkeypatch.setattr("app.api.query.run_agent_turn", fake_run_agent_turn)
    monkeypatch.setattr("app.api.query._legacy_event_stream", fake_legacy)

    async with api_client_with_opensearch.stream(
        "POST",
        "/query/stream",
        json={"query": "hello"},
    ) as response:
        assert response.status_code == 200
        body = await response.aread()

    text = body.decode("utf-8")
    assert "\"agent\": true" in text
    assert "Hi there!" in text
    assert "event: done" in text

