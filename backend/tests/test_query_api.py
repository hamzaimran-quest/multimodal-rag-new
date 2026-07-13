"""Phase 5 query streaming API tests (agent path)."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_agent_query_stream_sends_tokens_and_sources(api_client_with_opensearch, monkeypatch):
    from app.llm.agent import AgentTurnResult
    from app.retrieval.models import RetrievedChunk

    async def fake_stream_groq_answer(*, query: str, context: str, visual_note=None, chart_note=None, last_assistant_reply=None, model=None):
        assert query == "financial highlights"
        assert "Document: huawei.pdf" in context
        for token in ["Revenue ", "grew ", "year-over-year."]:
            yield token

    async def fake_iter_agent_turn(*args, **kwargs):
        yield {
            "type": "complete",
            "result": AgentTurnResult(
                retrieved_chunks=[
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
                tools_used=["search_documents"],
            ),
        }

    monkeypatch.setattr("app.api.query.stream_groq_answer", fake_stream_groq_answer)
    monkeypatch.setattr("app.api.query.iter_agent_turn", fake_iter_agent_turn)

    async with api_client_with_opensearch.stream(
        "POST",
        "/query/stream",
        json={"query": "financial highlights", "top_k": 8},
    ) as response:
        assert response.status_code == 200
        body = await response.aread()

    text = body.decode("utf-8")
    assert "event: meta" in text
    assert "\"agent\": true" in text
    assert "\"session_id\"" in text
    assert "event: token" in text
    assert "Revenue " in text
    assert "event: sources" in text
    assert "\"chunk_type\": \"image\"" in text
    assert "\"image_url\": \"/images/d1/page12_img0.png\"" in text
    assert "event: done" in text


@pytest.mark.asyncio
async def test_agent_query_stream_emits_error_event(api_client_with_opensearch, monkeypatch):
    from app.llm.agent import AgentTurnResult

    async def fake_stream_fail(*args, **kwargs):
        raise RuntimeError("groq unavailable")
        yield  # pragma: no cover

    async def fake_iter_agent_turn(*args, **kwargs):
        yield {
            "type": "complete",
            "result": AgentTurnResult(
                retrieved_chunks=[],
                tools_used=[],
            ),
        }

    monkeypatch.setattr("app.api.query.stream_groq_answer", fake_stream_fail)
    monkeypatch.setattr("app.api.query.iter_agent_turn", fake_iter_agent_turn)

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
async def test_query_stream_emits_charts_for_chart_request(api_client_with_opensearch, monkeypatch):
    from app.llm.agent import AgentTurnResult
    from app.retrieval.models import RetrievedChunk

    async def fake_stream_groq_answer(*, query: str, context: str, visual_note=None, chart_note=None, last_assistant_reply=None, model=None):
        yield "Answer."

    table_content = (
        "| Metric | 2022 | 2023 | 2024 |\n"
        "| --- | --- | --- | --- |\n"
        "| Series A | 100 | 110 | 120 |\n"
        "| Series B | 10 | 12 | 15 |"
    )
    chunk = RetrievedChunk(
        chunk_id="t1",
        doc_id="d1",
        filename="report.pdf",
        page_number=3,
        chunk_type="table",
        content=table_content,
        score=0.9,
    )

    async def fake_iter_agent_turn(*args, **kwargs):
        yield {
            "type": "complete",
            "result": AgentTurnResult(
                retrieved_chunks=[chunk],
                tools_used=["search_documents"],
            ),
        }

    monkeypatch.setattr("app.api.query.stream_groq_answer", fake_stream_groq_answer)
    monkeypatch.setattr("app.api.query.iter_agent_turn", fake_iter_agent_turn)
    llm_spec = {
        "chart_type": "bar",
        "title": "Series A",
        "labels": ["2022", "2023", "2024"],
        "series": [{"name": "Series A", "values": [100.0, 110.0, 120.0]}],
    }
    monkeypatch.setattr(
        "app.charts.build.extract_chart_data_spec",
        lambda user_query, markdown, chart_type=None: (llm_spec, None),
    )
    monkeypatch.setattr(
        "app.charts.build.build_quickchart_url",
        lambda config: "https://quickchart.io/chart?c=auto",
    )

    async with api_client_with_opensearch.stream(
        "POST",
        "/query/stream",
        json={"query": "plot a chart of the finances"},
    ) as response:
        assert response.status_code == 200
        body = await response.aread()

    text = body.decode("utf-8")
    assert "event: charts" in text
    assert "\"derivation\": \"tool\"" in text
    assert "\"chart_type\": \"bar\"" in text


@pytest.mark.asyncio
async def test_agent_query_stream_persists_messages_to_session(api_client_with_opensearch, monkeypatch):
    from app.llm.agent import AgentTurnResult

    async def fake_stream_groq_answer(*, query: str, context: str, visual_note=None, chart_note=None, last_assistant_reply=None, model=None):
        yield "Answer text."

    async def fake_iter_agent_turn(*args, **kwargs):
        yield {
            "type": "complete",
            "result": AgentTurnResult(
                retrieved_chunks=[],
                tools_used=[],
            ),
        }

    monkeypatch.setattr("app.api.query.stream_groq_answer", fake_stream_groq_answer)
    monkeypatch.setattr("app.api.query.iter_agent_turn", fake_iter_agent_turn)

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


def test_build_chart_note_tells_model_not_to_emit_plotting_code() -> None:
    from app.api.query import _build_chart_note

    note = _build_chart_note(
        [
            {
                "chart_type": "line",
                "title": "CNY Million",
                "chunk_id": "t1",
            }
        ]
    )
    assert note is not None
    assert "Charts panel" in note
    assert "matplotlib" in note.lower()
    assert _build_chart_note([]) is None


def test_build_chart_failed_note_tells_model_not_to_emit_plotting_code() -> None:
    from app.api.query import _build_chart_failed_note

    note = _build_chart_failed_note()
    assert "Charts panel" in note
    assert "matplotlib" in note.lower()
    assert "Markdown table" in note

    detailed = _build_chart_failed_note("Series length mismatch")
    assert "Series length mismatch" in detailed


def test_build_user_prompt_appends_ui_notes() -> None:
    from app.llm.groq import build_user_prompt

    prompt = build_user_prompt(
        "show the chart",
        "context",
        visual_note="Image is shown in UI.",
        chart_note="Chart creation was attempted but no chart was rendered in the Charts panel.",
    )
    assert "UI note:" in prompt
    assert "Image is shown in UI." in prompt
    assert "Chart creation was attempted but no chart was rendered in the Charts panel." in prompt


def test_build_user_prompt_hybrid_includes_sql_as_first_class_source() -> None:
    from app.llm.groq import HYBRID_SYSTEM_PROMPT, build_user_prompt, resolve_answer_system_prompt

    prompt = build_user_prompt(
        "revenue by segment and chairwoman quote",
        "--- Source 1 ---\nDocument: huawei.pdf\nContent:\nChairwoman message",
        sql_context="Segment A: 100 CNY million",
        hybrid=True,
    )
    assert "Database query results (authoritative for live data facts):" in prompt
    assert "<untrusted_database_result>" in prompt
    assert "Segment A: 100 CNY million" in prompt
    assert "Document excerpts (authoritative for uploaded file content):" in prompt
    assert "Chairwoman message" in prompt
    assert "Synthesize database results and document excerpts" in prompt
    assert "UI note:" not in prompt
    assert resolve_answer_system_prompt(hybrid=True) == HYBRID_SYSTEM_PROMPT
    assert "Not found in the provided sources" in resolve_answer_system_prompt(hybrid=True)


def test_build_user_prompt_sql_only_uses_database_results() -> None:
    from app.llm.groq import SQL_ONLY_SYSTEM_PROMPT, build_user_prompt, resolve_answer_system_prompt

    prompt = build_user_prompt(
        "compare revenue 2024 and 2025",
        "",
        sql_context="Total revenue 2024: 862bn, 2025: 881bn",
        sql_only=True,
    )
    assert "Database query results (authoritative for live data facts):" in prompt
    assert "Total revenue 2024: 862bn, 2025: 881bn" in prompt
    assert "Document excerpts" not in prompt
    assert "Format the database query results" in prompt
    assert resolve_answer_system_prompt(sql_only=True) == SQL_ONLY_SYSTEM_PROMPT
    assert "Not found in the provided database results" in SQL_ONLY_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_agent_query_stream_emits_tool_events(api_client_with_opensearch, monkeypatch):
    from app.llm.agent import AgentTurnResult

    async def fake_iter_agent_turn(*args, **kwargs):
        yield {"type": "tool", "name": "search_documents", "status": "running", "round": 1}
        yield {
            "type": "complete",
            "result": AgentTurnResult(
                direct_answer="Hi there!",
                tools_used=["search_documents"],
            ),
        }

    monkeypatch.setattr("app.api.query.iter_agent_turn", fake_iter_agent_turn)

    async with api_client_with_opensearch.stream(
        "POST",
        "/query/stream",
        json={"query": "hello"},
    ) as response:
        assert response.status_code == 200
        body = await response.aread()

    text = body.decode("utf-8")
    assert "\"agent\": true" in text
    assert "event: tool" in text
    assert "search_documents" in text
    assert "Hi there!" in text
    assert "event: done" in text
