"""Agent tool payload tests."""

from __future__ import annotations

from app.llm.tools import _chunk_for_tool, _clamp_agent_top_k
from app.retrieval.models import RetrievedChunk


def test_chunk_for_tool_uses_snippet_not_full_content(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "agent_tool_snippet_max_chars", 80)
    chunk = RetrievedChunk(
        chunk_id="c1",
        doc_id="d1",
        filename="report.pdf",
        page_number=2,
        chunk_type="text",
        content="A" * 500,
        score=0.9,
    )
    payload = _chunk_for_tool(chunk)
    assert "content" not in payload
    assert "snippet" in payload
    assert len(payload["snippet"]) <= 80
    assert payload["chunk_id"] == "c1"


def test_clamp_agent_top_k_respects_default(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "default_top_k", 5)
    assert _clamp_agent_top_k(10) == 5
    assert _clamp_agent_top_k(3) == 3
    assert _clamp_agent_top_k(None) is None
