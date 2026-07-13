"""Tests for LLM context building and trimming."""

from __future__ import annotations

from app.config import settings
from app.retrieval.context import build_llm_context, select_chunks_for_llm_context
from app.retrieval.models import RetrievedChunk


def _chunk(
    chunk_id: str,
    *,
    content: str,
    score: float = 0.5,
    chunk_type: str = "table",
    sheet_role: str | None = None,
) -> RetrievedChunk:
    extra: dict = {"source_format": "xlsx"}
    if sheet_role:
        extra["sheet_role"] = sheet_role
        extra["sheet_name"] = f"{sheet_role}_sheet"
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        filename="book.xlsx",
        page_number=1,
        chunk_type=chunk_type,
        content=content,
        score=score,
        extra_metadata=extra,
    )


def test_build_llm_context_formats_sources() -> None:
    context = build_llm_context([_chunk("a", content="alpha")])
    assert "--- Source 1 ---" in context
    assert "alpha" in context


def test_build_llm_context_docx_omits_page_number() -> None:
    chunk = RetrievedChunk(
        chunk_id="d1",
        doc_id="doc-1",
        filename="demo.docx",
        page_number=58,
        chunk_type="text",
        content="All list types are supported except fancy bullets.",
        score=0.9,
        extra_metadata={"source_format": "docx", "block_index": 58, "section": "Lists"},
    )
    context = build_llm_context([chunk])
    assert "Page: 58" not in context
    assert "Section: Lists" in context
    assert "fancy bullets" in context


def test_build_llm_context_pdf_keeps_page_number() -> None:
    chunk = RetrievedChunk(
        chunk_id="p1",
        doc_id="doc-1",
        filename="report.pdf",
        page_number=4,
        chunk_type="text",
        content="Chairman message.",
        score=0.9,
        extra_metadata={"source_format": "pdf"},
    )
    context = build_llm_context([chunk])
    assert "Page: 4" in context


def test_select_chunks_for_llm_context_trims_by_priority(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_context_max_chars", 500)
    chunks = [
        _chunk("primary", content="x" * 400, score=0.9, sheet_role="primary"),
        _chunk("country", content="United Kingdom | 80117401", score=0.0, sheet_role="standalone"),
        _chunk("cast", content="Actor A | 80117401", score=0.0, sheet_role="satellite"),
    ]
    selected = select_chunks_for_llm_context(chunks)
    selected_ids = {chunk.chunk_id for chunk in selected}
    assert "country" in selected_ids
    assert "cast" in selected_ids
    assert len(build_llm_context(selected)) <= 500


def test_select_chunks_for_llm_context_no_limit_when_zero(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_context_max_chars", 0)
    chunks = [_chunk("a", content="x" * 5000), _chunk("b", content="y" * 5000)]
    assert len(select_chunks_for_llm_context(chunks)) == 2
