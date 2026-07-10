"""Tests for UI document scope resolution."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.retrieval.models import RetrievedChunk
from app.retrieval.scope import (
    is_xlsx_filename,
    limit_xlsx_chunks,
    merge_scope_doc_ids,
    resolve_search_top_k,
    scope_hint_for_agent,
    scope_is_xlsx_only,
    validate_scope_doc_ids,
)


def test_merge_scope_doc_ids_none_for_empty() -> None:
    assert merge_scope_doc_ids() is None
    assert merge_scope_doc_ids(doc_ids=[]) is None


def test_merge_scope_doc_ids_dedupes() -> None:
    assert merge_scope_doc_ids(doc_ids=["a", "a", "b"], doc_id="b") == ["a", "b"]


def test_validate_scope_doc_ids_rejects_unknown(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.retrieval.scope.get_document_for_user",
        lambda client, doc_id, user_id: None,
    )
    with pytest.raises(HTTPException) as exc:
        validate_scope_doc_ids(object(), user_id=1, doc_ids=["missing"])
    assert exc.value.status_code == 404


def test_scope_hint_for_agent_single_doc() -> None:
    hint = scope_hint_for_agent(["doc-1"], scoped_filenames=["Report.pdf"])
    assert "Report.pdf" in hint
    assert "Do not ask which document" in hint


def test_scope_hint_for_agent_all_docs() -> None:
    hint = scope_hint_for_agent(None)
    assert "all uploaded documents" in hint


def test_is_xlsx_filename() -> None:
    assert is_xlsx_filename("FSI-2023-DOWNLOAD.xlsx")
    assert not is_xlsx_filename("report.pdf")


def test_scope_is_xlsx_only_requires_all_scoped_docs(monkeypatch) -> None:
    def fake_lookup(client, doc_id, user_id):
        return {
            "xlsx-1": {"filename": "data.xlsx"},
            "pdf-1": {"filename": "report.pdf"},
        }.get(doc_id)

    monkeypatch.setattr("app.retrieval.scope.get_document_for_user", fake_lookup)

    assert scope_is_xlsx_only(object(), user_id=1, scope_doc_ids=["xlsx-1"])
    assert not scope_is_xlsx_only(object(), user_id=1, scope_doc_ids=["pdf-1"])
    assert not scope_is_xlsx_only(object(), user_id=1, scope_doc_ids=["xlsx-1", "pdf-1"])
    assert not scope_is_xlsx_only(object(), user_id=1, scope_doc_ids=None)


def test_resolve_search_top_k_uses_excel_cap(monkeypatch) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "excel_top_k", 3)
    monkeypatch.setattr(settings, "default_top_k", 8)
    monkeypatch.setattr(
        "app.retrieval.scope.scope_is_xlsx_only",
        lambda client, user_id, scope_doc_ids: scope_doc_ids == ["xlsx-1"],
    )

    assert resolve_search_top_k(object(), user_id=1, scope_doc_ids=["xlsx-1"], top_k=12) == 3
    assert resolve_search_top_k(object(), user_id=1, scope_doc_ids=["pdf-1"], top_k=12) == 12


def test_limit_xlsx_chunks_keeps_other_formats() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="x1",
            doc_id="d1",
            filename="a.xlsx",
            page_number=1,
            chunk_type="table",
            content="row",
            score=0.9,
            extra_metadata={"source_format": "xlsx"},
        ),
        RetrievedChunk(
            chunk_id="x2",
            doc_id="d1",
            filename="a.xlsx",
            page_number=1,
            chunk_type="table",
            content="row",
            score=0.8,
            extra_metadata={"source_format": "xlsx"},
        ),
        RetrievedChunk(
            chunk_id="x3",
            doc_id="d1",
            filename="a.xlsx",
            page_number=1,
            chunk_type="table",
            content="row",
            score=0.7,
            extra_metadata={"source_format": "xlsx"},
        ),
        RetrievedChunk(
            chunk_id="x4",
            doc_id="d1",
            filename="a.xlsx",
            page_number=1,
            chunk_type="table",
            content="row",
            score=0.6,
            extra_metadata={"source_format": "xlsx"},
        ),
        RetrievedChunk(
            chunk_id="p1",
            doc_id="d2",
            filename="b.pdf",
            page_number=1,
            chunk_type="text",
            content="text",
            score=0.5,
            extra_metadata={"source_format": "pdf"},
        ),
    ]

    limited = limit_xlsx_chunks(chunks, limit=3)
    assert sum(1 for chunk in limited if chunk.chunk_id.startswith("x")) == 3
    assert any(chunk.chunk_id == "p1" for chunk in limited)
