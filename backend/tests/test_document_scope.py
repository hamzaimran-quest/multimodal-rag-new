"""Tests for UI document scope resolution."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.retrieval.scope import merge_scope_doc_ids, scope_hint_for_agent, validate_scope_doc_ids


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
