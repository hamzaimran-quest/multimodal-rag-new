"""Tests for document scope and inventory hints."""

from __future__ import annotations

from app.retrieval.scope import doc_inventory_hint_for_agent


def test_doc_inventory_hint_lists_indexed_documents() -> None:
    class FakeClient:
        pass

    def fake_list(client, user_id):
        assert user_id == 7
        return [
            {
                "doc_id": "d1",
                "filename": "huawei.pdf",
                "ingestion_status": "indexed",
                "page_count": 28,
                "chunk_count": 120,
            },
            {
                "doc_id": "d2",
                "filename": "draft.pdf",
                "ingestion_status": "processing",
                "page_count": 0,
                "chunk_count": 0,
            },
        ]

    import app.retrieval.scope as scope_module

    original = scope_module.list_document_records
    scope_module.list_document_records = fake_list
    try:
        hint = doc_inventory_hint_for_agent(FakeClient(), user_id=7, scope_doc_ids=None)
    finally:
        scope_module.list_document_records = original

    assert "huawei.pdf" in hint
    assert "28 pages" in hint
    assert "120 chunks" in hint
    assert "draft.pdf" not in hint


def test_doc_inventory_hint_empty_library() -> None:
    class FakeClient:
        pass

    import app.retrieval.scope as scope_module

    original = scope_module.list_document_records
    scope_module.list_document_records = lambda client, user_id: []
    try:
        hint = doc_inventory_hint_for_agent(FakeClient(), user_id=1, scope_doc_ids=None)
    finally:
        scope_module.list_document_records = original

    assert "no indexed documents" in hint.lower()
