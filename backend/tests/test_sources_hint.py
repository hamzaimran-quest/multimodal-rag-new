"""Tests for combined router sources hint."""

from __future__ import annotations

from app.retrieval.scope import sources_hint_for_agent


def test_sources_hint_combines_db_tables_and_scoped_docs() -> None:
    class FakeClient:
        pass

    def fake_list(client, user_id):
        return [
            {
                "doc_id": "d1",
                "filename": "huawei.pdf",
                "ingestion_status": "indexed",
                "chunk_count": 10,
                "page_count": 28,
                "doc_digest": {
                    "digest": "long digest",
                    "source_format": "pdf",
                    "page_count": 28,
                    "sections": ["Message from the Rotating Chairwoman"],
                    "sheet_names": [],
                },
            }
        ]

    import app.retrieval.scope as scope_module

    original = scope_module.list_document_records
    scope_module.list_document_records = fake_list
    try:
        hint = sources_hint_for_agent(
            FakeClient(),
            user_id=1,
            scope_doc_ids=["d1"],
            scoped_filenames=["huawei.pdf"],
            sql_display_name="Huawei DB",
            sql_description="Annual report mirror",
            sql_tables=["business_segments", "financial_highlights"],
        )
    finally:
        scope_module.list_document_records = original

    assert "## Available sources" in hint
    assert "Huawei DB" in hint
    assert "business_segments" in hint
    assert "huawei.pdf" in hint
    assert "query_database" in hint and "search_documents" in hint
    # Full schema-style multi-line column dumps must not appear.
    assert "revenue_2025_cny_million:" not in hint
