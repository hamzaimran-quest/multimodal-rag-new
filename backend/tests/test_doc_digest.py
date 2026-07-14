"""Tests for document digest building and router hints."""

from __future__ import annotations

from app.ingestion.models import ExtractedChunk
from app.retrieval.doc_digest import build_doc_digest_from_chunks, digest_text_from_record
from app.retrieval.scope import doc_digest_hint_for_agent


def _chunk(
    *,
    page: int,
    chunk_type: str = "text",
    section: str | None = None,
    subsection: str | None = None,
    source_format: str = "pdf",
    sheet_name: str | None = None,
    block_index: int = 0,
) -> ExtractedChunk:
    extra: dict = {"source_format": source_format, "block_index": block_index}
    if section:
        extra["section"] = section
    if subsection:
        extra["subsection"] = subsection
    if sheet_name:
        extra["sheet_name"] = sheet_name
    return ExtractedChunk(
        content="sample content for retrieval chunk",
        page_number=page,
        chunk_type=chunk_type,
        extraction_method="test",
        extra_metadata=extra,
    )


def test_build_doc_digest_collects_pdf_sections_in_order() -> None:
    chunks = [
        _chunk(page=4, section="Message from the Rotating Chairwoman", block_index=1),
        _chunk(page=9, section="Five-Year Financial Highlights", block_index=2),
        _chunk(page=23, section="2025 Business Review", subsection="By region", block_index=3),
        _chunk(page=23, section="2025 Business Review", subsection="By region", block_index=4),
    ]
    payload = build_doc_digest_from_chunks(
        filename="huawei.pdf",
        chunks=chunks,
        page_count=28,
    )
    assert payload["source_format"] == "pdf"
    assert "Rotating Chairwoman" in payload["digest"]
    assert "Financial Highlights" in payload["digest"]
    assert payload["sections"][0].startswith("Message from")


def test_build_doc_digest_includes_xlsx_sheets() -> None:
    chunks = [
        _chunk(page=1, chunk_type="table", source_format="xlsx", sheet_name="Orders"),
        _chunk(page=2, chunk_type="table", source_format="xlsx", sheet_name="Customers"),
    ]
    payload = build_doc_digest_from_chunks(
        filename="sales.xlsx",
        chunks=chunks,
        workbook_schema={
            "standalone_sheets": ["Summary"],
            "clusters": [{"primary_sheet": "Orders", "satellites": []}],
        },
    )
    assert payload["source_format"] == "xlsx"
    assert "Summary" in payload["digest"]
    assert "Orders" in payload["digest"]


def test_doc_digest_hint_for_agent_renders_cached_outline() -> None:
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
                    "digest": "File: huawei.pdf (pdf)\nSections:\n- Message from the Rotating Chairwoman",
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
        hint = doc_digest_hint_for_agent(FakeClient(), user_id=1, scope_doc_ids=["d1"])
    finally:
        scope_module.list_document_records = original

    assert "Document outlines" in hint
    assert "huawei.pdf" in hint
    assert "Rotating Chairwoman" in hint


def test_digest_text_from_record() -> None:
    assert digest_text_from_record({"doc_digest": {"digest": "outline text"}}) == "outline text"
    assert digest_text_from_record({}) == ""


def test_digest_text_from_record_router_compact() -> None:
    record = {
        "filename": "huawei.pdf",
        "doc_digest": {
            "digest": "File: huawei.pdf (pdf)\nSections:\n- Message\n- Highlights",
            "source_format": "pdf",
            "page_count": 28,
            "sections": [
                "Message from the Rotating Chairwoman",
                "Five-Year Financial Highlights",
                "Extra One",
                "Extra Two",
                "Extra Three",
                "Extra Four",
            ],
            "sheet_names": [],
        },
    }
    text = digest_text_from_record(record, for_router=True)
    assert "huawei.pdf" in text
    assert "28 pages" in text
    assert "Rotating Chairwoman" in text
    assert "Extra Four" not in text
