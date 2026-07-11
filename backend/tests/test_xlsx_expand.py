"""Tests for anchor-based XLSX retrieval expansion."""

from __future__ import annotations

from app.retrieval.models import RetrievedChunk
from app.retrieval.xlsx_expand import (
    _linked_sheets_from_schema,
    _resolve_anchor_targets,
    expand_xlsx_chunks_by_entity_keys,
)


def _chunk(
    chunk_id: str,
    *,
    doc_id: str = "d1",
    content: str,
    extra_metadata: dict,
    score: float = 1.0,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        filename="workbook.xlsx",
        page_number=1,
        chunk_type="table",
        content=content,
        score=score,
        extra_metadata=extra_metadata,
    )


def test_linked_sheets_from_schema_includes_primary_and_satellites() -> None:
    sheets = _linked_sheets_from_schema(
        {
            "clusters": [
                {
                    "primary_sheet": "Titles",
                    "primary_key_column": "show_id",
                    "satellites": [
                        {"sheet": "Countries", "key_column": "show_id"},
                        {"sheet": "Cast", "key_column": "show_id"},
                    ],
                }
            ]
        }
    )
    assert sheets == {
        "Titles": "show_id",
        "Countries": "show_id",
        "Cast": "show_id",
    }


def test_resolve_anchor_targets_prefers_query_matching_row() -> None:
    chunks = [
        _chunk(
            "primary",
            content="70234439 | Transformers Prime | Kids' TV",
            extra_metadata={
                "source_format": "xlsx",
                "content_format": "slim_rows",
                "table_headers": ["show_id", "title", "category"],
                "sheet_row_map": [10],
                "entity_key_column": "show_id",
                "row_entity_keys": {"10": "70234439"},
            },
        ),
        _chunk(
            "noise",
            content="111 | Other | Comedy\n222 | Another | Drama",
            extra_metadata={
                "source_format": "xlsx",
                "content_format": "slim_rows",
                "table_headers": ["show_id", "title", "category"],
                "sheet_row_map": [2, 3],
                "entity_key_column": "show_id",
                "entity_keys": ["111", "222"],
                "row_entity_keys": {"2": "111", "3": "222"},
            },
            score=0.95,
        ),
    ]

    anchors = _resolve_anchor_targets(chunks, "Transformers Prime country and cast")
    assert anchors[0][1] == "70234439"


def test_expand_xlsx_chunks_fetches_anchor_linked_chunks(monkeypatch) -> None:
    primary = _chunk(
        "primary",
        content="70234439 | Transformers Prime | Kids' TV",
        extra_metadata={
            "source_format": "xlsx",
            "content_format": "slim_rows",
            "table_headers": ["show_id", "title", "category"],
            "sheet_row_map": [10],
            "entity_key_column": "show_id",
            "row_entity_keys": {"10": "70234439"},
            "sheet_name": "Titles",
        },
    )
    country = _chunk(
        "country",
        content="70234439 | United States",
        extra_metadata={
            "source_format": "xlsx",
            "content_format": "slim_rows",
            "table_headers": ["show_id", "country"],
            "sheet_row_map": [4],
            "entity_key_column": "show_id",
            "entity_keys": ["70234439"],
            "row_entity_keys": {"4": "70234439"},
            "sheet_name": "Countries",
        },
        score=0.2,
    )

    def fake_fetch(client, *, doc_id, anchor_key, user_id, sheet_names=None, max_chunks=None):
        assert anchor_key == "70234439"
        return [country]

    def fake_schema(client, *, doc_id, user_id, cache):
        cache[doc_id] = {
            "clusters": [
                {
                    "primary_sheet": "Titles",
                    "primary_key_column": "show_id",
                    "satellites": [{"sheet": "Countries", "key_column": "show_id"}],
                }
            ]
        }
        return cache[doc_id]

    monkeypatch.setattr("app.retrieval.xlsx_expand.fetch_chunks_for_anchor", fake_fetch)
    monkeypatch.setattr("app.retrieval.xlsx_expand._workbook_schema_for_doc", fake_schema)

    merged = expand_xlsx_chunks_by_entity_keys(
        object(),
        [primary],
        query="Transformers Prime country",
        user_id=1,
    )
    assert {chunk.chunk_id for chunk in merged} == {"primary", "country"}
