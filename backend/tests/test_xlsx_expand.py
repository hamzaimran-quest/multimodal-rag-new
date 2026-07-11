"""Tests for anchor-based XLSX retrieval expansion."""

from __future__ import annotations

from app.retrieval.models import RetrievedChunk
from app.retrieval.xlsx_expand import (
    _resolve_anchor_targets,
    expand_xlsx_chunks_by_entity_keys,
    linked_sheets_for_anchor,
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


def test_linked_sheets_for_anchor_includes_cluster_soft_and_standalone_fk() -> None:
    schema = {
        "clusters": [
            {
                "primary_sheet": "Titles",
                "primary_key_column": "show_id",
                "satellites": [{"sheet": "Category", "key_column": "show_id"}],
            }
        ],
        "soft_links": [
            {
                "sheet": "Countries",
                "key_column": "show_id",
                "overlap_ratio": 0.867,
            }
        ],
        "standalone_fk_links": [
            {
                "sheet": "Directors",
                "key_column": "show_id",
                "primary_sheet": "Titles",
            }
        ],
        "standalone_sheets": ["Countries", "Cast", "Directors"],
    }
    sheets = linked_sheets_for_anchor(schema, {"Cast": "show_id"})
    assert sheets["Titles"] == "show_id"
    assert sheets["Category"] == "show_id"
    assert sheets["Countries"] == "show_id"
    assert sheets["Directors"] == "show_id"
    assert sheets["Cast"] == "show_id"


def test_linked_sheets_infers_standalone_fk_for_legacy_schema() -> None:
    schema = {
        "clusters": [
            {
                "primary_sheet": "netflix_titles",
                "primary_key_column": "show_id",
                "satellites": [{"sheet": "netflix_titles_category", "key_column": "show_id"}],
            }
        ],
        "standalone_sheets": [
            "netflix_titles_countries",
            "netflix_titles_cast",
            "netflix_titles_directors",
        ],
    }
    sheets = linked_sheets_for_anchor(schema, {})
    assert sheets["netflix_titles_countries"] == "show_id"
    assert sheets["netflix_titles_directors"] == "show_id"


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


def test_expand_fetches_missing_linked_sheets_per_sheet(monkeypatch) -> None:
    primary = _chunk(
        "primary",
        content="80117401 | Jandino: Whatever it Takes | Stand-Up Comedy",
        extra_metadata={
            "source_format": "xlsx",
            "content_format": "slim_rows",
            "table_headers": ["show_id", "title", "category"],
            "sheet_row_map": [10],
            "entity_key_column": "show_id",
            "row_entity_keys": {"10": "80117401"},
            "sheet_name": "Titles",
        },
    )
    cast = _chunk(
        "cast",
        content="Jandino Asporaat | 80117401",
        extra_metadata={
            "source_format": "xlsx",
            "content_format": "slim_rows",
            "table_headers": ["cast", "show_id"],
            "sheet_row_map": [4],
            "entity_key_column": "show_id",
            "entity_keys": ["80117401"],
            "row_entity_keys": {"4": "80117401"},
            "sheet_name": "Cast",
        },
        score=0.2,
    )
    country = _chunk(
        "country",
        content="80117401 | Netherlands",
        extra_metadata={
            "source_format": "xlsx",
            "content_format": "slim_rows",
            "table_headers": ["show_id", "country"],
            "sheet_row_map": [4],
            "entity_key_column": "show_id",
            "entity_keys": ["80117401"],
            "row_entity_keys": {"4": "80117401"},
            "sheet_name": "Countries",
        },
        score=0.1,
    )

    fetched: list[str] = []

    def fake_fetch_sheet(client, *, doc_id, anchor_key, sheet_name, user_id):
        fetched.append(sheet_name)
        if sheet_name == "Countries":
            return country
        if sheet_name == "Cast":
            return cast
        return None

    def fake_schema(client, *, doc_id, user_id, cache):
        cache[doc_id] = {
            "clusters": [
                {
                    "primary_sheet": "Titles",
                    "primary_key_column": "show_id",
                    "satellites": [{"sheet": "Category", "key_column": "show_id"}],
                }
            ],
            "soft_links": [{"sheet": "Countries", "key_column": "show_id"}],
            "standalone_fk_links": [{"sheet": "Cast", "key_column": "show_id"}],
        }
        return cache[doc_id]

    monkeypatch.setattr("app.retrieval.xlsx_expand.fetch_chunk_for_anchor_on_sheet", fake_fetch_sheet)
    monkeypatch.setattr("app.retrieval.xlsx_expand._workbook_schema_for_doc", fake_schema)

    merged, anchor_keys = expand_xlsx_chunks_by_entity_keys(
        object(),
        [primary],
        query="Jandino Whatever it Takes country",
        user_id=1,
    )
    assert {chunk.chunk_id for chunk in merged} == {"primary", "country", "cast"}
    assert anchor_keys == {"80117401"}
    assert "Countries" in fetched
    assert "Cast" in fetched
    assert "Titles" not in fetched


def test_expand_retries_anchor_resolution_with_fallback_query(monkeypatch) -> None:
    primary = _chunk(
        "primary",
        content="80117401 | Jandino: Whatever it Takes | Stand-Up Comedy",
        extra_metadata={
            "source_format": "xlsx",
            "content_format": "slim_rows",
            "table_headers": ["show_id", "title", "category"],
            "sheet_row_map": [10],
            "entity_key_column": "show_id",
            "row_entity_keys": {"10": "80117401"},
            "sheet_name": "Titles",
        },
    )

    def fake_fetch_sheet(client, *, doc_id, anchor_key, sheet_name, user_id):
        return None

    def fake_schema(client, *, doc_id, user_id, cache):
        cache[doc_id] = {
            "clusters": [
                {
                    "primary_sheet": "Titles",
                    "primary_key_column": "show_id",
                    "satellites": [],
                }
            ],
            "standalone_sheets": ["Titles", "Cast"],
        }
        return cache[doc_id]

    monkeypatch.setattr("app.retrieval.xlsx_expand.fetch_chunk_for_anchor_on_sheet", fake_fetch_sheet)
    monkeypatch.setattr("app.retrieval.xlsx_expand._workbook_schema_for_doc", fake_schema)
    monkeypatch.setattr(
        "app.retrieval.xlsx_expand.fetch_chunks_by_entity_keys",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy expand should not run")),
    )

    merged, anchor_keys = expand_xlsx_chunks_by_entity_keys(
        object(),
        [primary],
        query="Whatever it Takes category country cast title",
        user_id=1,
        anchor_fallback_query="tell me about Jandino: Whatever it Takes, its category and cast",
    )
    assert anchor_keys == {"80117401"}
    assert len(merged) == 1
