"""Tests for XLSX schema validation (no LLM)."""

from __future__ import annotations

from app.config import settings
from app.ingestion.xlsx_schema import validate_workbook_schema
from app.ingestion.xlsx_workbook import SheetData, WorkbookData


def _workbook() -> WorkbookData:
    shows = SheetData(
        name="Shows",
        index=1,
        headers=["show_id", "title"],
        rows=[(2, ["1", "Alpha"]), (3, ["2", "Beta"])],
    )
    countries = SheetData(
        name="Countries",
        index=2,
        headers=["show_id", "country"],
        rows=[(2, ["1", "France"]), (3, ["2", "Germany"])],
    )
    cast = SheetData(
        name="Cast",
        index=3,
        headers=["show_id", "actor"],
        rows=[(2, ["1", "Alice"]), (3, ["1", "Bob"]), (4, ["2", "Carol"])],
    )
    return WorkbookData(sheets=[shows, countries, cast])


def test_validate_accepts_high_overlap_joins(monkeypatch) -> None:
    monkeypatch.setattr(settings, "excel_schema_min_overlap_ratio", 0.9)
    proposal = {
        "clusters": [
            {
                "primary_sheet": "Shows",
                "primary_key_column": "show_id",
                "satellites": [
                    {
                        "sheet": "Countries",
                        "key_column": "show_id",
                        "payload_columns": ["country"],
                        "cardinality": "one_to_one",
                    },
                    {
                        "sheet": "Cast",
                        "key_column": "show_id",
                        "payload_columns": ["actor"],
                        "cardinality": "one_to_many",
                    },
                ],
            }
        ],
        "standalone_sheets": [],
    }
    schema = validate_workbook_schema(_workbook(), proposal)
    assert len(schema.clusters) == 1
    assert len(schema.clusters[0].satellites) == 2
    assert schema.clusters[0].satellites[1].cardinality == "one_to_many"
    assert all(entry.status == "accepted" for entry in schema.validation_log)


def test_validate_rejects_low_overlap_join(monkeypatch) -> None:
    monkeypatch.setattr(settings, "excel_schema_min_overlap_ratio", 0.9)
    workbook = WorkbookData(
        sheets=[
            SheetData(
                name="Shows",
                index=1,
                headers=["show_id", "title"],
                rows=[(2, ["1", "Alpha"])],
            ),
            SheetData(
                name="Countries",
                index=2,
                headers=["show_id", "country"],
                rows=[(2, ["999", "France"])],
            ),
        ]
    )
    proposal = {
        "clusters": [
            {
                "primary_sheet": "Shows",
                "primary_key_column": "show_id",
                "satellites": [
                    {
                        "sheet": "Countries",
                        "key_column": "show_id",
                        "payload_columns": ["country"],
                        "cardinality": "one_to_one",
                    }
                ],
            }
        ],
        "standalone_sheets": [],
    }
    schema = validate_workbook_schema(workbook, proposal)
    assert schema.clusters == []
    assert "Countries" in schema.standalone_sheets
    assert any(
        entry.status == "rejected" and entry.reason == "overlap_below_threshold"
        for entry in schema.validation_log
    )


def test_no_proposal_indexes_all_sheets_standalone() -> None:
    schema = validate_workbook_schema(_workbook(), None, llm_error="disabled")
    assert schema.clusters == []
    assert set(schema.standalone_sheets) == {"Shows", "Countries", "Cast"}
