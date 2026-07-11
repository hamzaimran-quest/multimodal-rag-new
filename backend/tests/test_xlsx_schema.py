"""Tests for XLSX schema validation (no LLM)."""

from __future__ import annotations

import json as json_module

import httpx
import pytest

from app.config import settings
from app.ingestion.xlsx_schema import (
    WorkbookSchemaRecognitionError,
    propose_workbook_schema,
    validate_workbook_schema,
)
from app.ingestion.xlsx_workbook import SheetData, WorkbookData


def _workbook() -> WorkbookData:
    shows = SheetData(
        name="Shows",
        index=1,
        headers=["show_id", "title", "type"],
        rows=[(2, ["1", "Alpha", "Series"]), (3, ["2", "Beta", "Film"])],
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


def test_validate_rejects_missing_proposal() -> None:
    with pytest.raises(WorkbookSchemaRecognitionError):
        validate_workbook_schema(_workbook(), None, llm_error="missing_proposal")


def test_validate_records_soft_link_for_near_miss_overlap(monkeypatch) -> None:
    monkeypatch.setattr(settings, "excel_schema_min_overlap_ratio", 0.9)
    monkeypatch.setattr(settings, "excel_schema_soft_link_overlap_ratio", 0.85)
    show_rows = [(index + 2, [str(index), f"Title {index}"]) for index in range(1, 18)]
    country_rows = [(index + 2, [str(index), f"Country {index}"]) for index in range(1, 18)]
    country_rows.extend([(20, ["998", "X"]), (21, ["999", "Y"]), (22, ["997", "Z"])])
    workbook = WorkbookData(
        sheets=[
            SheetData(
                name="Shows",
                index=1,
                headers=["show_id", "title"],
                rows=show_rows,
            ),
            SheetData(
                name="Countries",
                index=2,
                headers=["show_id", "country"],
                rows=country_rows,
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
    assert len(schema.soft_links) == 1
    assert schema.soft_links[0]["sheet"] == "Countries"
    assert schema.soft_links[0]["overlap_ratio"] == 0.85


def test_validate_builds_standalone_fk_links(monkeypatch) -> None:
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
                    }
                ],
            }
        ],
        "standalone_sheets": [],
    }
    schema = validate_workbook_schema(_workbook(), proposal)
    assert schema.standalone_fk_links == [
        {
            "primary_sheet": "Shows",
            "primary_key_column": "show_id",
            "sheet": "Cast",
            "key_column": "show_id",
        }
    ]


def test_propose_workbook_schema_retries_without_json_mode(monkeypatch) -> None:
    calls: list[bool] = []

    class FakeResponse:
        def __init__(self, *, status_code: int, payload: dict | None = None, text: str = "") -> None:
            self.status_code = status_code
            self._payload = payload or {}
            self.text = text or json_module.dumps(self._payload)

        def json(self) -> dict:
            return self._payload

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
                raise httpx.HTTPStatusError("error", request=request, response=self)

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, url, headers=None, json=None):
            use_json_object = "response_format" in (json or {})
            calls.append(use_json_object)
            if use_json_object:
                return FakeResponse(
                    status_code=400,
                    payload={
                        "error": {
                            "code": "json_validate_failed",
                            "message": "Failed to validate JSON.",
                        }
                    },
                )
            return FakeResponse(
                status_code=200,
                payload={
                    "choices": [
                        {
                            "message": {
                                "content": json_module.dumps(
                                    {
                                        "clusters": [
                                            {
                                                "primary_sheet": "Shows",
                                                "primary_key_column": "show_id",
                                                "satellites": [],
                                            }
                                        ],
                                        "standalone_sheets": ["Countries", "Cast"],
                                    }
                                )
                            }
                        }
                    ]
                },
            )

    monkeypatch.setattr(settings, "excel_schema_max_retries", 3)
    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    monkeypatch.setattr("app.ingestion.xlsx_schema.httpx.Client", FakeClient)

    proposal = propose_workbook_schema(_workbook())
    assert proposal["clusters"][0]["primary_sheet"] == "Shows"
    assert calls == [True, False]


def test_propose_workbook_schema_raises_after_max_retries(monkeypatch) -> None:
    class FakeResponse:
        def __init__(self) -> None:
            self.status_code = 400
            self._payload = {
                "error": {
                    "code": "json_validate_failed",
                    "message": "Failed to validate JSON.",
                }
            }
            self.text = json_module.dumps(self._payload)

        def json(self) -> dict:
            return self._payload

        def raise_for_status(self) -> None:
            request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
            raise httpx.HTTPStatusError("error", request=request, response=self)

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, url, headers=None, json=None):
            return FakeResponse()

    monkeypatch.setattr(settings, "excel_schema_max_retries", 3)
    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    monkeypatch.setattr("app.ingestion.xlsx_schema.httpx.Client", FakeClient)

    with pytest.raises(WorkbookSchemaRecognitionError) as exc_info:
        propose_workbook_schema(_workbook())
    assert str(exc_info.value) == "LLM failed to recognize workbook schema. Please try again."

