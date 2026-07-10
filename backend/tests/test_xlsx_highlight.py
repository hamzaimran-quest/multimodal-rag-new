"""Tests for XLSX row highlight matching."""

from __future__ import annotations

from app.ingestion.xlsx_highlight import apply_xlsx_highlights_to_sources, match_xlsx_highlight_row_range
from app.retrieval.models import RetrievedChunk


EMPLOYEE_TABLE = """\
| Employee ID | Full Name | Department | Designation | Hire Date | Annual Salary |
| --- | --- | --- | --- | --- | --- |
| E1001 | John Smith | Human Resources | HR Manager | 2023-05-15 | 60000 |
| E1002 | Jane Doe | Marketing | Marketing Specialist | 2023-08-20 | 50000 |
| E1003 | Michael Johnson | Engineering | Software Engineer | 2022-11-01 | 75000 |
| E1004 | Emily Brown | Finance | Financial Analyst | 2023-07-10 | 55000 |
| E1005 | David Wilson | Information Technology | IT Specialist | 2023-09-12 | 60000 |
"""


def test_match_xlsx_highlight_prefers_answer_date_over_query_noise() -> None:
    row_range = match_xlsx_highlight_row_range(
        EMPLOYEE_TABLE,
        [2, 43],
        [2, 3, 4, 5, 6, 7, 8],
        "When was Jane Doe hired?",
        "Jane Doe was hired on 2023-08-20.",
    )
    assert row_range == [4, 4]


def test_match_xlsx_highlight_uses_answer_name_when_query_is_vague() -> None:
    row_range = match_xlsx_highlight_row_range(
        EMPLOYEE_TABLE,
        [2, 43],
        [2, 3, 4, 5, 6, 7, 8],
        "Who joined in August 2023?",
        "Jane Doe joined on 2023-08-20.",
    )
    assert row_range == [4, 4]


def test_match_xlsx_highlight_keeps_small_ranges() -> None:
    row_range = match_xlsx_highlight_row_range(
        EMPLOYEE_TABLE,
        [4, 5],
        [3, 4, 5],
        "Jane Doe",
        "Jane Doe was hired on 2023-08-20.",
    )
    assert row_range == [4, 5]


def test_match_xlsx_highlight_falls_back_without_match() -> None:
    row_range = match_xlsx_highlight_row_range(
        EMPLOYEE_TABLE,
        [2, 43],
        None,
        "quarterly revenue forecast",
        "I do not have that information.",
    )
    assert row_range == [2, 43]


def test_apply_xlsx_highlights_to_sources_updates_matching_source() -> None:
    chunk = RetrievedChunk(
        chunk_id="c1",
        doc_id="d1",
        filename="employees.xlsx",
        page_number=1,
        chunk_type="table",
        content=EMPLOYEE_TABLE,
        score=0.9,
        extra_metadata={
            "source_format": "xlsx",
            "sheet_name": "Employees",
            "row_range": [2, 43],
            "sheet_row_map": [2, 3, 4, 5, 6, 7, 8],
        },
    )
    sources = [
        {
            "chunk_id": "c1",
            "source_format": "xlsx",
            "row_range": [2, 43],
        }
    ]

    apply_xlsx_highlights_to_sources(
        sources,
        [chunk],
        query="When was Jane Doe hired?",
        answer="Jane Doe was hired on 2023-08-20.",
    )

    assert sources[0]["row_range"] == [4, 4]
