"""Tests for XLSX slim row serialization."""

from __future__ import annotations

from app.ingestion.xlsx_serialize import (
    format_chunk_content_for_llm,
    resolve_row_band_size,
    rows_to_slim_values_text,
    slim_values_text_to_rows,
    table_rows_from_chunk_content,
)


def test_rows_to_slim_values_text_omits_markdown_overhead():
    text = rows_to_slim_values_text([
        ["Afghanistan", "2023", "8"],
        ["Pakistan", "2023", "97"],
    ])
    assert "|" not in text.splitlines()[0][:1]
    assert "Afghanistan | 2023 | 8" in text
    assert "| ---" not in text


def test_slim_values_roundtrip_with_headers():
    headers = ["Country", "Year", "Rank"]
    content = rows_to_slim_values_text([["Afghanistan", "2023", "8"]])
    rows = slim_values_text_to_rows(headers, content)
    assert rows[0] == headers
    assert rows[1] == ["Afghanistan", "2023", "8"]


def test_table_rows_from_chunk_content_supports_slim_and_markdown():
    slim_rows = table_rows_from_chunk_content(
        "Jane Doe | 2023-08-20 | 50000",
        {
            "content_format": "slim_rows",
            "table_headers": ["Full Name", "Hire Date", "Salary"],
        },
    )
    assert slim_rows[1][0] == "Jane Doe"

    markdown_rows = table_rows_from_chunk_content(
        "| Year | Revenue |\n| --- | --- |\n| 2023 | 100 |",
        {"content_format": "markdown_table"},
    )
    assert markdown_rows[0] == ["Year", "Revenue"]


def test_format_chunk_content_for_llm_includes_slim_headers():
    content = rows_to_slim_values_text([["Somalia", "2023", "1st", "111.9"]])
    formatted = format_chunk_content_for_llm(
        content,
        {
            "content_format": "slim_rows",
            "table_headers": ["Country", "Year", "Rank", "Total"],
        },
    )
    assert "Column headers:" in formatted
    assert "Country | Year | Rank | Total" in formatted
    assert "Somalia | 2023 | 1st | 111.9" in formatted


def test_format_chunk_content_for_llm_passthrough_markdown():
    markdown = "| Year | Revenue |\n| --- | --- |\n| 2023 | 100 |"
    assert format_chunk_content_for_llm(markdown, {"content_format": "markdown_table"}) == markdown


def test_resolve_row_band_size_adapts_to_column_count(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "excel_wide_column_threshold", 10)
    monkeypatch.setattr(settings, "excel_medium_column_threshold", 6)
    monkeypatch.setattr(settings, "excel_row_band_size_wide", 10)
    monkeypatch.setattr(settings, "excel_row_band_size_medium", 15)
    monkeypatch.setattr(settings, "excel_row_band_size", 30)

    assert resolve_row_band_size(16) == 10
    assert resolve_row_band_size(8) == 15
    assert resolve_row_band_size(3) == 30
