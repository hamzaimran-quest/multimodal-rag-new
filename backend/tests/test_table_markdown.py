"""Unit tests for table markdown conversion."""

from app.ingestion.tables import table_signature, table_to_markdown


def test_table_to_markdown_basic():
    rows = [
        ["Year", "Revenue"],
        ["2023", "100"],
        ["2024", "120"],
    ]
    md = table_to_markdown(rows)
    assert "| Year | Revenue |" in md
    assert "| 2023 | 100 |" in md
    assert "| --- | --- |" in md


def test_table_signature():
    rows = [["A", "B"], ["1", "2"]]
    cols, header = table_signature(rows)
    assert cols == 2
    assert header == ("A", "B")
