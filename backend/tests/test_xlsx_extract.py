"""Tests for XLSX extraction."""

from __future__ import annotations

from pathlib import Path

from app.ingestion.xlsx_extract import count_visible_sheets, extract_xlsx_chunks
from tests.xlsx_fixtures import build_sample_xlsx


def test_extract_xlsx_prefers_tables_and_skips_hidden(tmp_path: Path):
    xlsx_path = build_sample_xlsx(tmp_path / "sample.xlsx")
    chunks = extract_xlsx_chunks(str(xlsx_path), doc_id="doc-1", user_id=1)

    assert count_visible_sheets(str(xlsx_path)) == 2
    assert len(chunks) >= 2
    assert all(chunk.chunk_type == "table" for chunk in chunks)
    assert all(chunk.extra_metadata.get("source_format") == "xlsx" for chunk in chunks)

    revenue_chunks = [chunk for chunk in chunks if chunk.extra_metadata.get("sheet_name") == "Revenue"]
    assert len(revenue_chunks) == 1
    assert "2024" in revenue_chunks[0].content
    assert revenue_chunks[0].extra_metadata["row_range"] == [1, 3]
    assert revenue_chunks[0].extra_metadata.get("content_format") == "markdown_table"

    hidden_chunks = [chunk for chunk in chunks if chunk.extra_metadata.get("sheet_name") == "Hidden"]
    assert hidden_chunks == []

    band_chunks = [chunk for chunk in chunks if chunk.extra_metadata.get("sheet_name") == "Bands"]
    assert len(band_chunks) == 1
    assert band_chunks[0].extra_metadata.get("content_format") == "slim_rows"
    assert band_chunks[0].extra_metadata.get("table_headers") == ["Metric", "Value"]
    assert "| ---" not in band_chunks[0].content
    assert "Row 5 | 50" in band_chunks[0].content
