"""Tests for slim row-band extraction on wide XLSX sheets."""

from __future__ import annotations

from pathlib import Path

from app.ingestion.xlsx_extract import extract_xlsx_chunks


def test_fsi_style_sheet_uses_slim_bands_and_more_chunks():
    fsi_path = Path(__file__).resolve().parents[2] / "FSI-2023-DOWNLOAD.xlsx"
    if not fsi_path.is_file():
        return

    chunks = extract_xlsx_chunks(str(fsi_path), doc_id="fsi", user_id=1)
    assert len(chunks) >= 12
    assert all(chunk.extra_metadata.get("content_format") == "slim_rows" for chunk in chunks)
    assert all("| ---" not in chunk.content for chunk in chunks)
    assert all(chunk.extra_metadata.get("table_headers") for chunk in chunks)

    words = [len(chunk.content.split()) for chunk in chunks]
    assert max(words) < 700
