"""Unit tests for DOCX viewer bbox lookup and path resolution."""

from __future__ import annotations

from pathlib import Path

from app.ingestion.docx_bbox_lookup import (
    _lcs_matched_words,
    _needle_tokens,
    _table_markdown_to_plain,
    locate_chunks_in_viewer_pdf,
)
from app.ingestion.models import ExtractedChunk
from app.ingestion.pipeline import (
    VIEWER_PDF_NAME,
    find_document_path,
    find_pdf_path,
    find_viewer_pdf_path,
    viewer_pdf_path,
)
from tests.pdf_fixtures import build_sample_pdf

SAMPLE_TEXT = (
    "Five-Year Financial Highlights. Revenue grew steadily across regions. "
    "Operating profit improved year over year with disciplined cost control. "
    "Net income increased due to stronger margins and stable demand."
)


def test_table_markdown_to_plain_concatenates_cells() -> None:
    markdown = (
        "| Region | Revenue |\n"
        "| --- | --- |\n"
        "| North | 120 |\n"
        "| South | 95 |\n"
    )
    assert _table_markdown_to_plain(markdown) == "Region Revenue North 120 South 95"


def _fake_word(text: str, x0: float) -> dict:
    return {"text": text, "x0": x0, "x1": x0 + 10, "top": 100.0, "bottom": 110.0}


def test_lcs_matches_interleaved_table_words() -> None:
    """A floating table's cells are interleaved with body prose in PDF reading
    order; the chunk tokens must still be recovered as an in-order subsequence."""
    chunk = ExtractedChunk(
        content="| ITEM | NEEDED |\n| --- | --- |\n| Books | 1 |\n| Pens | 3 |",
        page_number=1,
        chunk_type="table",
        extraction_method="docx_native",
        extra_metadata={"source_format": "docx"},
    )
    needle = _needle_tokens(chunk)
    assert needle == ["item", "needed", "books", "1", "pens", "3"]

    stream = [
        "Tables", "ITEM", "NEEDED", "Tables", "in", "Word", "vary", "Books", "1",
        "simple", "to", "the", "complex", "Pens", "3", "majority", "of", "cases",
    ]
    page_words = [_fake_word(tok, i * 12.0) for i, tok in enumerate(stream)]

    matched = _lcs_matched_words(needle, page_words)
    assert [w["text"] for w in matched] == ["ITEM", "NEEDED", "Books", "1", "Pens", "3"]


def test_lcs_ignores_punctuation_and_case() -> None:
    needle = _needle_tokens(
        ExtractedChunk(
            content="Revenue grew steadily.",
            page_number=1,
            chunk_type="text",
            extraction_method="docx_native",
        )
    )
    page_words = [_fake_word(t, i * 12.0) for i, t in enumerate(["REVENUE,", "grew", "Steadily!"])]
    matched = _lcs_matched_words(needle, page_words)
    assert len(matched) == 3


def test_locate_chunks_in_viewer_pdf_matches_known_text(tmp_path: Path) -> None:
    pdf_path = build_sample_pdf(tmp_path / "viewer.pdf")
    chunk = ExtractedChunk(
        content=SAMPLE_TEXT,
        page_number=3,
        chunk_type="text",
        extraction_method="docx_native",
        extra_metadata={"source_format": "docx", "block_index": 3},
    )

    locate_chunks_in_viewer_pdf([chunk], pdf_path, total_blocks=10)

    location = chunk.extra_metadata["viewer_location"]
    assert location["match_status"] == "ok"
    assert location["viewer_page"] == 1
    assert isinstance(location["bbox"], list) and len(location["bbox"]) == 4
    assert isinstance(location["line_bboxes"], list) and len(location["line_bboxes"]) >= 1


def test_locate_chunks_records_failed_match(tmp_path: Path) -> None:
    pdf_path = build_sample_pdf(tmp_path / "viewer.pdf")
    chunk = ExtractedChunk(
        content="This text does not appear anywhere in the rendered preview document at all.",
        page_number=1,
        chunk_type="text",
        extraction_method="docx_native",
        extra_metadata={"source_format": "docx", "block_index": 1},
    )

    locate_chunks_in_viewer_pdf([chunk], pdf_path, total_blocks=1)

    assert chunk.extra_metadata["viewer_location"]["match_status"] == "failed"


def test_find_document_path_prefers_stored_filename(tmp_path, monkeypatch) -> None:
    uploads = tmp_path / "uploads"
    doc_dir = uploads / "7" / "doc-1"
    doc_dir.mkdir(parents=True)
    (doc_dir / "report.docx").write_bytes(b"docx")
    (doc_dir / VIEWER_PDF_NAME).write_bytes(b"%PDF-fake")

    monkeypatch.setattr("app.config.settings.uploads_dir", uploads)

    assert find_document_path(7, "doc-1", "report.docx") == doc_dir / "report.docx"
    assert find_viewer_pdf_path(7, "doc-1") == doc_dir / VIEWER_PDF_NAME
    assert find_pdf_path(7, "doc-1") == doc_dir / VIEWER_PDF_NAME
    assert viewer_pdf_path(7, "doc-1") == doc_dir / VIEWER_PDF_NAME


def test_find_pdf_path_returns_native_upload_when_no_viewer(tmp_path, monkeypatch) -> None:
    uploads = tmp_path / "uploads"
    doc_dir = uploads / "2" / "doc-2"
    doc_dir.mkdir(parents=True)
    native = doc_dir / "annual.pdf"
    native.write_bytes(b"%PDF-native")

    monkeypatch.setattr("app.config.settings.uploads_dir", uploads)

    assert find_pdf_path(2, "doc-2") == native
