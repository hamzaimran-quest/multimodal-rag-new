"""Tests for PDF font-size heading detection and section metadata."""

from __future__ import annotations

from pathlib import Path

import pdfplumber
import pytest

from app.config import PROJECT_ROOT
from app.ingestion.pdf_headings import classify_line, group_lines, structured_paragraphs_for_page
from app.ingestion.pdf_layout import column_bands_for_reading
from app.ingestion.text import _text_chunks_with_bbox
from tests.pdf_fixtures import build_two_column_headings_pdf

HUAWEI_PDF = PROJECT_ROOT / "huawei.pdf"


def test_classify_line_distinguishes_heading_subheading_and_body():
    body_size = 10.0
    heading_line = [{"text": "Main", "size": 20.0, "top": 0, "bottom": 12, "x0": 0, "x1": 10}]
    subheading_line = [{"text": "Sub", "size": 13.0, "top": 0, "bottom": 10, "x0": 0, "x1": 10}]
    body_line = [{"text": "Body", "size": 10.0, "top": 0, "bottom": 8, "x0": 0, "x1": 10}]

    assert classify_line(heading_line, body_size) == "heading"
    assert classify_line(subheading_line, body_size) == "subheading"
    assert classify_line(body_line, body_size) == "body"


@pytest.mark.skipif(not HUAWEI_PDF.exists(), reason="huawei.pdf not found at project root")
def test_huawei_page10_chairman_heading_attached_to_first_body_chunk():
    with pdfplumber.open(HUAWEI_PDF) as pdf:
        page = pdf.pages[9]
        chunks = _text_chunks_with_bbox(page, [], page_number=10)

    assert chunks
    first = chunks[0]
    assert first.content.startswith("Message from the Chairman")
    assert "Openness and innovation" in first.content
    assert first.extra_metadata.get("section") == "Message from the Chairman"
    assert "Openness and innovation" in (first.extra_metadata.get("subsection") or "")


@pytest.mark.skipif(not HUAWEI_PDF.exists(), reason="huawei.pdf not found at project root")
def test_huawei_page10_subheading_lines_merge_before_body():
    with pdfplumber.open(HUAWEI_PDF) as pdf:
        words = pdf.pages[9].extract_words(extra_attrs=["size"]) or []
        paragraphs = structured_paragraphs_for_page(words, pdf.pages[9].width)

    assert paragraphs
    first = paragraphs[0]
    assert first.section == "Message from the Chairman"
    assert "Openness and innovation" in (first.subsection or "")
    assert first.words[0]["text"] == "In"


@pytest.mark.skipif(not HUAWEI_PDF.exists(), reason="huawei.pdf not found at project root")
def test_huawei_page13_resets_section_between_column_bands():
    with pdfplumber.open(HUAWEI_PDF) as pdf:
        page = pdf.pages[12]
        words = page.extract_words(extra_attrs=["size"]) or []
        bands = column_bands_for_reading(words, page.width)
        paragraphs = structured_paragraphs_for_page(words, page.width)

    assert len(bands) == 2
    assert paragraphs
    assert " ".join(word["text"] for word in paragraphs[0].words).startswith("Huawei Cloud")
    assert paragraphs[0].section in (None, "")
    assert all("Board's review report" not in " ".join(word["text"] for word in para.words) for para in paragraphs[:2])


def test_two_column_headings_keep_separate_sections_per_band(tmp_path: Path):
    pdf_path = build_two_column_headings_pdf(tmp_path / "two_column_headings.pdf")
    with pdfplumber.open(str(pdf_path)) as pdf:
        page = pdf.pages[0]
        words = page.extract_words(extra_attrs=["size"]) or []
        bands = column_bands_for_reading(words, page.width)
        if len(bands) < 2:
            pytest.skip("Synthetic fixture did not produce a detectable two-column gutter")
        chunks = _text_chunks_with_bbox(page, [], page_number=1)

    sections = [chunk.extra_metadata.get("section") for chunk in chunks]
    assert "LEFT TITLE HEADING" in sections
    assert "RIGHT TITLE HEADING" in sections
    left_chunk = next(chunk for chunk in chunks if chunk.extra_metadata.get("section") == "LEFT TITLE HEADING")
    right_chunk = next(chunk for chunk in chunks if chunk.extra_metadata.get("section") == "RIGHT TITLE HEADING")
    assert "Left column body begins" in left_chunk.content
    assert "Right column body begins" in right_chunk.content
    assert "RIGHT TITLE HEADING" not in left_chunk.content
    assert "LEFT TITLE HEADING" not in right_chunk.content
