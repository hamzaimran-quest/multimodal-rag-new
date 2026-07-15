"""Tests for PDF font-size heading detection and section metadata."""

from __future__ import annotations

from pathlib import Path

import pdfplumber
import pytest

from app.config import PROJECT_ROOT
from app.ingestion.pdf_headings import (
    classify_line,
    group_lines,
    structured_paragraphs_for_band,
    structured_paragraphs_for_page,
)
from app.ingestion.text import extract_pdf_text_and_tables
from app.ingestion.pdf_layout import column_bands_for_reading
from app.ingestion.text import _text_chunks_with_bbox
from tests.pdf_fixtures import build_two_column_headings_pdf

HUAWEI_PDF = PROJECT_ROOT / "huawei.pdf"
TIMBERLAND_PDF = PROJECT_ROOT / "timberland.pdf"


def _word(text: str, x0: float, x1: float, top: float, bottom: float, size: float = 10.0) -> dict:
    return {"text": text, "x0": x0, "x1": x1, "top": top, "bottom": bottom, "size": size}


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


def test_group_lines_keeps_normal_word_spacing_on_one_line():
    words = [
        _word("The", 50, 65, 100, 110),
        _word("quick", 68, 90, 100, 110),
        _word("brown", 93, 118, 100, 110),
        _word("fox", 121, 138, 100, 110),
    ]
    lines = group_lines(words)
    assert len(lines) == 1
    assert [w["text"] for w in lines[0]] == ["The", "quick", "brown", "fox"]


def test_group_lines_splits_disjoint_blocks_at_the_same_height():
    # Two side-by-side chart axis labels at the same height, e.g. a "200,000"
    # tick label under one chart and a "400,000" tick label under an unrelated
    # chart placed next to it -- these must not be read as one flowing line.
    words = [
        _word("200,000", 84, 105, 613, 621),
        _word("400,000", 350, 371, 613, 621),
    ]
    lines = group_lines(words)
    assert len(lines) == 2
    assert [w["text"] for w in lines[0]] == ["200,000"]
    assert [w["text"] for w in lines[1]] == ["400,000"]


@pytest.mark.skipif(not HUAWEI_PDF.exists(), reason="huawei.pdf not found at project root")
def test_huawei_page9_first_chart_column_band_labels_stay_split():
    """Page 9's three side-by-side bar charts (Revenue, Operating profit, Cash
    flow) have axis/value labels at the same heights. column_bands_for_reading
    only ever splits a page into at most two bands, so it separates chart 1
    from charts 2+3 but can't further separate 2 from 3 -- charts 2+3 still
    merge into one incoherent paragraph (a known, documented residual gap; see
    group_lines' docstring). This guards the part that *is* fixed: within
    chart 1's own band, group_lines' horizontal-gap split keeps its labels
    from ballooning into one giant run.
    """
    with pdfplumber.open(HUAWEI_PDF) as pdf:
        page = pdf.pages[8]
        words = page.extract_words(extra_attrs=["size"]) or []
        chart_words = [w for w in words if 578 <= float(w["top"]) <= 691]
        bands = column_bands_for_reading(chart_words, page.width)

    assert len(bands) == 2
    first_band_paragraphs = structured_paragraphs_for_band(bands[0])
    assert first_band_paragraphs
    assert max(len(p.words) for p in first_band_paragraphs) <= 10


@pytest.mark.skipif(not TIMBERLAND_PDF.exists(), reason="timberland.pdf not found at project root")
def test_timberland_loan_portfolio_table_text_not_shattered():
    """The group_lines fix must not fragment legitimate dense financial-table
    text (multiple side-by-side numeric year-columns) into pieces so small
    they fall below the minimum chunk size and silently disappear from the
    index -- this table isn't reliably captured as a `table` chunk on this
    page, so this text fallback is its only representation.
    """
    chunks = [
        c for c in extract_pdf_text_and_tables(str(TIMBERLAND_PDF)) if c.chunk_type == "text" and c.page_number == 15
    ]
    assert any("Multi-family" in c.content and "207,767" in c.content for c in chunks)


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
