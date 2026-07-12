"""Tests for generic PDF column layout and reading-order detection."""

from __future__ import annotations

from pathlib import Path

import pdfplumber
import pytest

from app.config import PROJECT_ROOT
from app.ingestion.pdf_layout import detect_column_gutter, reorder_words_for_reading
from app.ingestion.text import _text_chunks_with_bbox
from tests.pdf_fixtures import build_sample_pdf, build_two_column_pdf

HUAWEI_PDF = PROJECT_ROOT / "huawei.pdf"
TIMBERLAND_PDF = PROJECT_ROOT / "timberland.pdf"


def _row_major_order(words: list[dict]) -> list[dict]:
    return sorted(words, key=lambda w: (round(float(w["top"]), 1), float(w["x0"])))


def _joined(words: list[dict], limit: int = 40) -> str:
    return " ".join(str(word["text"]) for word in words[:limit])


def test_detect_column_gutter_on_synthetic_two_column_pdf(tmp_path: Path):
    pdf_path = build_two_column_pdf(tmp_path / "two_column.pdf")
    with pdfplumber.open(str(pdf_path)) as pdf:
        page = pdf.pages[0]
        words = page.extract_words() or []

    assert detect_column_gutter(words, page.width) is not None


def test_reorder_words_reads_each_column_top_down(tmp_path: Path):
    pdf_path = build_two_column_pdf(tmp_path / "two_column.pdf")
    with pdfplumber.open(str(pdf_path)) as pdf:
        page = pdf.pages[0]
        words = page.extract_words() or []
        ordered = reorder_words_for_reading(words, page.width)

    text = _joined(ordered, 80)
    left_index = text.index("LEFTCOLUMN_START")
    right_index = text.index("RIGHTCOLUMN_START")
    assert left_index < right_index
    assert "END_LEFT" in text
    assert text.index("END_LEFT") < right_index


def test_single_column_pdf_unchanged_reading_order(tmp_path: Path):
    pdf_path = build_sample_pdf(tmp_path / "single_column.pdf")
    with pdfplumber.open(str(pdf_path)) as pdf:
        page = pdf.pages[0]
        words = page.extract_words() or []

    assert detect_column_gutter(words, page.width) is None
    assert _joined(reorder_words_for_reading(words, page.width)) == _joined(
        _row_major_order(words)
    )


@pytest.mark.skipif(not HUAWEI_PDF.exists(), reason="huawei.pdf not found at project root")
def test_huawei_page13_avoids_row_wise_column_interleaving():
    with pdfplumber.open(HUAWEI_PDF) as pdf:
        page = pdf.pages[12]
        words = page.extract_words() or []
        ordered = reorder_words_for_reading(words, page.width)

    text = _joined(ordered, 30)
    assert text.startswith("Huawei Cloud")
    assert "Board's review report" not in text.split()[:12]


@pytest.mark.skipif(not HUAWEI_PDF.exists(), reason="huawei.pdf not found at project root")
def test_huawei_page13_text_chunks_use_column_reading_order():
    with pdfplumber.open(HUAWEI_PDF) as pdf:
        page = pdf.pages[12]
        chunks = _text_chunks_with_bbox(page, [], page_number=13)

    assert chunks
    first = chunks[0].content
    assert "Huawei Cloud" in first
    assert "Board's review report" not in first.split()[:20]


@pytest.mark.skipif(not TIMBERLAND_PDF.exists(), reason="timberland.pdf not found at project root")
def test_timberland_page5_avoids_row_wise_column_interleaving():
    with pdfplumber.open(TIMBERLAND_PDF) as pdf:
        page = pdf.pages[4]
        words = page.extract_words() or []
        ordered = reorder_words_for_reading(words, page.width)

    text = _joined(ordered, 20)
    assert text.startswith("2025 FORM")
    assert "We have included our Form 10-K" in text
    assert "2025 FORM 10-K We have included" not in text


@pytest.mark.skipif(not TIMBERLAND_PDF.exists(), reason="timberland.pdf not found at project root")
def test_timberland_page3_single_column_letter_still_reads_naturally():
    with pdfplumber.open(TIMBERLAND_PDF) as pdf:
        page = pdf.pages[2]
        words = page.extract_words() or []
        ordered = reorder_words_for_reading(words, page.width)

    text = _joined(ordered, 18)
    assert text.startswith("Dear Fellow Shareholders")


@pytest.mark.skipif(not TIMBERLAND_PDF.exists(), reason="timberland.pdf not found at project root")
def test_timberland_ingestion_produces_text_chunks():
    with pdfplumber.open(TIMBERLAND_PDF) as pdf:
        all_chunks = []
        for index, page in enumerate(pdf.pages, start=1):
            all_chunks.extend(_text_chunks_with_bbox(page, [], page_number=index))

    assert len(all_chunks) > 10
    assert any("Shareholders" in chunk.content for chunk in all_chunks)
