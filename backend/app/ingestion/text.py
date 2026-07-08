"""PDF text extraction with table regions excluded."""

from __future__ import annotations

import logging
import statistics

import pdfplumber

from app.config import settings
from app.ingestion.chunking import chunk_text, normalize_whitespace
from app.ingestion.images import extract_image_chunks_for_page
from app.ingestion.models import ExtractedChunk
from app.ingestion.pdf_tables import extract_tables_for_page

logger = logging.getLogger(__name__)

MIN_TEXT_WORDS = 8
# Vertical gap (as a multiple of median line height) that starts a new paragraph.
PARAGRAPH_GAP_RATIO = 1.35


def _words_outside_bboxes(page: pdfplumber.page.Page, bboxes: list[tuple[float, ...]]) -> str:
    if not bboxes:
        text = page.extract_text() or ""
        return normalize_whitespace(text)

    kept_words: list[str] = []
    for word in page.extract_words() or []:
        x0, x1 = word["x0"], word["x1"]
        top, bottom = word["top"], word["bottom"]
        inside_table = any(
            x0 >= bbox[0] and x1 <= bbox[2] and top >= bbox[1] and bottom <= bbox[3]
            for bbox in bboxes
        )
        if not inside_table:
            kept_words.append(word["text"])

    return normalize_whitespace(" ".join(kept_words))


def _kept_words(page: pdfplumber.page.Page, bboxes: list[tuple[float, ...]]) -> list[dict]:
    """Words in reading order that fall outside any detected table bbox."""
    kept: list[dict] = []
    for word in page.extract_words() or []:
        x0, x1 = float(word["x0"]), float(word["x1"])
        top, bottom = float(word["top"]), float(word["bottom"])
        inside_table = any(
            x0 >= bbox[0] and x1 <= bbox[2] and top >= bbox[1] and bottom <= bbox[3]
            for bbox in bboxes
        )
        if not inside_table:
            kept.append(word)
    return kept


def _union_bbox(words: list[dict]) -> list[float]:
    """Bounding box (PDF coordinate space, top-origin) covering all given words."""
    x0 = min(float(w["x0"]) for w in words)
    top = min(float(w["top"]) for w in words)
    x1 = max(float(w["x1"]) for w in words)
    bottom = max(float(w["bottom"]) for w in words)
    return [x0, top, x1, bottom]


def _line_bboxes(words: list[dict]) -> list[list[float]]:
    """Tight per-line bounding boxes for a chunk's words.

    Highlighting each line individually avoids shading the inter-line gaps and
    (critically) the empty gutter between columns that a single union box would
    cover on multi-column pages.
    """
    return [_union_bbox(line) for line in _group_lines(words) if line]


def _group_lines(words: list[dict]) -> list[list[dict]]:
    """Group words into visual lines by their vertical position."""
    ordered = sorted(words, key=lambda w: (round(float(w["top"]), 1), float(w["x0"])))
    lines: list[list[dict]] = []
    current: list[dict] = []
    current_top: float | None = None
    for word in ordered:
        top = float(word["top"])
        height = max(float(word["bottom"]) - top, 1.0)
        if current_top is None or abs(top - current_top) <= height * 0.6:
            current.append(word)
            current_top = top if current_top is None else current_top
        else:
            lines.append(current)
            current = [word]
            current_top = top
    if current:
        lines.append(current)
    return lines


def _group_paragraph_words(words: list[dict]) -> list[list[dict]]:
    """Group words into paragraphs using vertical gaps between lines."""
    lines = _group_lines(words)
    if not lines:
        return []

    heights = [max(float(w["bottom"]) for w in ln) - min(float(w["top"]) for w in ln) for ln in lines]
    median_height = statistics.median(heights) if heights else 10.0

    paragraphs: list[list[dict]] = []
    current: list[dict] = list(lines[0])
    prev_bottom = max(float(w["bottom"]) for w in lines[0])
    for line in lines[1:]:
        line_top = min(float(w["top"]) for w in line)
        if line_top - prev_bottom > median_height * PARAGRAPH_GAP_RATIO:
            paragraphs.append(current)
            current = []
        current.extend(line)
        prev_bottom = max(float(w["bottom"]) for w in line)
    if current:
        paragraphs.append(current)
    return paragraphs


def _window_paragraph_words(words: list[dict], max_words: int, overlap_words: int) -> list[list[dict]]:
    """Overlapping word windows mirroring chunk_text() windowing, but over word objects."""
    if len(words) <= max_words:
        return [words]
    step = max(max_words - overlap_words, 1)
    windows: list[list[dict]] = []
    for start in range(0, len(words), step):
        window = words[start : start + max_words]
        if not window:
            break
        windows.append(window)
        if start + max_words >= len(words):
            break
    return windows


def _text_chunks_with_bbox(page: pdfplumber.page.Page, table_bboxes: list[tuple[float, ...]], page_number: int) -> list[ExtractedChunk]:
    """Produce text chunks with a union bbox per chunk, using word-level coordinates."""
    words = _kept_words(page, table_bboxes)
    if not words:
        # Fallback: no word geometry available; emit text chunks without bbox.
        body_text = _words_outside_bboxes(page, table_bboxes)
        chunks: list[ExtractedChunk] = []
        for paragraph in _paragraphs_from_text(body_text):
            for piece in chunk_text(paragraph):
                if len(piece.split()) < MIN_TEXT_WORDS:
                    continue
                chunks.append(
                    ExtractedChunk(
                        content=piece,
                        page_number=page_number,
                        chunk_type="text",
                        extraction_method="pdfplumber",
                    )
                )
        return chunks

    max_words = settings.chunk_max_words
    overlap_words = settings.chunk_overlap_words
    chunks = []
    for paragraph_words in _group_paragraph_words(words):
        if len(paragraph_words) < MIN_TEXT_WORDS:
            continue
        for window in _window_paragraph_words(paragraph_words, max_words, overlap_words):
            content = normalize_whitespace(" ".join(str(w["text"]) for w in window))
            if len(content.split()) < MIN_TEXT_WORDS:
                continue
            chunks.append(
                ExtractedChunk(
                    content=content,
                    page_number=page_number,
                    chunk_type="text",
                    extraction_method="pdfplumber",
                    bbox=_union_bbox(window),
                    extra_metadata={"line_bboxes": _line_bboxes(window)},
                )
            )
    return chunks


def _paragraphs_from_text(text: str) -> list[str]:
    raw_parts = [normalize_whitespace(p) for p in text.split("\n")]
    paragraphs: list[str] = []
    buffer: list[str] = []

    for part in raw_parts:
        if not part:
            if buffer:
                paragraphs.append(" ".join(buffer))
                buffer = []
            continue
        buffer.append(part)

    if buffer:
        paragraphs.append(" ".join(buffer))

    if not paragraphs and text:
        paragraphs = [text]

    return [p for p in paragraphs if len(p.split()) >= MIN_TEXT_WORDS]


def extract_page_chunks(
    page: pdfplumber.page.Page,
    page_number: int,
    pdf_path: str,
    *,
    doc_id: str | None = None,
    user_id: int | None = None,
) -> list[ExtractedChunk]:
    chunks: list[ExtractedChunk] = []

    table_chunks, table_bboxes = extract_tables_for_page(page, page_number, pdf_path)
    chunks.extend(table_chunks)

    chunks.extend(_text_chunks_with_bbox(page, table_bboxes, page_number))

    chunks.extend(
        extract_image_chunks_for_page(
            page,
            page_number=page_number,
            doc_id=doc_id,
            user_id=user_id,
            excluded_bboxes=table_bboxes,
        )
    )

    return chunks


def extract_pdf_text_and_tables(pdf_path: str, *, doc_id: str | None = None) -> list[ExtractedChunk]:
    all_chunks: list[ExtractedChunk] = []
    with pdfplumber.open(pdf_path) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            try:
                all_chunks.extend(extract_page_chunks(page, index, pdf_path, doc_id=doc_id))
            except Exception:
                logger.exception("Failed extracting page %s from %s", index, pdf_path)
    return all_chunks
