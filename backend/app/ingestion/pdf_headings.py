"""Font-size heading detection for PDF text, with per-column section tracking."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from app.ingestion.chunking import normalize_whitespace
from app.ingestion.pdf_layout import column_bands_for_reading

# Vertical gap (as a multiple of median line height) that starts a new paragraph.
PARAGRAPH_GAP_RATIO = 1.35

MIN_BODY_WORDS = 8
MIN_HEADING_WORDS = 2

# Relative to the page/column body font size (median word size).
# 14pt subheading vs 8.5pt body (~1.65x) must stay below the heading cutoff.
HEADING_SIZE_RATIO = 1.85
SUBHEADING_SIZE_RATIO = 1.15

# Headings are short visual lines; long large-type lines are likely titles spanning width.
HEADING_MAX_WORDS = 14
SUBHEADING_MAX_WORDS = 20


@dataclass(frozen=True)
class StructuredParagraph:
    words: list[dict]
    section: str | None = None
    subsection: str | None = None


def _word_size(word: dict) -> float:
    raw = word.get("size")
    if raw is None:
        height = float(word.get("bottom", 0.0)) - float(word.get("top", 0.0))
        return height if height > 0 else 0.0
    return float(raw)


def _body_font_size(words: list[dict]) -> float:
    sizes = [_word_size(word) for word in words if _word_size(word) > 0]
    if not sizes:
        return 10.0
    return float(statistics.median(sizes))


def _line_text(line: list[dict]) -> str:
    return normalize_whitespace(" ".join(str(word.get("text", "")) for word in line))


def _line_median_size(line: list[dict]) -> float:
    sizes = [_word_size(word) for word in line if _word_size(word) > 0]
    return float(statistics.median(sizes)) if sizes else 0.0


def classify_line(line: list[dict], body_size: float) -> str:
    """Classify a line as body, subheading, or heading from relative font size."""
    if not line or body_size <= 0:
        return "body"

    line_size = _line_median_size(line)
    word_count = len(line)
    if line_size >= body_size * HEADING_SIZE_RATIO and word_count <= HEADING_MAX_WORDS:
        return "heading"
    if line_size >= body_size * SUBHEADING_SIZE_RATIO and word_count <= SUBHEADING_MAX_WORDS:
        return "subheading"
    return "body"


def group_lines(words: list[dict]) -> list[list[dict]]:
    """Group words into visual lines, preserving input reading order."""
    lines: list[list[dict]] = []
    current: list[dict] = []
    current_top: float | None = None
    for word in words:
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


def structured_paragraphs_for_band(words: list[dict]) -> list[StructuredParagraph]:
    """Build body paragraphs with section/subsection context for one column band."""
    lines = group_lines(words)
    if not lines:
        return []

    body_size = _body_font_size(words)
    heights = [
        max(float(word["bottom"]) for word in line) - min(float(word["top"]) for word in line)
        for line in lines
    ]
    median_height = statistics.median(heights) if heights else 10.0

    paragraphs: list[StructuredParagraph] = []
    current_words: list[dict] = []
    current_section: str | None = None
    current_subsection: str | None = None
    prev_bottom = max(float(word["bottom"]) for word in lines[0])

    def flush() -> None:
        nonlocal current_words
        if current_words:
            paragraphs.append(
                StructuredParagraph(
                    words=current_words,
                    section=current_section,
                    subsection=current_subsection,
                )
            )
            current_words = []

    for line in lines:
        kind = classify_line(line, body_size)
        line_top = min(float(word["top"]) for word in line)
        line_bottom = max(float(word["bottom"]) for word in line)

        if kind == "heading":
            flush()
            line_text = _line_text(line)
            if current_section and not current_words:
                current_section = f"{current_section} {line_text}"
            else:
                current_section = line_text
            current_subsection = None
            prev_bottom = line_bottom
            continue

        if kind == "subheading":
            flush()
            line_text = _line_text(line)
            if current_subsection:
                current_subsection = f"{current_subsection} {line_text}"
            else:
                current_subsection = line_text
            prev_bottom = line_bottom
            continue

        if current_words and line_top - prev_bottom > median_height * PARAGRAPH_GAP_RATIO:
            flush()

        current_words.extend(line)
        prev_bottom = line_bottom

    flush()
    return paragraphs


def structured_paragraphs_for_page(words: list[dict], page_width: float) -> list[StructuredParagraph]:
    """Detect headings per column band; reset section context at each column boundary."""
    paragraphs: list[StructuredParagraph] = []
    for band in column_bands_for_reading(words, page_width):
        if band:
            paragraphs.extend(structured_paragraphs_for_band(band))
    return paragraphs


def format_chunk_content(section: str | None, subsection: str | None, body: str) -> str:
    """Prefix section headings into chunk content for retrieval and embeddings."""
    parts: list[str] = []
    if section:
        parts.append(section)
    if subsection and subsection != section:
        parts.append(subsection)
    parts.append(body)
    return "\n".join(parts)


def chunk_extra_metadata(
    section: str | None,
    subsection: str | None,
    line_bboxes: list[list[float]],
) -> dict:
    extra: dict = {"line_bboxes": line_bboxes}
    if section:
        extra["section"] = section
    if subsection:
        extra["subsection"] = subsection
    return extra


def min_words_for_paragraph(paragraph: StructuredParagraph) -> int:
    """Body paragraphs need the standard minimum; heading-only bands are already merged."""
    return MIN_BODY_WORDS


def nearest_heading_above_bbox(
    page: object,
    bbox: tuple[float, ...],
    *,
    max_gap: float = 48.0,
    horizontal_pad: float = 36.0,
) -> str | None:
    """Nearest heading/subheading line above a table bbox (generic caption detection)."""
    import pdfplumber

    if not isinstance(page, pdfplumber.page.Page):
        return None

    x0, top, x1, _bottom = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    words = [
        word
        for word in (page.extract_words(extra_attrs=["size"]) or [])
        if float(word["bottom"]) <= top + 4.0
    ]
    if not words:
        return None

    lines = group_lines(words)
    body_size = _body_font_size(words)
    candidates: list[tuple[float, str]] = []

    for line in lines:
        if not line:
            continue
        line_x0 = min(float(word["x0"]) for word in line)
        line_x1 = max(float(word["x1"]) for word in line)
        if line_x1 < x0 - horizontal_pad or line_x0 > x1 + horizontal_pad:
            continue
        kind = classify_line(line, body_size)
        if kind not in {"heading", "subheading"}:
            continue
        line_bottom = max(float(word["bottom"]) for word in line)
        gap = top - line_bottom
        if gap < 0 or gap > max_gap:
            continue
        text = _line_text(line)
        if len(text.split()) < MIN_HEADING_WORDS:
            continue
        candidates.append((gap, text))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]
