"""Post-hoc bbox lookup: locate python-docx chunk text in a rendered viewer PDF.

The rendered PDF frequently interleaves a chunk's words with unrelated text
(floating tables beside paragraphs, multi-column flow), so words extracted in
geometric reading order are NOT contiguous for a given chunk. Matching therefore
uses an order-preserving longest-common-subsequence between the chunk's tokens
and the page's words: this tolerates arbitrary interspersed words while still
requiring the chunk's own tokens to appear in order. The resulting bbox is built
only from the matched words, which naturally outlines the chunk's real region
(e.g. a table's cells) and excludes the interleaved surrounding prose.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pdfplumber

from app.config import settings
from app.ingestion.chunking import normalize_whitespace
from app.ingestion.models import ExtractedChunk

logger = logging.getLogger(__name__)

# Upper bound on tokens compared per chunk, to bound the O(needle*page) matching
# cost. Large enough to cover a full text window / typical table.
MAX_NEEDLE_TOKENS = 300
_SEPARATOR_RE = re.compile(r"^\|[-:|\s]+\|$")
_TOKEN_STRIP_RE = re.compile(r"[^\w]+", re.UNICODE)


def _union_bbox(words: list[dict]) -> list[float]:
    x0 = min(float(w["x0"]) for w in words)
    top = min(float(w["top"]) for w in words)
    x1 = max(float(w["x1"]) for w in words)
    bottom = max(float(w["bottom"]) for w in words)
    return [x0, top, x1, bottom]


def _group_lines(words: list[dict]) -> list[list[dict]]:
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


def _line_bboxes(words: list[dict]) -> list[list[float]]:
    return [_union_bbox(line) for line in _group_lines(words) if line]


def _table_markdown_to_plain(markdown: str) -> str:
    parts: list[str] = []
    for line in markdown.splitlines():
        line = line.strip()
        if not line or _SEPARATOR_RE.match(line):
            continue
        if line.startswith("|") and line.endswith("|"):
            cells = [normalize_whitespace(cell) for cell in line[1:-1].split("|")]
            parts.extend(cell for cell in cells if cell)
        else:
            parts.append(normalize_whitespace(line))
    return normalize_whitespace(" ".join(parts))


def _search_needle(chunk: ExtractedChunk) -> str:
    if chunk.chunk_type == "table":
        return _table_markdown_to_plain(chunk.content)
    return normalize_whitespace(chunk.content)


def _normalize_token(text: str) -> str:
    """Case/punctuation-insensitive token key so quote/dash/format differences
    between python-docx and the LibreOffice render don't break matching."""
    return _TOKEN_STRIP_RE.sub("", text).lower()


def _needle_tokens(chunk: ExtractedChunk) -> list[str]:
    tokens = [_normalize_token(w) for w in _search_needle(chunk).split()]
    tokens = [t for t in tokens if t]
    if len(tokens) <= MAX_NEEDLE_TOKENS:
        return tokens
    return tokens[:MAX_NEEDLE_TOKENS]


def _estimate_start_page(block_index: int, total_blocks: int, total_pages: int) -> int:
    if total_blocks <= 0 or total_pages <= 0:
        return 1
    ratio = max(0.0, min(1.0, (block_index - 1) / total_blocks))
    return max(1, min(total_pages, int(ratio * total_pages) + 1))


def _lcs_matched_words(needle: list[str], page_words: list[dict]) -> list[dict]:
    """Return the page word objects on the longest common (in-order) subsequence
    with ``needle``. Interspersed non-matching page words are skipped freely."""
    page_tokens = [_normalize_token(str(w["text"])) for w in page_words]
    n, m = len(needle), len(page_tokens)
    if n == 0 or m == 0:
        return []

    # dp[i][j] = LCS length of needle[i:] and page_tokens[j:]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        row, next_row = dp[i], dp[i + 1]
        needle_tok = needle[i]
        for j in range(m - 1, -1, -1):
            if needle_tok == page_tokens[j]:
                row[j] = next_row[j + 1] + 1
            else:
                row[j] = next_row[j] if next_row[j] >= row[j + 1] else row[j + 1]

    matched: list[dict] = []
    i = j = 0
    while i < n and j < m:
        if needle[i] == page_tokens[j]:
            matched.append(page_words[j])
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return matched


def _best_match_in_range(
    pdf: pdfplumber.PDF,
    needle: list[str],
    start_page: int,
    end_page: int,
    min_ratio: float,
) -> tuple[int, list[dict], float] | None:
    """Best-covered page in [start_page, end_page]. Fail-closed below min_ratio."""
    if not needle:
        return None

    best: tuple[int, list[dict], float] | None = None
    for page_num in range(start_page, end_page + 1):
        if page_num < 1 or page_num > len(pdf.pages):
            continue
        page_words = pdf.pages[page_num - 1].extract_words() or []
        matched = _lcs_matched_words(needle, page_words)
        if not matched:
            continue
        coverage = len(matched) / len(needle)
        if best is None or coverage > best[2]:
            best = (page_num, matched, coverage)

    if best is None or best[2] < min_ratio:
        return None
    return best


def _set_failed_location(chunk: ExtractedChunk, *, reason: str) -> None:
    extra = dict(chunk.extra_metadata)
    extra["viewer_location"] = {"match_status": "failed"}
    chunk.extra_metadata = extra
    logger.warning(
        "DOCX viewer bbox lookup failed for block_index=%s chunk_type=%s: %s",
        chunk.page_number,
        chunk.chunk_type,
        reason,
    )


def locate_chunks_in_viewer_pdf(
    chunks: list[ExtractedChunk],
    viewer_pdf_path: str | Path,
    *,
    total_blocks: int,
) -> None:
    """Attach viewer_location metadata to each chunk in document order."""
    window = settings.docx_viewer_search_window_pages
    min_ratio = settings.docx_viewer_min_match_ratio
    anchor_page = 1

    with pdfplumber.open(str(viewer_pdf_path)) as pdf:
        total_pages = len(pdf.pages)

        for chunk in chunks:
            needle = _needle_tokens(chunk)
            if not needle:
                _set_failed_location(chunk, reason="empty search needle")
                continue

            search_start = max(1, anchor_page)
            search_end = min(total_pages, anchor_page + window)
            match = _best_match_in_range(pdf, needle, search_start, search_end, min_ratio)

            if match is None:
                estimated = _estimate_start_page(chunk.page_number, total_blocks, total_pages)
                fallback_start = max(1, estimated - window)
                fallback_end = min(total_pages, estimated + window)
                if (fallback_start, fallback_end) != (search_start, search_end):
                    match = _best_match_in_range(
                        pdf, needle, fallback_start, fallback_end, min_ratio
                    )

            if match is None:
                _set_failed_location(chunk, reason="no confident match in search window")
                continue

            page_num, matched_words, coverage = match
            anchor_page = page_num
            extra = dict(chunk.extra_metadata)
            extra["viewer_location"] = {
                "match_status": "ok",
                "viewer_page": page_num,
                "bbox": _union_bbox(matched_words),
                "line_bboxes": _line_bboxes(matched_words),
                "match_ratio": round(coverage, 3),
            }
            chunk.extra_metadata = extra
