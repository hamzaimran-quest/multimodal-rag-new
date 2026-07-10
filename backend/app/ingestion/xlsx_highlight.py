"""Match spreadsheet query sources to a specific sheet row for viewer highlighting."""

from __future__ import annotations

import re

from app.retrieval.models import RetrievedChunk

_STOPWORDS = frozenset({
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "for",
    "from",
    "he",
    "her",
    "him",
    "his",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "have",
    "has",
    "had",
    "information",
    "data",
})

_MIN_MATCH_SCORE = 4
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")


def _query_terms(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return [token for token in tokens if token not in _STOPWORDS and len(token) > 1]


def _answer_signals(answer: str) -> tuple[list[str], list[str], list[str]]:
    terms = _query_terms(answer)
    dates = _DATE_RE.findall(answer)
    numbers = [value for value in _NUMBER_RE.findall(answer) if len(value.replace(",", "")) >= 4]
    return terms, dates, numbers


def _parse_markdown_data_rows(content: str) -> list[str]:
    lines = [line.strip() for line in content.strip().splitlines() if line.strip().startswith("|")]
    if len(lines) < 2:
        return []
    body_lines = lines[2:]
    rows: list[str] = []
    for line in body_lines:
        cells = [cell.strip().lower() for cell in line.strip("|").split("|")]
        rows.append(" ".join(cell for cell in cells if cell))
    return rows


def _parse_data_rows(
    content: str,
    *,
    content_format: str | None = None,
) -> list[str]:
    if content_format == "slim_rows":
        return [line.strip().lower() for line in content.splitlines() if line.strip()]
    return _parse_markdown_data_rows(content)


def _data_sheet_rows(
    data_row_count: int,
    row_range: list[int],
    sheet_row_map: list[int] | None,
) -> list[int]:
    if sheet_row_map and len(sheet_row_map) == data_row_count:
        return sheet_row_map
    if sheet_row_map and len(sheet_row_map) >= data_row_count + 1:
        return sheet_row_map[1 : 1 + data_row_count]
    start = row_range[0]
    return list(range(start + 1, start + 1 + data_row_count))


def _row_match_score(
    row_text: str,
    *,
    query_terms: list[str],
    answer_terms: list[str],
    answer_dates: list[str],
    answer_numbers: list[str],
) -> int:
    if not row_text:
        return 0

    score = 0
    for term in query_terms:
        if term in row_text:
            score += 1
    for term in answer_terms:
        if term in row_text:
            score += 3
    for date in answer_dates:
        if date.lower() in row_text:
            score += 12
    for number in answer_numbers:
        normalized = number.replace(",", "")
        if normalized in row_text.replace(",", ""):
            score += 8
    return score


def match_xlsx_highlight_row_range(
    content: str,
    row_range: list[int] | None,
    sheet_row_map: list[int] | None,
    query: str,
    answer: str,
    *,
    content_format: str | None = None,
) -> list[int] | None:
    """Return a single-row sheet range that best explains the grounded answer."""
    if not row_range or len(row_range) != 2:
        return row_range

    data_rows = _parse_data_rows(content, content_format=content_format)
    if not data_rows:
        return row_range

    full_start, full_end = row_range
    if full_end - full_start <= 2:
        return row_range

    query_terms = _query_terms(query)
    answer_terms, answer_dates, answer_numbers = _answer_signals(answer)
    if not query_terms and not answer_terms and not answer_dates and not answer_numbers:
        return row_range

    sheet_rows = _data_sheet_rows(len(data_rows), row_range, sheet_row_map)
    best_score = 0
    best_sheet_row: int | None = None
    for index, row_text in enumerate(data_rows):
        if index >= len(sheet_rows):
            break
        score = _row_match_score(
            row_text,
            query_terms=query_terms,
            answer_terms=answer_terms,
            answer_dates=answer_dates,
            answer_numbers=answer_numbers,
        )
        if score > best_score:
            best_score = score
            best_sheet_row = sheet_rows[index]

    if best_sheet_row is not None and best_score >= _MIN_MATCH_SCORE:
        return [best_sheet_row, best_sheet_row]

    return row_range


def apply_xlsx_highlights_to_sources(
    sources: list[dict],
    chunks: list[RetrievedChunk],
    *,
    query: str,
    answer: str,
) -> None:
    """Narrow XLSX source row ranges using the final grounded answer."""
    if not answer.strip():
        return

    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    for source in sources:
        if source.get("source_format") != "xlsx":
            continue
        chunk = chunks_by_id.get(source["chunk_id"])
        if chunk is None:
            continue
        extra = chunk.extra_metadata or {}
        source["row_range"] = match_xlsx_highlight_row_range(
            chunk.content,
            extra.get("row_range"),
            extra.get("sheet_row_map"),
            query,
            answer,
            content_format=extra.get("content_format"),
        )
