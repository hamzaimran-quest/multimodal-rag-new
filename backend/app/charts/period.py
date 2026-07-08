"""Structural period-label detection (no content-specific literals)."""

from __future__ import annotations

import re

_YEAR_RE = re.compile(r"^(?:FY|CY)?\s*(\d{4})$", re.IGNORECASE)
_QUARTER_RE = re.compile(r"^Q[1-4](?:\s*(?:FY|CY)?\s*\d{2,4})?$", re.IGNORECASE)
_HALF_RE = re.compile(r"^H[12](?:\s*(?:FY|CY)?\s*\d{2,4})?$", re.IGNORECASE)
_PERIOD_INDEX_RE = re.compile(r"^(?:P|T)\d{1,2}$", re.IGNORECASE)
_EMBEDDED_YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")

MAX_PERIOD_LABEL_LEN = 48
MAX_ACRONYM_HEADER_LEN = 3


def _looks_like_financial_number(value: str) -> bool:
    stripped = value.replace(",", "").replace("%", "").strip()
    if not stripped:
        return False
    try:
        float(stripped)
        return True
    except ValueError:
        return False


def _valid_year(year_text: str) -> bool:
    year = int(year_text)
    return 1900 <= year <= 2099


def extract_period_key(label: str) -> str | None:
    """
    Return a canonical period token from a header/label cell, if structurally present.

  Examples matched by structure (not by name):
    - bare years, quarters, halves
    - embedded calendar years inside longer unit-bearing headers
    """
    text = " ".join(label.split())
    if not text or len(text) > MAX_PERIOD_LABEL_LEN:
        return None

    if _YEAR_RE.match(text):
        year = _YEAR_RE.match(text).group(1)  # type: ignore[union-attr]
        return year if _valid_year(year) else None
    if _QUARTER_RE.match(text):
        return text.upper().replace(" ", "")
    if _HALF_RE.match(text):
        return text.upper().replace(" ", "")
    if _PERIOD_INDEX_RE.match(text):
        return text.upper()

  # Compact period tokens with digits + short alpha prefix/suffix (e.g. FY24-style without hardcoding).
    if len(text) <= 6 and not _looks_like_financial_number(text):
        digit_count = sum(ch.isdigit() for ch in text)
        alpha_count = sum(ch.isalpha() for ch in text)
        if digit_count >= 2 and alpha_count <= 4:
            return text.upper()

    years = _EMBEDDED_YEAR_RE.findall(text)
    if len(years) == 1 and _valid_year(years[0]):
        return years[0]

    return None


def is_period_label(value: str) -> bool:
    """Return True when a cell structurally denotes a time period."""
    return extract_period_key(value) is not None


def period_display_label(header: str) -> str:
    """Prefer canonical period key; fall back to trimmed header text."""
    key = extract_period_key(header)
    if key:
        return key
    return " ".join(header.split())


def is_percentage_like_cell(value: str) -> bool:
    text = value.strip()
    if not text or "%" not in text:
        return False
    return parse_numeric_cell_safe(text) is not None


def parse_numeric_cell_safe(value: str) -> float | None:
    from app.charts.table_parse import parse_numeric_cell

    return parse_numeric_cell(value)


def is_annotation_column(header: str, column_values: list[str]) -> bool:
    """
    Detect non-period helper columns structurally (delta/ratio/acronym headers).

    No fixed vocabulary — uses header shape and cell value patterns only.
    """
    if extract_period_key(header):
        return False

    text = " ".join(header.split())
    non_empty_values = [value for value in column_values if value.strip()]

    if not text:
        return len(non_empty_values) == 0

    # Very short alphabetic headers without an embedded period token (e.g. compact ratio labels).
    if len(text) <= MAX_ACRONYM_HEADER_LEN and text.isalpha():
        return True

    if not non_empty_values:
        return True

    percentage_like = sum(1 for value in non_empty_values if is_percentage_like_cell(value))
    if percentage_like / len(non_empty_values) >= 0.75:
        return True

    return False
