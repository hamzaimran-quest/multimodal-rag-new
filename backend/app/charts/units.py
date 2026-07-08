"""Structural value-axis unit detection from table header cells."""

from __future__ import annotations

import re
from collections import Counter

_PAREN_RE = re.compile(r"\(([^)]+)\)")
_SCALE_WORD_RE = re.compile(r"\b(million|billion|thousand|trillion|percent)\b", re.IGNORECASE)
_CURRENCY_CODE_RE = re.compile(r"^[A-Z]{3}$")
_PERCENT_HINT_RE = re.compile(r"%|percent", re.IGNORECASE)

DEFAULT_VALUE_AXIS_LABEL = "Value"


def _normalize_token(token: str) -> str:
    return " ".join(token.split())


def _looks_like_unit_token(token: str) -> bool:
    text = _normalize_token(token)
    if not text or len(text) > 48:
        return False
    if _PERCENT_HINT_RE.search(text):
        return True

    parts = text.split()
    if len(parts) == 1 and _CURRENCY_CODE_RE.match(parts[0]):
        return True
    if _SCALE_WORD_RE.search(text):
        return True
    if (
        len(parts) >= 2
        and parts[0].isalpha()
        and 2 <= len(parts[0]) <= 4
        and parts[0].isupper()
        and _SCALE_WORD_RE.search(" ".join(parts[1:]))
    ):
        return True
    return False


def _tokens_from_cell(cell: str) -> list[str]:
    return [_normalize_token(match) for match in _PAREN_RE.findall(cell) if match.strip()]


def _unit_candidates_from_cells(cells: list[str]) -> list[str]:
    candidates: list[str] = []
    for cell in cells:
        for token in _tokens_from_cell(cell):
            if _looks_like_unit_token(token):
                candidates.append(token)
    return candidates


def detect_value_axis_label(
    rows: list[list[str]],
    *,
    orientation: str,
    period_column_indices: list[int] | None = None,
) -> str:
    """
    Infer a Y-axis label from parenthetical unit markers in header cells.

    Uses only structural patterns (scale words, currency codes, percent markers).
  Fails closed to "Value" when markers are missing or contradictory.
    """
    if not rows:
        return DEFAULT_VALUE_AXIS_LABEL

    header = rows[0]
    candidates: list[str] = []

    if orientation == "wide":
        indices = period_column_indices if period_column_indices else list(range(1, len(header)))
        header_cells = [header[idx] for idx in indices if idx < len(header)]
        if header:
            header_cells.append(header[0])
        candidates = _unit_candidates_from_cells(header_cells)
    else:
        candidates = _unit_candidates_from_cells(header)

    if not candidates:
        return DEFAULT_VALUE_AXIS_LABEL

    normalized = [candidate.casefold() for candidate in candidates]
    counts = Counter(normalized)
    best_key, frequency = counts.most_common(1)[0]

    if len(counts) > 1 and frequency < max(2, len(candidates) // 2):
        return DEFAULT_VALUE_AXIS_LABEL

    for candidate in candidates:
        if candidate.casefold() == best_key:
            return candidate

    return DEFAULT_VALUE_AXIS_LABEL
