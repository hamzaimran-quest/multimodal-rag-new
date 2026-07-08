"""Table rows to markdown and shared table helpers."""

from __future__ import annotations

import re

MISALIGNMENT_THRESHOLD = 0.15
LABEL_NONEMPTY_THRESHOLD = 0.9

_NUMERIC_RE = re.compile(r"^[\s\-$€£¥%(),.\d]+$")


def clean_cell(value: object | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def is_numeric_like(value: str) -> bool:
    if not value.strip():
        return False
    return bool(_NUMERIC_RE.match(value.strip()))


_YEAR_RE = re.compile(r"^\d{4}$")


def is_year_like(value: str) -> bool:
    return bool(_YEAR_RE.match(value.strip()))


def is_financial_value(value: str) -> bool:
    if not is_numeric_like(value):
        return False
    return not is_year_like(value)


def column_misalignment_ratio(rows: list[list[object | None]]) -> float:
    """Fraction of data rows whose column count differs from header width."""
    cleaned = [[clean_cell(c) for c in row] for row in rows if any(clean_cell(c) for c in row)]
    if len(cleaned) < 2:
        return 0.0
    header_width = len(cleaned[0])
    data_rows = cleaned[1:]
    if not data_rows:
        return 0.0
    mismatched = sum(1 for row in data_rows if len(row) != header_width)
    return mismatched / len(data_rows)


def validate_reconstructed_table(rows: list[list[object | None]]) -> tuple[bool, dict]:
    cleaned = [[clean_cell(c) for c in row] for row in rows if any(clean_cell(c) for c in row)]
    if len(cleaned) < 2:
        return False, {"reason": "too_few_rows"}

    header = cleaned[0]
    data_rows = cleaned[1:]
    width = len(header)

    misalignment_ratio = column_misalignment_ratio(cleaned)
    if misalignment_ratio > MISALIGNMENT_THRESHOLD:
        return False, {"reason": "misalignment", "misalignment_ratio": misalignment_ratio}

    if not data_rows:
        return False, {"reason": "no_data_rows"}

    label_filled = sum(1 for row in data_rows if row and row[0].strip())
    label_nonempty_rate = label_filled / len(data_rows)
    if label_nonempty_rate < LABEL_NONEMPTY_THRESHOLD:
        return False, {"reason": "label_column_sparse", "label_nonempty_rate": label_nonempty_rate}

    for col in range(1, width):
        all_empty = all(col >= len(row) or not row[col].strip() for row in data_rows)
        header_empty = col >= len(header) or not header[col].strip()
        if all_empty and not header_empty:
            return False, {"reason": "empty_data_column", "column_index": col}

    return True, {
        "misalignment_ratio": misalignment_ratio,
        "label_nonempty_rate": label_nonempty_rate,
    }


# Backward-compatible aliases for internal pdf_tables imports.
_clean_cell = clean_cell
_is_numeric_like = is_numeric_like
_column_misalignment_ratio = column_misalignment_ratio


def table_to_markdown(rows: list[list[object | None]]) -> str:
    """Convert extracted table rows into a markdown table string."""
    if not rows:
        return ""

    cleaned = [[clean_cell(cell) for cell in row] for row in rows if any(clean_cell(c) for c in row)]
    if not cleaned:
        return ""

    width = max(len(row) for row in cleaned)
    normalized: list[list[str]] = []
    for row in cleaned:
        padded = row + [""] * (width - len(row))
        normalized.append(padded[:width])

    header = normalized[0]
    separator = ["---"] * width
    body = normalized[1:] if len(normalized) > 1 else []

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def table_signature(rows: list[list[object | None]]) -> tuple[int, tuple[str, ...]]:
    """Column count + header row for multi-page table continuation detection."""
    cleaned = [[clean_cell(cell) for cell in row] for row in rows if any(clean_cell(c) for c in row)]
    if not cleaned:
        return 0, ()
    header = tuple(cleaned[0])
    return len(header), header
