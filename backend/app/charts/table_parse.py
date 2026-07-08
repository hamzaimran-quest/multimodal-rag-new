"""Parse markdown tables and numeric cell values."""

from __future__ import annotations

import re

_NUMERIC_VALUE_RE = re.compile(
    r"^\s*\(?[\$€£¥]?\s*([\d,]+(?:\.\d+)?)\s*%?\)?\s*$"
)


def parse_markdown_table(markdown: str) -> list[list[str]]:
    """Parse a markdown table into rows of cell strings."""
    rows: list[list[str]] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if re.match(r"^\|\s*[-:| ]+\|\s*$", stripped):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and any(cell for cell in cells):
            rows.append(cells)
    return rows


def parse_numeric_cell(value: str) -> float | None:
    """Parse a table cell into a float when structurally numeric."""
    text = " ".join(value.split())
    if not text:
        return None

    negative = text.startswith("(") and text.endswith(")")
    match = _NUMERIC_VALUE_RE.match(text)
    if not match:
        return None

    raw = match.group(1).replace(",", "")
    try:
        number = float(raw)
    except ValueError:
        return None
    return -number if negative else number
