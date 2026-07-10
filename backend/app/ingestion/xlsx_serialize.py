"""Compact XLSX row serialization for indexed chunks (headers stored in metadata)."""

from __future__ import annotations

from app.config import settings


def resolve_row_band_size(column_count: int) -> int:
    """Pick a smaller band for wide sheets to keep chunks token-efficient."""
    if column_count >= settings.excel_wide_column_threshold:
        return max(1, settings.excel_row_band_size_wide)
    if column_count >= settings.excel_medium_column_threshold:
        return max(1, settings.excel_row_band_size_medium)
    return max(1, settings.excel_row_band_size)


def rows_to_slim_values_text(data_rows: list[list[str]]) -> str:
    """Serialize data rows as one pipe-delimited values line per row."""
    lines: list[str] = []
    for row in data_rows:
        cells = [cell.strip() for cell in row]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def slim_values_text_to_rows(headers: list[str], content: str) -> list[list[str]]:
    """Rebuild a header + data table from slim values text and stored headers."""
    if not headers:
        return []
    rows: list[list[str]] = [list(headers)]
    width = len(headers)
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        values = [value.strip() for value in stripped.split(" | ")]
        padded = values + [""] * max(0, width - len(values))
        rows.append(padded[:width])
    return rows


def table_rows_from_chunk_content(
    content: str,
    extra_metadata: dict | None,
) -> list[list[str]]:
    """Parse table rows from chunk content (markdown or slim row bands)."""
    extra = extra_metadata or {}
    headers = extra.get("table_headers")
    if extra.get("content_format") == "slim_rows" and headers:
        return slim_values_text_to_rows(list(headers), content)

    from app.charts.table_parse import parse_markdown_table

    return parse_markdown_table(content)


def format_chunk_content_for_llm(content: str, extra_metadata: dict | None) -> str:
    """Attach stored column headers to slim XLSX row bands for LLM context."""
    extra = extra_metadata or {}
    headers = extra.get("table_headers")
    body = content.strip()
    if extra.get("content_format") == "slim_rows" and headers:
        header_line = " | ".join(str(header).strip() for header in headers)
        if not body:
            return f"Column headers:\n{header_line}"
        return f"Column headers:\n{header_line}\n\n{body}"
    return body
