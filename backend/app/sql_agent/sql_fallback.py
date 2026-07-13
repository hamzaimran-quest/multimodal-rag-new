"""Lightweight heuristics for SQL tool fallback when the router skips tools."""

from __future__ import annotations

import re
from typing import Literal

SqlFallbackDecision = Literal["sql", "hybrid"]


def looks_like_sql_query(query: str, tables: list[str]) -> SqlFallbackDecision | None:
    """Return sql/hybrid when the message plausibly needs the database tool."""
    normalized = query.strip().lower()
    if not normalized:
        return None

    table_map = {table.lower(): table for table in tables}
    matched: list[str] = []
    for token in re.findall(r"[a-z_][a-z0-9_]*", normalized):
        if token in table_map and table_map[token] not in matched:
            matched.append(table_map[token])

    doc_keywords = (
        "document",
        "pdf",
        "docx",
        "xlsx",
        "uploaded",
        "spreadsheet",
        "page ",
        "figure",
        "image",
    )
    has_doc_signal = any(keyword in normalized for keyword in doc_keywords)

    if "compare" in normalized and has_doc_signal:
        return "hybrid"

    if has_doc_signal:
        return None

    aggregate_keywords = ("how many", "count", "total", "average", "sum", "top ", "bottom ", "group by")
    if any(keyword in normalized for keyword in aggregate_keywords):
        return "sql"

    entity_keywords = ("film", "movie", "customer", "order", "rental", "payment", "actor", "inventory")
    for keyword in entity_keywords:
        if keyword in normalized and keyword in table_map:
            return "sql"

    if matched:
        return "sql"

    return None
