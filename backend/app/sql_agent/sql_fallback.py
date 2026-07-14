"""Lightweight heuristics for SQL tool fallback when the router skips tools."""

from __future__ import annotations

import re
from typing import Literal

SqlFallbackDecision = Literal["sql", "hybrid"]

_MULTI_INTENT_RE = re.compile(
    r"\b(?:and also|as well as|and tell me|and what|plus)\b",
    re.IGNORECASE,
)
_METRIC_CUE_RE = re.compile(
    r"\b(?:revenue|growth|segment|segments|cagr|profit|total|count|average|sum|"
    r"how many)\b",
    re.IGNORECASE,
)
_NARRATIVE_CUE_RE = re.compile(
    r"\b(?:chairwoman|chairman|statement|stated|says|"
    r"message from|in the document|in the pdf|in the text|"
    r"from the document|from the pdf|from the text|"
    r"document|pdf|uploaded|docx|xlsx|spreadsheet|figure|image)\b",
    re.IGNORECASE,
)


def looks_like_hybrid_question(query: str) -> bool:
    """True when the ask needs both database facts and document/narrative content."""
    normalized = (query or "").strip().lower()
    if not normalized:
        return False

    has_metric = bool(_METRIC_CUE_RE.search(normalized))
    has_narrative = bool(_NARRATIVE_CUE_RE.search(normalized))
    has_multi = bool(_MULTI_INTENT_RE.search(normalized))

    if has_metric and has_narrative:
        return True
    if has_multi and has_narrative:
        return True
    if "compare" in normalized and has_narrative:
        return True
    return False


def looks_like_sql_query(query: str, tables: list[str]) -> SqlFallbackDecision | None:
    """Return sql/hybrid when the message plausibly needs the database tool."""
    normalized = query.strip().lower()
    if not normalized:
        return None

    if looks_like_hybrid_question(normalized):
        return "hybrid"

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
