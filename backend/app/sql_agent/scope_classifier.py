"""Classify whether a user query belongs to SQL, RAG, or hybrid."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal

import httpx

from app.config import settings
from app.llm.groq import GROQ_CHAT_COMPLETIONS_URL

logger = logging.getLogger(__name__)

RouteDecision = Literal["sql", "rag", "hybrid"]


@dataclass(frozen=True)
class ScopeClassification:
    decision: RouteDecision
    confidence: float
    matched_tables: list[str]
    reason: str


SCOPE_CLASSIFIER_SYSTEM = """You route user questions between a PostgreSQL database and uploaded documents.

Return JSON only:
{"decision":"sql"|"rag"|"hybrid","confidence":0.0-1.0,"matched_tables":["..."],"reason":"..."}

Rules:
- sql: row lookups, counts, aggregates, filters, joins answerable from the schema tables
- rag: uploaded PDF/DOCX/XLSX content, images, charts, citations, document wording
- hybrid: explicitly compares database facts with uploaded documents
- Entity/title lookups (films, customers, orders, products) are sql when a matching table exists
- Greetings and meta help are rag
- Prefer sql when schema tables clearly cover the question"""


def _heuristic_scope(query: str, tables: list[str]) -> ScopeClassification | None:
    normalized = query.strip().lower()
    if not normalized:
        return ScopeClassification("rag", 0.5, [], "empty query")

    table_map = {table.lower(): table for table in tables}
    matched = []
    for token in re.findall(r"[a-z_][a-z0-9_]*", normalized):
        if token in table_map and table_map[token] not in matched:
            matched.append(table_map[token])

    doc_keywords = ("document", "pdf", "docx", "xlsx", "uploaded", "spreadsheet", "page ", "figure", "image")
    if any(keyword in normalized for keyword in doc_keywords):
        return ScopeClassification("rag", 0.8, [], "document keywords")

    if "compare" in normalized and ("document" in normalized or "upload" in normalized):
        return ScopeClassification("hybrid", 0.75, matched, "explicit compare")

    aggregate_keywords = ("how many", "count", "total", "average", "sum", "top ", "bottom ", "group by")
    if any(keyword in normalized for keyword in aggregate_keywords):
        return ScopeClassification("sql", 0.7, matched, "aggregate pattern")

    entity_keywords = ("film", "movie", "customer", "order", "rental", "payment", "actor", "inventory")
    if any(keyword in normalized for keyword in entity_keywords):
        for keyword in entity_keywords:
            if keyword in normalized and keyword in table_map:
                return ScopeClassification("sql", 0.78, [table_map[keyword]], f"{keyword} table match")

    if matched:
        return ScopeClassification("sql", 0.72, matched, "table name overlap")

    return None


async def classify_query_scope(
    *,
    query: str,
    schema_digest: str,
    tables: list[str],
    display_name: str,
    description: str,
) -> ScopeClassification:
    """Use aux LLM with heuristic fallback to choose sql/rag/hybrid."""
    if not settings.sql_scope_classifier_enabled:
        heuristic = _heuristic_scope(query, tables)
        return heuristic or ScopeClassification("rag", 0.5, [], "classifier disabled")

    if not settings.groq_configured:
        heuristic = _heuristic_scope(query, tables)
        return heuristic or ScopeClassification("rag", 0.5, [], "groq not configured")

    user_prompt = (
        f"Database: {display_name}\n"
        f"Description: {description.strip()}\n"
        f"Tables: {', '.join(tables[:40])}\n\n"
        f"Schema digest:\n{schema_digest}\n\n"
        f"User question: {query.strip()}"
    )
    payload = {
        "model": settings.resolved_sql_scope_classifier_model,
        "messages": [
            {"role": "system", "content": SCOPE_CLASSIFIER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }
    try:
        async with httpx.AsyncClient(timeout=settings.sql_scope_classifier_timeout_seconds) as client:
            response = await client.post(
                GROQ_CHAT_COMPLETIONS_URL,
                headers={
                    "Authorization": f"Bearer {settings.groq_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        decision = str(parsed.get("decision") or "rag").lower()
        if decision not in {"sql", "rag", "hybrid"}:
            decision = "rag"
        confidence = float(parsed.get("confidence") or 0.0)
        matched_tables = [str(name) for name in (parsed.get("matched_tables") or []) if name]
        reason = str(parsed.get("reason") or "llm classification")
        result = ScopeClassification(decision, confidence, matched_tables, reason)
        logger.info(
            "SQL scope_classifier decision=%s confidence=%.2f tables=%s reason=%r",
            result.decision,
            result.confidence,
            result.matched_tables,
            result.reason,
        )
        return result
    except Exception as exc:
        logger.warning("SQL scope_classifier failed: %s", exc)
        heuristic = _heuristic_scope(query, tables)
        return heuristic or ScopeClassification("rag", 0.5, [], "classifier failed")
