"""Auto chart creation when the user requests a chart and retrieval includes tables."""

from __future__ import annotations

import json
import logging
from typing import Any

from opensearchpy import OpenSearch

from app.llm.tools import execute_create_chart
from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)

_CHART_REQUEST_MARKERS = (
    "chart",
    "graph",
    "plot",
    "visualize",
    "visualise",
    "draw a",
    "draw the",
)


def chart_requested(query: str) -> bool:
    """True when the user is asking to create or see a chart/graph/plot."""
    lower = query.strip().lower()
    return any(marker in lower for marker in _CHART_REQUEST_MARKERS)


def best_table_chunk(chunks: list[RetrievedChunk]) -> RetrievedChunk | None:
    tables = [chunk for chunk in chunks if chunk.chunk_type == "table"]
    if not tables:
        return None
    return max(tables, key=lambda chunk: chunk.score)


def try_auto_chart_from_retrieval(
    client: OpenSearch,
    *,
    user_id: int,
    user_query: str,
    retrieved_chunks: list[RetrievedChunk],
    scope_doc_ids: list[str] | None = None,
    chart_type: str | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """
    If the user asked for a chart and retrieval includes a table, chart the
    highest-scoring table chunk. Returns (charts, optional note for the answer model).
    """
    if not chart_requested(user_query):
        return [], None

    table = best_table_chunk(retrieved_chunks)
    if table is None:
        return [], (
            "The user asked for a chart, but no table was found in the retrieved "
            "sources. Explain briefly that a chart cannot be created without tabular data."
        )

    payload_json, charts, _source_chunks = execute_create_chart(
        client,
        user_id=user_id,
        query=user_query,
        chart_type=chart_type,
        scope_doc_ids=scope_doc_ids,
        chunk_id=table.chunk_id,
    )
    payload = json.loads(payload_json)

    if charts:
        logger.info(
            "AUTO chart created chunk_id=%s chart_type=%s",
            table.chunk_id,
            charts[0].get("chart_type"),
        )
        return charts, None

    message = str(payload.get("message") or "A chart cannot be created for this data.")
    logger.info("AUTO chart not_chartable chunk_id=%s reason=%s", table.chunk_id, message[:120])
    return [], (
        f"The user asked for a chart. Chart creation was attempted on the retrieved "
        f"table but failed: {message} Explain this briefly in your answer."
    )
