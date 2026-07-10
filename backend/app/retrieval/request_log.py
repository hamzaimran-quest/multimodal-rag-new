"""Structured logging for retrieval and chart-eligibility per API request."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.charts.profile import analyze_table_chartability
from app.charts.spec import validate_and_build_chart_spec
from app.ingestion.xlsx_serialize import table_rows_from_chunk_content
from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)

_QUERY_PREVIEW_LIMIT = 120

# A markdown table separator cell, e.g. "| --- |".
_SEPARATOR_CELL_RE = re.compile(r"\|\s*:?-{3,}:?\s*\|")
# A line that is *only* a valid table separator row (pipes, dashes, colons, spaces).
_PURE_SEPARATOR_LINE_RE = re.compile(r"^\|?[\s:\-|]+\|?$")


def _chunk_type_counts(chunks: list[RetrievedChunk]) -> dict[str, int]:
    counts: dict[str, int] = {"text": 0, "table": 0, "image": 0, "other": 0}
    for chunk in chunks:
        key = chunk.chunk_type if chunk.chunk_type in counts else "other"
        counts[key] += 1
    return counts


def _summarize_chunk(chunk: RetrievedChunk, *, chart_eligibility: dict[str, Any] | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "filename": chunk.filename,
        "page_number": chunk.page_number,
        "chunk_type": chunk.chunk_type,
        "score": round(chunk.score, 4),
        "extraction_method": chunk.extraction_method,
        "has_image_url": bool(chunk.image_url),
        "content_chars": len(chunk.content),
    }
    if chart_eligibility is not None:
        summary["chart"] = chart_eligibility
    return summary


def build_chart_eligibility_records(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    """Per retrieved table chunk: runtime structural chartability outcome."""
    records: list[dict[str, Any]] = []
    for chunk in chunks:
        if chunk.chunk_type != "table":
            continue

        rows = table_rows_from_chunk_content(chunk.content, chunk.extra_metadata or {})
        profile = analyze_table_chartability(rows) if rows else None
        runtime_chartable = bool(profile and profile.get("chartable"))

        record: dict[str, Any] = {
            "chunk_id": chunk.chunk_id,
            "filename": chunk.filename,
            "page_number": chunk.page_number,
            "runtime_chartable": runtime_chartable,
            "chart_profile": profile,
            "chart_offered": False,
            "validation_outcome": "not_chartable_at_runtime",
        }

        if not profile or not runtime_chartable:
            records.append(record)
            continue

        spec = validate_and_build_chart_spec(chunk.content, profile)
        if spec is None:
            record["validation_outcome"] = "validation_failed"
            records.append(record)
            continue

        record["chart_offered"] = True
        record["validation_outcome"] = "offered"
        record["chart_type"] = spec.get("chart_type")
        record["period_count"] = spec.get("period_count")
        record["metric_count"] = spec.get("metric_count")
        record["orientation"] = spec.get("orientation")
        records.append(record)

    return records


def build_request_summary(
    *,
    endpoint: str,
    query: str,
    top_k: int,
    doc_id: str | None,
    chunks: list[RetrievedChunk],
    charts: list[dict[str, Any]],
    chart_eligibility: list[dict[str, Any]],
) -> dict[str, Any]:
    eligibility_by_chunk = {row["chunk_id"]: row for row in chart_eligibility}
    chunk_summaries = [
        _summarize_chunk(
            chunk,
            chart_eligibility=eligibility_by_chunk.get(chunk.chunk_id),
        )
        for chunk in chunks
    ]

    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    runtime_chartable = sum(1 for row in chart_eligibility if row.get("runtime_chartable"))
    charts_offered = sum(1 for row in chart_eligibility if row.get("chart_offered"))

    return {
        "endpoint": endpoint,
        "query_preview": query[:_QUERY_PREVIEW_LIMIT],
        "query_chars": len(query),
        "top_k": top_k,
        "doc_id_filter": doc_id,
        "retrieved_total": len(chunks),
        "chunk_type_counts": _chunk_type_counts(chunks),
        "has_image_chunk": any(c.chunk_type == "image" for c in chunks),
        "table_chunks_retrieved": len(table_chunks),
        "table_chunks_runtime_chartable": runtime_chartable,
        "charts_offered": charts_offered,
        "tool_charts_emitted": len(charts),
        "chunks": chunk_summaries,
        "chart_eligibility": chart_eligibility,
    }


def log_retrieval_request(
    *,
    endpoint: str,
    query: str,
    top_k: int,
    doc_id: str | None,
    chunks: list[RetrievedChunk],
    charts: list[dict[str, Any]] | None = None,
) -> None:
    """Log one structured record per request after retrieval (and chart evaluation when applicable)."""
    chart_eligibility = build_chart_eligibility_records(chunks)
    charts = charts if charts is not None else []
    summary = build_request_summary(
        endpoint=endpoint,
        query=query,
        top_k=top_k,
        doc_id=doc_id,
        chunks=chunks,
        charts=charts,
        chart_eligibility=chart_eligibility,
    )

    logger.info(
        "RETRIEVAL_REQUEST endpoint=%s retrieved=%s table=%s runtime_chartable=%s charts_offered=%s "
        "types=%s doc_filter=%s query_chars=%s",
        endpoint,
        summary["retrieved_total"],
        summary["table_chunks_retrieved"],
        summary["table_chunks_runtime_chartable"],
        summary["charts_offered"],
        summary["chunk_type_counts"],
        doc_id,
        summary["query_chars"],
    )
    logger.info("RETRIEVAL_REQUEST_DETAIL %s", json.dumps(summary, ensure_ascii=False, default=str))


def detect_inline_table_markdown(text: str) -> bool:
    """Heuristic: True when table pipe-syntax is jammed into a prose/bullet line.

    A well-formed Markdown table has its separator row (``| --- |``) on its own
    line. When the model inlines a table into a sentence or bullet, the separator
    cell appears mixed with other text on a single line and fails to render.
    """
    for raw_line in text.splitlines():
        if "|" not in raw_line:
            continue
        if not _SEPARATOR_CELL_RE.search(raw_line):
            continue
        if not _PURE_SEPARATOR_LINE_RE.match(raw_line.strip()):
            return True
    return False


def log_llm_context(*, query: str, context: str, source_count: int) -> None:
    """Log the exact context assembled for the LLM, for answer-quality debugging."""
    logger.info(
        "LLM_CONTEXT query_preview=%r sources=%s context_chars=%s",
        query[:_QUERY_PREVIEW_LIMIT],
        source_count,
        len(context),
    )
    logger.info(
        "LLM_CONTEXT_DETAIL %s",
        json.dumps({"query": query, "context": context}, ensure_ascii=False, default=str),
    )


def log_llm_answer(*, query: str, answer: str) -> None:
    """Log the final generated answer and flag suspected inline-table formatting bugs."""
    inline_table = detect_inline_table_markdown(answer)
    logger.info(
        "LLM_ANSWER query_preview=%r answer_chars=%s inline_table_detected=%s",
        query[:_QUERY_PREVIEW_LIMIT],
        len(answer),
        inline_table,
    )
    logger.info(
        "LLM_ANSWER_DETAIL %s",
        json.dumps(
            {"query": query, "answer": answer, "inline_table_detected": inline_table},
            ensure_ascii=False,
            default=str,
        ),
    )
    if inline_table:
        logger.warning(
            "LLM_ANSWER_INLINE_TABLE query_preview=%r — model emitted table pipe-syntax "
            "inside prose/bullets; it will not render as a table",
            query[:_QUERY_PREVIEW_LIMIT],
        )


def log_query_stream_outcome(*, query: str, ok: bool, error: str | None = None) -> None:
    preview = query[:_QUERY_PREVIEW_LIMIT]
    if ok:
        logger.info("QUERY_STREAM_DONE ok=true query_preview=%r", preview)
    else:
        logger.warning("QUERY_STREAM_DONE ok=false query_preview=%r error=%s", preview, error)
