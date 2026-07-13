"""Agent tool implementations — thin wrappers over existing retrieval APIs."""

from __future__ import annotations

import json
import logging
from typing import Any

from opensearchpy import OpenSearch

from app.config import settings
from app.ingestion.embeddings import embed_texts
from app.ingestion.xlsx_serialize import format_chunk_content_for_llm
from app.opensearch.documents import get_document_for_user, list_document_records
from app.retrieval.image_attach import retrieve_intent_images
from app.charts.build import attempt_chart_from_chunk
from app.charts.candidates import chart_follow_up_on_priors, rank_chart_table_candidates
from app.retrieval.models import RetrievedChunk
from app.retrieval.service import get_chunk_for_user, hybrid_retrieve
from app.retrieval.query_anchor import merge_retrieval_anchor_phrases
from app.retrieval.scope import limit_xlsx_chunks, resolve_search_top_k, scope_is_pdf_only, scope_is_xlsx_only
from app.retrieval.table_query_signal import should_reserve_table_slots
from app.retrieval.table_slot import merge_with_table_slot
from app.retrieval.xlsx_expand import expand_xlsx_chunks_by_entity_keys

logger = logging.getLogger(__name__)


def _chunk_snippet(chunk: RetrievedChunk) -> str:
    """Compact preview for router tool JSON (not used for grounded answers)."""
    content = format_chunk_content_for_llm(chunk.content, chunk.extra_metadata)
    cap = max(1, settings.agent_tool_snippet_max_chars)
    snippet = content.replace("\n", " ").strip()
    if len(snippet) <= cap:
        return snippet
    return snippet[: cap - 1].rstrip() + "…"


def _chunk_for_tool(chunk: RetrievedChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "filename": chunk.filename,
        "page_number": chunk.page_number,
        "chunk_type": chunk.chunk_type,
        "snippet": _chunk_snippet(chunk),
        "score": round(chunk.score, 4),
    }


def _clamp_agent_top_k(top_k: int | None) -> int | None:
    if top_k is None:
        return None
    return max(1, min(int(top_k), settings.default_top_k))


def execute_search_documents(
    client: OpenSearch,
    *,
    user_id: int,
    query: str,
    top_k: int | None = None,
    scope_doc_ids: list[str] | None = None,
    anchor_fallback_query: str | None = None,
) -> tuple[str, list[RetrievedChunk]]:
    """Run hybrid search; return JSON tool payload and raw chunks for UI post-processing."""
    if scope_doc_ids:
        for scoped_id in scope_doc_ids:
            if get_document_for_user(client, scoped_id, user_id) is None:
                payload = {"error": "document_not_found", "doc_id": scoped_id}
                return json.dumps(payload, ensure_ascii=False), []

    retrieval_query = merge_retrieval_anchor_phrases(
        query,
        fallback_queries=[anchor_fallback_query] if anchor_fallback_query else [],
    )

    k = resolve_search_top_k(
        client,
        user_id=user_id,
        scope_doc_ids=scope_doc_ids,
        top_k=_clamp_agent_top_k(top_k),
        query=retrieval_query,
    )
    response = hybrid_retrieve(
        client,
        retrieval_query,
        user_id=user_id,
        top_k=k,
        doc_ids=scope_doc_ids,
    )
    results = merge_with_table_slot(
        client,
        retrieval_query,
        None,
        response.results,
        user_id=user_id,
        top_k=k,
        doc_ids=scope_doc_ids,
        pdf_scope=should_reserve_table_slots(
            pdf_scope=scope_is_pdf_only(client, user_id=user_id, scope_doc_ids=scope_doc_ids),
            query=retrieval_query,
        ),
    )
    results, anchor_keys = expand_xlsx_chunks_by_entity_keys(
        client,
        results,
        query=retrieval_query,
        user_id=user_id,
        doc_ids=scope_doc_ids,
        anchor_fallback_query=anchor_fallback_query,
    )
    results = limit_xlsx_chunks(
        results,
        limit=settings.excel_top_k,
        anchor_keys=anchor_keys,
    )
    payload = {
        "query": query,
        "total": response.total,
        "chunks": [_chunk_for_tool(chunk) for chunk in results],
    }
    return json.dumps(payload, ensure_ascii=False), results


def execute_list_documents(
    client: OpenSearch,
    *,
    user_id: int,
    scope_doc_ids: list[str] | None = None,
) -> str:
    records = list_document_records(client, user_id)
    if scope_doc_ids is not None:
        allowed = set(scope_doc_ids)
        records = [record for record in records if record.get("doc_id") in allowed]
    documents = [
        {
            "doc_id": record.get("doc_id"),
            "filename": record.get("filename"),
            "ingestion_status": record.get("ingestion_status"),
            "chunk_count": record.get("chunk_count", 0),
            "page_count": record.get("page_count", 0),
        }
        for record in records
    ]
    return json.dumps({"documents": documents, "total": len(documents)}, ensure_ascii=False)


def execute_search_images(
    client: OpenSearch,
    *,
    user_id: int,
    query: str,
    scope_doc_ids: list[str] | None = None,
    top_k: int | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Image-only hybrid search for explicit visual requests."""
    if scope_doc_ids:
        for scoped_id in scope_doc_ids:
            if get_document_for_user(client, scoped_id, user_id) is None:
                return json.dumps({"error": "document_not_found", "doc_id": scoped_id}), []

    k = top_k or settings.image_intent_top_k
    query_vector = embed_texts([query])[0]
    images = retrieve_intent_images(
        client,
        query,
        query_vector,
        user_id=user_id,
        doc_ids=scope_doc_ids,
    )
    if images:
        images = sorted(images, key=lambda i: i["score"], reverse=True)[:1]
    payload = {
        "query": query,
        "total": len(images),
        "images": [
            {
                "image_chunk_id": img["image_chunk_id"],
                "filename": img.get("filename"),
                "page_number": img.get("page_number"),
                "caption": img.get("caption") or "",
                "score": img.get("score"),
            }
            for img in images
        ],
    }
    return json.dumps(payload, ensure_ascii=False), images


def execute_create_chart(
    client: OpenSearch,
    *,
    user_id: int,
    query: str,
    chart_type: str | None = None,
    scope_doc_ids: list[str] | None = None,
    chunk_id: str | None = None,
    prior_table_chunk_ids: list[str] | None = None,
    top_k: int | None = None,
) -> tuple[str, list[dict[str, Any]], list[RetrievedChunk]]:
    """
    Build a bar or line chart from a table chunk via aux LLM + QuickChart.

    Returns JSON tool payload, chart specs (including chart_url), and source table chunk(s).
    """
    if scope_doc_ids:
        for scoped_id in scope_doc_ids:
            if get_document_for_user(client, scoped_id, user_id) is None:
                return json.dumps({"error": "document_not_found", "doc_id": scoped_id}), [], []

    normalized_type = (chart_type or "").strip().lower() or None
    if normalized_type and normalized_type not in {"bar", "line"}:
        return (
            json.dumps(
                {
                    "error": "invalid_chart_type",
                    "message": "chart_type must be bar or line.",
                },
                ensure_ascii=False,
            ),
            [],
            [],
        )

    allowed_docs = set(scope_doc_ids) if scope_doc_ids else None
    prior_ids = {str(chunk_id).strip() for chunk_id in (prior_table_chunk_ids or []) if str(chunk_id).strip()}
    seen_chunk_ids: set[str] = set()
    examined_chunks: list[RetrievedChunk] = []
    errors: list[str] = []

    def _append_candidate(bucket: list[RetrievedChunk], chunk: RetrievedChunk | None) -> None:
        if chunk is None or chunk.chunk_type != "table":
            return
        if allowed_docs is not None and chunk.doc_id not in allowed_docs:
            return
        if chunk.chunk_id in seen_chunk_ids:
            return
        seen_chunk_ids.add(chunk.chunk_id)
        bucket.append(chunk)

    def _try_candidates(candidate_chunks: list[RetrievedChunk]) -> tuple[dict[str, Any] | None, RetrievedChunk | None]:
        ranked = rank_chart_table_candidates(
            candidate_chunks,
            query,
            prior_chunk_ids=prior_ids,
        )
        for chunk in ranked:
            if chunk.chunk_id not in {examined.chunk_id for examined in examined_chunks}:
                examined_chunks.append(chunk)
            chart, error = attempt_chart_from_chunk(
                chunk,
                user_query=query,
                chart_type=normalized_type,  # type: ignore[arg-type]
            )
            if chart is not None:
                return chart, chunk
            if error:
                errors.append(error)
        return None, None

    if chunk_id:
        chunk = get_chunk_for_user(client, chunk_id, user_id)
        if chunk is None:
            return (
                json.dumps({"error": "chunk_not_found", "chunk_id": chunk_id}, ensure_ascii=False),
                [],
                [],
            )
        if allowed_docs is not None and chunk.doc_id not in allowed_docs:
            return (
                json.dumps(
                    {
                        "error": "chunk_not_in_scope",
                        "chunk_id": chunk_id,
                        "scope_doc_ids": scope_doc_ids,
                    },
                    ensure_ascii=False,
                ),
                [],
                [],
            )
        chart, source_chunk = _try_candidates([chunk])
        if chart is not None and source_chunk is not None:
            payload = {
                "status": "created",
                "chart_type": chart["chart_type"],
                "chunk_id": chart["chunk_id"],
                "filename": chart["filename"],
                "page_number": chart["page_number"],
                "period_count": chart.get("period_count"),
                "metric_count": chart.get("metric_count"),
                "message": f"Created {chart['chart_type']} chart from {chart['filename']} (page {chart['page_number']}).",
            }
            return json.dumps(payload, ensure_ascii=False), [chart], [source_chunk]
    else:
        search_query = query.strip()
        if not search_query and not prior_ids:
            return json.dumps({"error": "query is required"}), [], []

        prior_chunks: list[RetrievedChunk] = []
        for prior_id in prior_ids:
            _append_candidate(prior_chunks, get_chunk_for_user(client, prior_id, user_id))

        if prior_chunks:
            chart, source_chunk = _try_candidates(prior_chunks)
            if chart is not None and source_chunk is not None:
                payload = {
                    "status": "created",
                    "chart_type": chart["chart_type"],
                    "chunk_id": chart["chunk_id"],
                    "filename": chart["filename"],
                    "page_number": chart["page_number"],
                    "period_count": chart.get("period_count"),
                    "metric_count": chart.get("metric_count"),
                    "message": f"Created {chart['chart_type']} chart from {chart['filename']} (page {chart['page_number']}).",
                }
                return json.dumps(payload, ensure_ascii=False), [chart], [source_chunk]

        skip_search = chart_follow_up_on_priors(query, prior_chunks)
        search_chunks: list[RetrievedChunk] = []
        if search_query and not skip_search:
            k = resolve_search_top_k(
                client,
                user_id=user_id,
                scope_doc_ids=scope_doc_ids,
                top_k=top_k,
                query=search_query,
            )
            response = hybrid_retrieve(
                client,
                search_query,
                user_id=user_id,
                top_k=k,
                doc_ids=scope_doc_ids,
            )
            table_chunks = sorted(
                (chunk for chunk in response.results if chunk.chunk_type == "table"),
                key=lambda chunk: chunk.score,
                reverse=True,
            )
            if not scope_is_xlsx_only(client, user_id=user_id, scope_doc_ids=scope_doc_ids):
                table_chunks = limit_xlsx_chunks(table_chunks)
            for chunk in table_chunks:
                _append_candidate(search_chunks, chunk)
        elif skip_search:
            logger.info(
                "CREATE_CHART search_skipped reason=chart_follow_up priors=%s query_preview=%r",
                len(prior_chunks),
                query[:120],
            )

        if search_chunks:
            chart, source_chunk = _try_candidates(search_chunks)
            if chart is not None and source_chunk is not None:
                payload = {
                    "status": "created",
                    "chart_type": chart["chart_type"],
                    "chunk_id": chart["chunk_id"],
                    "filename": chart["filename"],
                    "page_number": chart["page_number"],
                    "period_count": chart.get("period_count"),
                    "metric_count": chart.get("metric_count"),
                    "message": f"Created {chart['chart_type']} chart from {chart['filename']} (page {chart['page_number']}).",
                }
                return json.dumps(payload, ensure_ascii=False), [chart], [source_chunk]

    message = errors[0] if errors else "A chart cannot be created for this data."
    return (
        json.dumps(
            {
                "status": "not_chartable",
                "message": message,
                "query": query,
                "tables_examined": len(seen_chunk_ids),
            },
            ensure_ascii=False,
        ),
        [],
        examined_chunks,
    )
