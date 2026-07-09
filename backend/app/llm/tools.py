"""Agent tool implementations — thin wrappers over existing retrieval APIs."""

from __future__ import annotations

import json
from typing import Any

from opensearchpy import OpenSearch

from app.config import settings
from app.ingestion.embeddings import embed_texts
from app.opensearch.documents import get_document_for_user, list_document_records
from app.retrieval.image_attach import retrieve_intent_images
from app.charts.build import attempt_chart_from_chunk
from app.retrieval.models import RetrievedChunk
from app.retrieval.service import get_chunk_for_user, hybrid_retrieve


def _chunk_for_tool(chunk: RetrievedChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "filename": chunk.filename,
        "page_number": chunk.page_number,
        "chunk_type": chunk.chunk_type,
        "content": chunk.content[:2000],
        "score": round(chunk.score, 4),
    }


def execute_search_documents(
    client: OpenSearch,
    *,
    user_id: int,
    query: str,
    top_k: int | None = None,
    doc_id: str | None = None,
    default_doc_id: str | None = None,
) -> tuple[str, list[RetrievedChunk]]:
    """Run hybrid search; return JSON tool payload and raw chunks for UI post-processing."""
    effective_doc_id = doc_id or default_doc_id
    if effective_doc_id is not None:
        owned = get_document_for_user(client, effective_doc_id, user_id)
        if owned is None:
            payload = {"error": "document_not_found", "doc_id": effective_doc_id}
            return json.dumps(payload, ensure_ascii=False), []

    k = top_k or settings.default_top_k
    k = max(1, min(50, int(k)))
    response = hybrid_retrieve(
        client,
        query,
        user_id=user_id,
        top_k=k,
        doc_id=effective_doc_id,
    )
    payload = {
        "query": query,
        "total": response.total,
        "chunks": [_chunk_for_tool(chunk) for chunk in response.results],
    }
    return json.dumps(payload, ensure_ascii=False), response.results


def execute_list_documents(client: OpenSearch, *, user_id: int) -> str:
    records = list_document_records(client, user_id)
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
    doc_id: str | None = None,
    default_doc_id: str | None = None,
    top_k: int | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Image-only hybrid search for explicit visual requests."""
    effective_doc_id = doc_id or default_doc_id
    if effective_doc_id is not None:
        owned = get_document_for_user(client, effective_doc_id, user_id)
        if owned is None:
            return json.dumps({"error": "document_not_found", "doc_id": effective_doc_id}), []

    k = top_k or settings.image_intent_top_k
    query_vector = embed_texts([query])[0]
    images = retrieve_intent_images(
        client,
        query,
        query_vector,
        user_id=user_id,
        doc_id=effective_doc_id,
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
    doc_id: str | None = None,
    default_doc_id: str | None = None,
    chunk_id: str | None = None,
    period_label: str | None = None,
    top_k: int | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Search for table data and build a chart when structurally chartable.

    Returns JSON tool payload and chart specs for the query stream.
    """
    effective_doc_id = doc_id or default_doc_id
    if effective_doc_id is not None:
        owned = get_document_for_user(client, effective_doc_id, user_id)
        if owned is None:
            return json.dumps({"error": "document_not_found", "doc_id": effective_doc_id}), []

    normalized_type = (chart_type or "").strip().lower() or None
    if normalized_type and normalized_type not in {"bar", "line", "pie"}:
        return (
            json.dumps(
                {
                    "error": "invalid_chart_type",
                    "message": "chart_type must be bar, line, or pie.",
                },
                ensure_ascii=False,
            ),
            [],
        )

    candidates: list[RetrievedChunk] = []
    if chunk_id:
        chunk = get_chunk_for_user(client, chunk_id, user_id)
        if chunk is None:
            return (
                json.dumps({"error": "chunk_not_found", "chunk_id": chunk_id}, ensure_ascii=False),
                [],
            )
        if effective_doc_id and chunk.doc_id != effective_doc_id:
            return (
                json.dumps(
                    {"error": "chunk_not_in_scope", "chunk_id": chunk_id, "doc_id": effective_doc_id},
                    ensure_ascii=False,
                ),
                [],
            )
        candidates = [chunk]
    else:
        search_query = query.strip()
        if not search_query:
            return json.dumps({"error": "query is required"}), []
        k = top_k or settings.default_top_k
        k = max(1, min(50, int(k)))
        response = hybrid_retrieve(
            client,
            search_query,
            user_id=user_id,
            top_k=k,
            doc_id=effective_doc_id,
        )
        candidates = [chunk for chunk in response.results if chunk.chunk_type == "table"]

    if not candidates:
        return (
            json.dumps(
                {
                    "status": "not_chartable",
                    "message": "A chart cannot be created for this data: no table was found for the query.",
                    "query": query,
                },
                ensure_ascii=False,
            ),
            [],
        )

    errors: list[str] = []
    for chunk in candidates:
        chart, error = attempt_chart_from_chunk(
            chunk,
            chart_type=normalized_type,  # type: ignore[arg-type]
            period_label=period_label,
        )
        if chart is not None:
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
            return json.dumps(payload, ensure_ascii=False), [chart]

        if error:
            errors.append(error)

    message = errors[0] if errors else "A chart cannot be created for this data."
    return (
        json.dumps(
            {
                "status": "not_chartable",
                "message": message,
                "query": query,
                "tables_examined": len(candidates),
            },
            ensure_ascii=False,
        ),
        [],
    )
