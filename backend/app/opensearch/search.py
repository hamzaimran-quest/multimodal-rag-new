"""Search helpers for k-NN and hybrid retrieval."""

from typing import Any

from opensearchpy import OpenSearch

from app.config import settings


def _build_filters(
    user_id: int,
    doc_id: str | None = None,
    doc_ids: list[str] | None = None,
    chunk_type: str | None = None,
) -> dict[str, Any]:
    filters: list[dict[str, Any]] = [{"term": {"user_id": str(user_id)}}]
    scope_ids = list(doc_ids) if doc_ids else ([doc_id] if doc_id else None)
    if scope_ids:
        if len(scope_ids) == 1:
            filters.append({"term": {"doc_id": scope_ids[0]}})
        else:
            filters.append({"terms": {"doc_id": scope_ids}})
    if chunk_type:
        filters.append({"term": {"chunk_type": chunk_type}})
    return {"bool": {"filter": filters}}


def knn_search(
    client: OpenSearch,
    query_vector: list[float],
    k: int = 5,
    user_id: int | None = None,
    doc_id: str | None = None,
    doc_ids: list[str] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "size": k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": k,
                }
            }
        },
    }
    if user_id is not None:
        body["post_filter"] = _build_filters(user_id, doc_id=doc_id, doc_ids=doc_ids)
    elif doc_ids or doc_id:
        scope_ids = list(doc_ids) if doc_ids else ([doc_id] if doc_id else None)
        if scope_ids:
            if len(scope_ids) == 1:
                body["post_filter"] = {"term": {"doc_id": scope_ids[0]}}
            else:
                body["post_filter"] = {"terms": {"doc_id": scope_ids}}

    return client.search(index=settings.chunks_index, body=body)


def hybrid_search(
    client: OpenSearch,
    query_text: str,
    query_vector: list[float],
    k: int = 8,
    user_id: int | None = None,
    doc_id: str | None = None,
    doc_ids: list[str] | None = None,
    chunk_type: str | None = None,
) -> dict[str, Any]:
    """Run BM25 + k-NN hybrid search with score normalization pipeline."""
    body: dict[str, Any] = {
        "size": k,
        "query": {
            "hybrid": {
                "queries": [
                    {"match": {"content": query_text}},
                    {"knn": {"embedding": {"vector": query_vector, "k": k}}},
                ]
            }
        },
    }
    if user_id is not None:
        body["post_filter"] = _build_filters(user_id, doc_id=doc_id, doc_ids=doc_ids, chunk_type=chunk_type)
    elif doc_ids or doc_id or chunk_type:
        extra_filters: list[dict[str, Any]] = []
        scope_ids = list(doc_ids) if doc_ids else ([doc_id] if doc_id else None)
        if scope_ids:
            if len(scope_ids) == 1:
                extra_filters.append({"term": {"doc_id": scope_ids[0]}})
            else:
                extra_filters.append({"terms": {"doc_id": scope_ids}})
        if chunk_type:
            extra_filters.append({"term": {"chunk_type": chunk_type}})
        body["post_filter"] = {"bool": {"filter": extra_filters}}

    return client.search(
        index=settings.chunks_index,
        body=body,
        params={"search_pipeline": settings.hybrid_search_pipeline},
    )
