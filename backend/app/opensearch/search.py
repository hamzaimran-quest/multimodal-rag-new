"""Search helpers for k-NN and hybrid retrieval."""

from typing import Any

from opensearchpy import OpenSearch

from app.config import settings


def _build_filters(
    user_id: int,
    doc_id: str | None = None,
    chunk_type: str | None = None,
    metadata_filters: dict[str, str] | None = None,
    exclude_metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    filters: list[dict[str, Any]] = [{"term": {"user_id": str(user_id)}}]
    if doc_id:
        filters.append({"term": {"doc_id": doc_id}})
    if chunk_type:
        filters.append({"term": {"chunk_type": chunk_type}})
    for key, value in (metadata_filters or {}).items():
        filters.append({"term": {f"extra_metadata.{key}": value}})
    for key, value in (exclude_metadata or {}).items():
        filters.append({"bool": {"must_not": [{"term": {f"extra_metadata.{key}": value}}]}})
    return {"bool": {"filter": filters}}


def knn_search(
    client: OpenSearch,
    query_vector: list[float],
    k: int = 5,
    user_id: int | None = None,
    doc_id: str | None = None,
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
        body["post_filter"] = _build_filters(user_id, doc_id)
    elif doc_id:
        body["post_filter"] = {"term": {"doc_id": doc_id}}

    return client.search(index=settings.chunks_index, body=body)


def hybrid_search(
    client: OpenSearch,
    query_text: str,
    query_vector: list[float],
    k: int = 8,
    user_id: int | None = None,
    doc_id: str | None = None,
    chunk_type: str | None = None,
    metadata_filters: dict[str, str] | None = None,
    exclude_metadata: dict[str, str] | None = None,
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
        body["post_filter"] = _build_filters(
            user_id,
            doc_id,
            chunk_type,
            metadata_filters=metadata_filters,
            exclude_metadata=exclude_metadata,
        )
    elif doc_id or chunk_type or metadata_filters or exclude_metadata:
        extra_filters: list[dict[str, Any]] = []
        if doc_id:
            extra_filters.append({"term": {"doc_id": doc_id}})
        if chunk_type:
            extra_filters.append({"term": {"chunk_type": chunk_type}})
        for key, value in (metadata_filters or {}).items():
            extra_filters.append({"term": {f"extra_metadata.{key}": value}})
        for key, value in (exclude_metadata or {}).items():
            extra_filters.append({"bool": {"must_not": [{"term": {f"extra_metadata.{key}": value}}]}})
        body["post_filter"] = {"bool": {"filter": extra_filters}}

    return client.search(
        index=settings.chunks_index,
        body=body,
        params={"search_pipeline": settings.hybrid_search_pipeline},
    )
