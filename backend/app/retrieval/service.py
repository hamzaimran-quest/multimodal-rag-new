"""Hybrid retrieval: embed query, search OpenSearch, map results."""

from __future__ import annotations

from typing import Any

from opensearchpy import OpenSearch

from app.config import settings
from app.ingestion.embeddings import embed_texts
from app.opensearch.search import hybrid_search
from app.retrieval.models import RetrievedChunk, SearchResponse


def image_path_to_url(image_path: str | None) -> str | None:
    """Map stored filesystem path to the authenticated /images URL."""
    if not image_path:
        return None

    normalized = image_path.replace("\\", "/").lstrip("/")

    for prefix in ("data/images/", "images/"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break

    images_root = str(settings.resolved_images_dir).replace("\\", "/").lstrip("/")
    if normalized.startswith(images_root):
        normalized = normalized[len(images_root) :].lstrip("/")

    # Stored paths are {user_id}/{doc_id}/{filename}; API is /images/{doc_id}/{filename}.
    parts = normalized.split("/")
    if len(parts) >= 3:
        return f"/images/{parts[-2]}/{parts[-1]}"
    if len(parts) == 2:
        return f"/images/{parts[0]}/{parts[1]}"
    return None


def get_chunk_for_user(
    client: OpenSearch,
    chunk_id: str,
    user_id: int,
) -> RetrievedChunk | None:
    """Fetch a single chunk by id when it belongs to the user."""
    try:
        response = client.get(index=settings.chunks_index, id=chunk_id)
    except Exception:
        return None

    source = response.get("_source") or {}
    if str(source.get("user_id")) != str(user_id):
        return None

    return parse_search_hit({"_source": source, "_score": 0.0})


def parse_search_hit(hit: dict[str, Any]) -> RetrievedChunk:
    source = hit["_source"]
    extra = source.get("extra_metadata") or {}
    return RetrievedChunk(
        chunk_id=source["chunk_id"],
        doc_id=source["doc_id"],
        filename=source["filename"],
        page_number=source["page_number"],
        chunk_type=source["chunk_type"],
        content=source["content"],
        score=float(hit.get("_score") or 0.0),
        image_url=image_path_to_url(source.get("image_path")),
        extraction_method=extra.get("extraction_method"),
        bbox=source.get("bbox"),
        extra_metadata=extra or None,
    )


def hybrid_retrieve(
    client: OpenSearch,
    query: str,
    *,
    user_id: int,
    top_k: int | None = None,
    doc_id: str | None = None,
    doc_ids: list[str] | None = None,
) -> SearchResponse:
    """Embed the query and run BM25 + k-NN hybrid search scoped to user_id."""
    k = top_k or settings.default_top_k
    query_vector = embed_texts([query])[0]
    effective_doc_ids = list(doc_ids) if doc_ids else ([doc_id] if doc_id else None)

    response = hybrid_search(
        client,
        query_text=query,
        query_vector=query_vector,
        k=k,
        user_id=user_id,
        doc_ids=effective_doc_ids,
    )

    results = [parse_search_hit(hit) for hit in response["hits"]["hits"]]
    return SearchResponse(
        query=query,
        top_k=k,
        doc_id=effective_doc_ids[0] if effective_doc_ids and len(effective_doc_ids) == 1 else None,
        doc_ids=effective_doc_ids,
        total=len(results),
        results=results,
    )
