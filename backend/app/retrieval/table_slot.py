"""Reserve multiple table chunks in PDF search results for LLM disambiguation."""

from __future__ import annotations

from opensearchpy import OpenSearch

from app.config import settings
from app.ingestion.embeddings import embed_texts
from app.opensearch.search import hybrid_search
from app.retrieval.models import RetrievedChunk
from app.retrieval.service import parse_search_hit


def _dedupe_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen: set[str] = set()
    unique: list[RetrievedChunk] = []
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        unique.append(chunk)
    return unique


def merge_with_table_slot(
    client: OpenSearch,
    query: str,
    query_vector: list[float] | None,
    results: list[RetrievedChunk],
    *,
    user_id: int,
    top_k: int,
    doc_id: str | None = None,
    doc_ids: list[str] | None = None,
    pdf_scope: bool = False,
    table_slots: int | None = None,
) -> list[RetrievedChunk]:
    """
    Reserve table chunks in search results when scope or query signals need them.

    Hybrid score orders candidates; prose queries still work because the answer model
    can ignore irrelevant tables.
    """
    slots = table_slots if table_slots is not None else settings.pdf_table_slots
    if not pdf_scope or slots <= 0:
        return results

    vector = query_vector if query_vector is not None else embed_texts([query])[0]
    pool_size = max(slots, settings.pdf_table_candidate_pool)
    table_response = hybrid_search(
        client,
        query_text=query,
        query_vector=vector,
        k=pool_size,
        user_id=user_id,
        doc_id=doc_id,
        doc_ids=doc_ids,
        chunk_type="table",
    )
    searched_tables = [parse_search_hit(hit) for hit in table_response["hits"]["hits"]]
    existing_tables = [chunk for chunk in results if chunk.chunk_type == "table"]
    pool = _dedupe_chunks(existing_tables + searched_tables)
    if not pool:
        return results

    pool.sort(key=lambda chunk: chunk.score, reverse=True)
    chosen_tables = pool[:slots]

    non_tables = [chunk for chunk in results if chunk.chunk_type != "table"]
    non_tables.sort(key=lambda chunk: chunk.score, reverse=True)
    text_capacity = max(top_k - len(chosen_tables), 0)
    merged = non_tables[:text_capacity] + chosen_tables
    merged.sort(key=lambda chunk: chunk.score, reverse=True)
    return merged[:top_k]
