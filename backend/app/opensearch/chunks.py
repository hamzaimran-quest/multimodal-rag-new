"""OpenSearch chunk indexing operations."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from opensearchpy import OpenSearch

from app.config import settings
from app.ingestion.models import ExtractedChunk


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def index_chunks(
    client: OpenSearch,
    *,
    doc_id: str,
    user_id: int,
    filename: str,
    chunks: list[ExtractedChunk],
    embeddings: list[list[float]],
    upload_timestamp: str | None = None,
) -> int:
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings length mismatch")

    ts = upload_timestamp or _now_iso()
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        chunk_id = str(uuid.uuid4())
        body: dict[str, Any] = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "user_id": str(user_id),
            "filename": filename,
            "page_number": chunk.page_number,
            "chunk_type": chunk.chunk_type,
            "content": chunk.content,
            "embedding": embedding,
            "upload_timestamp": ts,
            "extra_metadata": {
                "extraction_method": chunk.extraction_method,
                # Cheap grouping key for query-time image proximity lookup:
                # one batched terms query can fetch all image chunks that share a
                # page with any retrieved text anchor.
                "attachment_key": f"{doc_id}:{chunk.page_number}",
                **chunk.extra_metadata,
            },
        }
        if chunk.bbox:
            body["bbox"] = chunk.bbox
        if chunk.image_path:
            body["image_path"] = chunk.image_path

        client.index(
            index=settings.chunks_index,
            id=chunk_id,
            body=body,
            refresh=False,
        )

    client.indices.refresh(index=settings.chunks_index)
    return len(chunks)


def delete_chunks_for_document(client: OpenSearch, doc_id: str) -> int:
    result = client.delete_by_query(
        index=settings.chunks_index,
        body={"query": {"term": {"doc_id": doc_id}}},
        refresh=True,
    )
    return int(result.get("deleted", 0))


def count_chunks_for_document(client: OpenSearch, doc_id: str) -> int:
    result = client.count(
        index=settings.chunks_index,
        body={"query": {"term": {"doc_id": doc_id}}},
    )
    return int(result.get("count", 0))
