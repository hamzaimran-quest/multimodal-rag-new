"""Test helpers for indexing dummy chunk vectors."""

import uuid
from datetime import datetime, timezone
from typing import Any

from opensearchpy import OpenSearch

from app.config import settings
from app.opensearch.search import hybrid_search, knn_search

__all__ = [
    "make_dummy_embedding",
    "index_dummy_chunk",
    "knn_search",
    "hybrid_search",
]


def make_dummy_embedding(seed: float = 0.1) -> list[float]:
    """Build a deterministic 384-dim unit-ish vector for integration tests."""
    dim = settings.embedding_dimension
    values = [((i + 1) * seed) % 1.0 for i in range(dim)]
    norm = sum(v * v for v in values) ** 0.5 or 1.0
    return [v / norm for v in values]


def index_dummy_chunk(
    client: OpenSearch,
    *,
    content: str,
    embedding: list[float] | None = None,
    chunk_type: str = "text",
    doc_id: str | None = None,
    user_id: int = 1,
    filename: str = "test.pdf",
    page_number: int = 1,
) -> dict[str, Any]:
    """Index a single test chunk and return the document body."""
    chunk_id = str(uuid.uuid4())
    doc_id = doc_id or str(uuid.uuid4())
    body = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "user_id": str(user_id),
        "filename": filename,
        "page_number": page_number,
        "chunk_type": chunk_type,
        "content": content,
        "embedding": embedding or make_dummy_embedding(),
        "upload_timestamp": datetime.now(timezone.utc).isoformat(),
        "extra_metadata": {"extraction_method": "test"},
    }
    client.index(
        index=settings.chunks_index,
        id=chunk_id,
        body=body,
        refresh=True,
    )
    return body
