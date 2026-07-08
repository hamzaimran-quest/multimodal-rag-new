"""API tests for GET/POST /search."""

from __future__ import annotations

import uuid

import pytest
from opensearchpy import OpenSearch

from app.ingestion.embeddings import embed_texts
from app.opensearch.chunks import delete_chunks_for_document
from app.opensearch.documents import create_document_record
from app.opensearch.testing import index_dummy_chunk
from tests.conftest import requires_opensearch


def _seed_chunk(client: OpenSearch, doc_id: str, content: str, user_id: int = 1) -> None:
    create_document_record(client, doc_id, "sample.pdf", user_id=user_id, status="indexed")
    embedding = embed_texts([content])[0]
    index_dummy_chunk(
        client,
        content=content,
        embedding=embedding,
        doc_id=doc_id,
        user_id=user_id,
        filename="sample.pdf",
        page_number=1,
    )


@pytest.mark.asyncio
@requires_opensearch
@pytest.mark.slow
async def test_search_get_returns_hybrid_results(
    api_client_with_opensearch,
    opensearch_client: OpenSearch,
):
    doc_id = str(uuid.uuid4())
    try:
        _seed_chunk(
            opensearch_client,
            doc_id,
            "Five-Year Financial Highlights revenue growth operating profit",
        )

        response = await api_client_with_opensearch.get(
            "/search",
            params={"query": "financial highlights revenue", "top_k": 3, "doc_id": doc_id},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["query"] == "financial highlights revenue"
        assert body["top_k"] == 3
        assert body["doc_id"] == doc_id
        assert body["total"] >= 1
        assert "Financial Highlights" in body["results"][0]["content"]
    finally:
        delete_chunks_for_document(opensearch_client, doc_id)


@pytest.mark.asyncio
@requires_opensearch
@pytest.mark.slow
async def test_search_post_json_body(
    api_client_with_opensearch,
    opensearch_client: OpenSearch,
):
    doc_id = str(uuid.uuid4())
    try:
        _seed_chunk(
            opensearch_client,
            doc_id,
            "Net income and earnings per share financial performance",
        )

        response = await api_client_with_opensearch.post(
            "/search",
            json={"query": "earnings per share", "top_k": 5, "doc_id": doc_id},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["total"] >= 1
        assert any("earnings" in r["content"].lower() for r in body["results"])
    finally:
        delete_chunks_for_document(opensearch_client, doc_id)
