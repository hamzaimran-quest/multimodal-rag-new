"""Cross-user document isolation tests."""

from __future__ import annotations

import uuid

import pytest
from opensearchpy import OpenSearch

from app.ingestion.embeddings import embed_texts
from app.opensearch.chunks import delete_chunks_for_document
from app.opensearch.documents import create_document_record
from app.opensearch.testing import index_dummy_chunk
from tests.conftest import requires_opensearch


def _seed_owned_doc(client: OpenSearch, user_id: int, doc_id: str, content: str) -> None:
    create_document_record(client, doc_id, "private.pdf", user_id=user_id, status="indexed")
    embedding = embed_texts([content])[0]
    index_dummy_chunk(
        client,
        content=content,
        embedding=embedding,
        doc_id=doc_id,
        user_id=user_id,
        filename="private.pdf",
        page_number=1,
    )


@pytest.mark.asyncio
@requires_opensearch
async def test_user_cannot_read_other_users_document_status(
    api_client_with_opensearch,
    second_authed_client,
    opensearch_client: OpenSearch,
):
    doc_id = str(uuid.uuid4())
    try:
        _seed_owned_doc(opensearch_client, user_id=1, doc_id=doc_id, content="Owner-only financial data")

        denied = await second_authed_client.get(f"/documents/{doc_id}/status")
        assert denied.status_code == 404
    finally:
        delete_chunks_for_document(opensearch_client, doc_id)
        opensearch_client.delete(index="rag_documents", id=doc_id, refresh=True, ignore=[404])


@pytest.mark.asyncio
@requires_opensearch
@pytest.mark.slow
async def test_search_does_not_return_other_users_chunks(
    api_client_with_opensearch,
    second_authed_client,
    opensearch_client: OpenSearch,
):
    doc_id = str(uuid.uuid4())
    try:
        _seed_owned_doc(
            opensearch_client,
            user_id=1,
            doc_id=doc_id,
            content="Secret revenue operating profit confidential",
        )

        response = await second_authed_client.get(
            "/search",
            params={"query": "secret revenue operating profit", "top_k": 5, "doc_id": doc_id},
        )
        assert response.status_code == 404
    finally:
        delete_chunks_for_document(opensearch_client, doc_id)
        opensearch_client.delete(index="rag_documents", id=doc_id, refresh=True, ignore=[404])
