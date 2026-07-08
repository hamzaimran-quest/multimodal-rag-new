"""Integration tests for hybrid retrieval."""

from __future__ import annotations

import uuid

import pytest

from app.ingestion.embeddings import embed_texts
from app.opensearch.chunks import delete_chunks_for_document
from app.opensearch.testing import index_dummy_chunk
from app.retrieval.service import hybrid_retrieve
from tests.conftest import requires_opensearch


def _index_with_real_embedding(client, *, content: str, doc_id: str, filename: str, page: int):
    embedding = embed_texts([content])[0]
    return index_dummy_chunk(
        client,
        content=content,
        embedding=embedding,
        doc_id=doc_id,
        filename=filename,
        page_number=page,
    )


@requires_opensearch
@pytest.mark.slow
def test_hybrid_retrieve_ranks_relevant_chunk(opensearch_client):
    doc_id = str(uuid.uuid4())
    try:
        _index_with_real_embedding(
            opensearch_client,
            content="Five-Year Financial Highlights revenue growth CNY Million",
            doc_id=doc_id,
            filename="huawei.pdf",
            page=12,
        )
        _index_with_real_embedding(
            opensearch_client,
            content="Unrelated timberland branch deposit insurance regulatory filing",
            doc_id=doc_id,
            filename="timberland.pdf",
            page=3,
        )

        response = hybrid_retrieve(
            opensearch_client,
            "financial highlights revenue",
            user_id=1,
            top_k=2,
            doc_id=doc_id,
        )

        assert response.total >= 1
        top = response.results[0]
        assert "Financial Highlights" in top.content
        assert top.filename == "huawei.pdf"
    finally:
        delete_chunks_for_document(opensearch_client, doc_id)


@requires_opensearch
@pytest.mark.slow
def test_hybrid_retrieve_doc_id_filter(opensearch_client):
    doc_a = str(uuid.uuid4())
    doc_b = str(uuid.uuid4())
    try:
        _index_with_real_embedding(
            opensearch_client,
            content="Annual revenue operating profit financial statement",
            doc_id=doc_a,
            filename="doc_a.pdf",
            page=1,
        )
        _index_with_real_embedding(
            opensearch_client,
            content="Annual revenue operating profit financial statement copy",
            doc_id=doc_b,
            filename="doc_b.pdf",
            page=1,
        )

        scoped = hybrid_retrieve(
            opensearch_client,
            "annual revenue operating profit",
            user_id=1,
            top_k=5,
            doc_id=doc_a,
        )
        assert scoped.total >= 1
        assert all(r.doc_id == doc_a for r in scoped.results)
    finally:
        delete_chunks_for_document(opensearch_client, doc_a)
        delete_chunks_for_document(opensearch_client, doc_b)
