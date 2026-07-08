"""OpenSearch integration tests: indices, k-NN, hybrid search."""

from datetime import datetime, timezone

import pytest

from app.config import settings
from app.opensearch.indices import ensure_indices, get_index_mapping
from app.opensearch.pipelines import ensure_hybrid_search_pipeline
from app.opensearch.testing import (
    hybrid_search,
    index_dummy_chunk,
    knn_search,
    make_dummy_embedding,
)
from tests.conftest import requires_opensearch


@requires_opensearch
def test_indices_exist_with_correct_mapping(opensearch_client):
    ensure_indices(opensearch_client)
    mapping = get_index_mapping(opensearch_client, settings.chunks_index)
    props = mapping[settings.chunks_index]["mappings"]["properties"]
    assert props["embedding"]["type"] == "knn_vector"
    assert props["embedding"]["dimension"] == settings.embedding_dimension
    assert props["content"]["type"] == "text"


@requires_opensearch
def test_document_registry_index(opensearch_client):
    ensure_indices(opensearch_client)
    doc_id = "registry-test-doc"
    opensearch_client.index(
        index=settings.documents_index,
        id=doc_id,
        body={
            "doc_id": doc_id,
            "filename": "huawei.pdf",
            "ingestion_status": "processing",
            "upload_timestamp": datetime.now(timezone.utc).isoformat(),
        },
        refresh=True,
    )
    result = opensearch_client.get(index=settings.documents_index, id=doc_id)
    assert result["_source"]["ingestion_status"] == "processing"
    opensearch_client.delete(index=settings.documents_index, id=doc_id, refresh=True)


@requires_opensearch
def test_knn_search_with_dummy_vectors(opensearch_client, unique_doc_id):
    vec_a = make_dummy_embedding(seed=0.11)
    vec_b = make_dummy_embedding(seed=0.89)

    index_dummy_chunk(
        opensearch_client,
        content="Revenue financial highlights CNY Million",
        embedding=vec_a,
        doc_id=unique_doc_id,
        filename="huawei.pdf",
        page_number=12,
    )
    index_dummy_chunk(
        opensearch_client,
        content="Unrelated timberland branch deposits",
        embedding=vec_b,
        doc_id=unique_doc_id,
        filename="timberland.pdf",
        page_number=3,
    )

    response = knn_search(opensearch_client, vec_a, k=2)
    hits = response["hits"]["hits"]
    assert len(hits) >= 1
    top = hits[0]["_source"]
    assert "financial highlights" in top["content"]


@requires_opensearch
def test_hybrid_search_pipeline(opensearch_client, unique_doc_id):
    assert ensure_hybrid_search_pipeline(opensearch_client) in {True, False}

    vec = make_dummy_embedding(seed=0.42)
    index_dummy_chunk(
        opensearch_client,
        content="Five-Year Financial Highlights revenue growth",
        embedding=vec,
        doc_id=unique_doc_id,
        filename="huawei.pdf",
        page_number=12,
    )

    response = hybrid_search(
        opensearch_client,
        query_text="financial highlights revenue",
        query_vector=vec,
        k=5,
        doc_id=unique_doc_id,
    )
    hits = response["hits"]["hits"]
    assert len(hits) >= 1
    assert hits[0]["_source"]["chunk_type"] == "text"
