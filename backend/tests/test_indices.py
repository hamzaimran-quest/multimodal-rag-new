"""Unit tests for index mapping definitions."""

from app.config import settings
from app.opensearch.indices import chunks_index_body, documents_index_body


def test_chunks_index_has_knn_enabled():
    body = chunks_index_body()
    assert body["settings"]["index"]["knn"] is True


def test_chunks_embedding_dimension():
    body = chunks_index_body()
    embedding = body["mappings"]["properties"]["embedding"]
    assert embedding["type"] == "knn_vector"
    assert embedding["dimension"] == settings.embedding_dimension


def test_chunks_required_fields():
    props = chunks_index_body()["mappings"]["properties"]
    for field in (
        "chunk_id",
        "doc_id",
        "user_id",
        "filename",
        "page_number",
        "chunk_type",
        "content",
        "embedding",
        "upload_timestamp",
    ):
        assert field in props


def test_documents_registry_fields():
    props = documents_index_body()["mappings"]["properties"]
    for field in (
        "doc_id",
        "user_id",
        "filename",
        "ingestion_status",
        "ingestion_progress",
        "progress_message",
        "upload_timestamp",
    ):
        assert field in props
