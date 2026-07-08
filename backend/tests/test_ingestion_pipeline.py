"""Integration tests for PDF ingestion pipeline."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT, settings
from app.ingestion.embeddings import embed_texts
from app.ingestion.pipeline import run_ingestion, save_upload_file
from app.ingestion.text import extract_pdf_text_and_tables
from app.opensearch.chunks import count_chunks_for_document, delete_chunks_for_document
from app.opensearch.documents import create_document_record, get_document_record
from tests.conftest import requires_opensearch
from tests.pdf_fixtures import build_sample_pdf


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> Path:
    return build_sample_pdf(tmp_path / "sample.pdf")


@requires_opensearch
def test_extract_sample_pdf_produces_chunks(sample_pdf_path: Path):
    chunks = extract_pdf_text_and_tables(str(sample_pdf_path))
    assert len(chunks) >= 1
    assert any("Financial Highlights" in c.content for c in chunks)
    assert all(c.chunk_type in {"text", "table"} for c in chunks)


@requires_opensearch
@pytest.mark.slow
def test_embed_texts_dimension():
    vectors = embed_texts(["revenue growth", "operating profit"])
    assert len(vectors) == 2
    assert len(vectors[0]) == settings.embedding_dimension
    for vector in vectors:
        norm = sum(v * v for v in vector) ** 0.5
        assert abs(norm - 1.0) < 1e-4


@requires_opensearch
def test_run_ingestion_indexes_chunks(opensearch_client, sample_pdf_path: Path):
    doc_id = str(uuid.uuid4())
    filename = "sample.pdf"
    pdf_bytes = sample_pdf_path.read_bytes()

    create_document_record(opensearch_client, doc_id, filename, user_id=1, status="processing")
    save_upload_file(1, doc_id, filename, pdf_bytes)

    try:
        count = run_ingestion(opensearch_client, 1, doc_id, filename)
        assert count >= 1
        assert count_chunks_for_document(opensearch_client, doc_id) == count

        record = get_document_record(opensearch_client, doc_id)
        assert record is not None
        assert record["ingestion_status"] == "indexed"
        assert record["ingestion_progress"] == 100
        assert record["chunk_count"] == count
    finally:
        delete_chunks_for_document(opensearch_client, doc_id)
        opensearch_client.delete(index=settings.documents_index, id=doc_id, refresh=True)


@pytest.mark.slow
@requires_opensearch
def test_ingest_huawei_pdf(opensearch_client):
    pdf_path = PROJECT_ROOT / "huawei.pdf"
    if not pdf_path.exists():
        pytest.skip("huawei.pdf not found at project root")

    doc_id = str(uuid.uuid4())
    filename = "huawei.pdf"
    create_document_record(opensearch_client, doc_id, filename, user_id=1, status="processing")
    save_upload_file(1, doc_id, filename, pdf_path.read_bytes())

    try:
        count = run_ingestion(opensearch_client, 1, doc_id, filename)
        assert count > 10
        record = get_document_record(opensearch_client, doc_id)
        assert record["ingestion_status"] == "indexed"
        assert record["ingestion_progress"] == 100
    finally:
        delete_chunks_for_document(opensearch_client, doc_id)
        opensearch_client.delete(index=settings.documents_index, id=doc_id, refresh=True)
