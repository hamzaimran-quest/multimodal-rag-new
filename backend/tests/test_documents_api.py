"""API tests for document upload and management."""

from __future__ import annotations

from pathlib import Path

import pytest
from opensearchpy import OpenSearch

from app.config import settings
from app.ingestion.pipeline import run_ingestion
from app.opensearch.chunks import delete_chunks_for_document
from tests.conftest import requires_opensearch
from tests.pdf_fixtures import build_sample_pdf


@pytest.mark.asyncio
@requires_opensearch
async def test_upload_document_returns_processing(
    api_client_with_opensearch,
    opensearch_client: OpenSearch,
    tmp_path: Path,
):
    pdf_path = build_sample_pdf(tmp_path / "upload.pdf")
    doc_id: str | None = None

    try:
        with pdf_path.open("rb") as handle:
            response = await api_client_with_opensearch.post(
                "/documents/upload",
                files={"file": ("upload.pdf", handle, "application/pdf")},
            )

        assert response.status_code == 200
        payload = response.json()
        doc_id = payload["doc_id"]
        assert payload["status"] == "processing"
        assert payload["filename"] == "upload.pdf"

        run_ingestion(opensearch_client, 1, doc_id, "upload.pdf")

        status = await api_client_with_opensearch.get(f"/documents/{doc_id}/status")
        assert status.status_code == 200
        body = status.json()
        assert body["ingestion_status"] == "indexed"
        assert body["ingestion_progress"] == 100
        assert body["progress_message"] == "Completed"
        assert body["chunk_count"] >= 1

        listed = await api_client_with_opensearch.get("/documents")
        assert listed.status_code == 200
        ids = [d["doc_id"] for d in listed.json()["documents"]]
        assert doc_id in ids

        deleted = await api_client_with_opensearch.delete(f"/documents/{doc_id}")
        assert deleted.status_code == 200
        assert deleted.json()["deleted_chunks"] >= 1
        doc_id = None
    finally:
        if doc_id:
            delete_chunks_for_document(opensearch_client, doc_id)
            opensearch_client.delete(
                index=settings.documents_index, id=doc_id, refresh=True, ignore=[404]
            )
