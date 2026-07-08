"""API tests for authenticated PDF serving with HTTP byte-range support."""

from __future__ import annotations

from pathlib import Path

import pytest
from opensearchpy import OpenSearch

from app.config import settings
from app.ingestion.pipeline import run_ingestion
from app.opensearch.chunks import delete_chunks_for_document
from tests.conftest import requires_opensearch
from tests.pdf_fixtures import build_sample_pdf


async def _upload_and_ingest(client, opensearch_client, tmp_path: Path) -> str:
    pdf_path = build_sample_pdf(tmp_path / "range.pdf")
    with pdf_path.open("rb") as handle:
        response = await client.post(
            "/documents/upload",
            files={"file": ("range.pdf", handle, "application/pdf")},
        )
    assert response.status_code == 200, response.text
    doc_id = response.json()["doc_id"]
    run_ingestion(opensearch_client, 1, doc_id, "range.pdf")
    return doc_id


def _cleanup(opensearch_client: OpenSearch, doc_id: str) -> None:
    delete_chunks_for_document(opensearch_client, doc_id)
    opensearch_client.delete(index=settings.documents_index, id=doc_id, refresh=True, ignore=[404])


@pytest.mark.asyncio
@requires_opensearch
async def test_full_pdf_download_advertises_range_support(
    api_client_with_opensearch, opensearch_client: OpenSearch, tmp_path: Path
):
    doc_id = await _upload_and_ingest(api_client_with_opensearch, opensearch_client, tmp_path)
    try:
        resp = await api_client_with_opensearch.get(f"/documents/{doc_id}/file")
        assert resp.status_code == 200
        assert resp.headers.get("accept-ranges") == "bytes"
        assert resp.headers.get("content-type") == "application/pdf"
        assert resp.content[:5] == b"%PDF-"
    finally:
        _cleanup(opensearch_client, doc_id)


@pytest.mark.asyncio
@requires_opensearch
async def test_range_request_returns_partial_content(
    api_client_with_opensearch, opensearch_client: OpenSearch, tmp_path: Path
):
    doc_id = await _upload_and_ingest(api_client_with_opensearch, opensearch_client, tmp_path)
    try:
        full = await api_client_with_opensearch.get(f"/documents/{doc_id}/file")
        full_bytes = full.content
        assert len(full_bytes) > 20

        ranged = await api_client_with_opensearch.get(
            f"/documents/{doc_id}/file", headers={"Range": "bytes=0-9"}
        )
        # The core guarantee: a range request returns ONLY the requested bytes, not the whole file.
        assert ranged.status_code == 206
        assert ranged.headers.get("content-range") == f"bytes 0-9/{len(full_bytes)}"
        assert ranged.headers.get("content-length") == "10"
        assert ranged.content == full_bytes[:10]
        assert len(ranged.content) < len(full_bytes)

        # A mid-file range.
        mid = await api_client_with_opensearch.get(
            f"/documents/{doc_id}/file", headers={"Range": "bytes=10-19"}
        )
        assert mid.status_code == 206
        assert mid.content == full_bytes[10:20]
    finally:
        _cleanup(opensearch_client, doc_id)


@pytest.mark.asyncio
@requires_opensearch
async def test_unsatisfiable_range_returns_416(
    api_client_with_opensearch, opensearch_client: OpenSearch, tmp_path: Path
):
    doc_id = await _upload_and_ingest(api_client_with_opensearch, opensearch_client, tmp_path)
    try:
        resp = await api_client_with_opensearch.get(
            f"/documents/{doc_id}/file", headers={"Range": "bytes=99999999-100000000"}
        )
        assert resp.status_code == 416
    finally:
        _cleanup(opensearch_client, doc_id)


@pytest.mark.asyncio
@requires_opensearch
async def test_pdf_serving_enforces_ownership_404(
    api_client_with_opensearch,
    second_authed_client,
    opensearch_client: OpenSearch,
    tmp_path: Path,
):
    doc_id = await _upload_and_ingest(api_client_with_opensearch, opensearch_client, tmp_path)
    try:
        # A different user must not be able to read the PDF: 404, not 403.
        resp = await second_authed_client.get(f"/documents/{doc_id}/file")
        assert resp.status_code == 404
    finally:
        _cleanup(opensearch_client, doc_id)
