"""API tests for spreadsheet viewer endpoints."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from opensearchpy import OpenSearch

from app.config import settings
from app.ingestion.pipeline import run_ingestion, save_upload_file
from app.opensearch.chunks import delete_chunks_for_document
from app.opensearch.documents import create_document_record
from tests.conftest import requires_opensearch
from tests.xlsx_fixtures import build_sample_xlsx


@pytest.fixture
async def authed_user_id(api_client_with_opensearch) -> int:
    response = await api_client_with_opensearch.get("/auth/me")
    assert response.status_code == 200
    return int(response.json()["id"])


@pytest.mark.asyncio
@requires_opensearch
async def test_spreadsheet_api_returns_metadata_and_sheet_grid(
    api_client_with_opensearch,
    opensearch_client: OpenSearch,
    authed_user_id: int,
    tmp_path: Path,
):
    xlsx_path = build_sample_xlsx(tmp_path / "report.xlsx")
    doc_id = str(uuid.uuid4())
    filename = "report.xlsx"

    create_document_record(
        opensearch_client,
        doc_id,
        filename,
        user_id=authed_user_id,
        status="processing",
    )
    save_upload_file(authed_user_id, doc_id, filename, xlsx_path.read_bytes())

    try:
        run_ingestion(opensearch_client, authed_user_id, doc_id, filename)

        metadata = await api_client_with_opensearch.get(f"/documents/{doc_id}/spreadsheet")
        assert metadata.status_code == 200
        body = metadata.json()
        assert body["filename"] == filename
        assert body["sheet_count"] == 2
        assert [sheet["name"] for sheet in body["sheets"]] == ["Revenue", "Bands"]

        sheet = await api_client_with_opensearch.get(f"/documents/{doc_id}/spreadsheet/sheets/Revenue")
        assert sheet.status_code == 200
        grid = sheet.json()
        assert grid["name"] == "Revenue"
        assert grid["rows"][0] == ["Year", "Revenue"]
        assert grid["rows"][2] == ["2024", "120"]
    finally:
        delete_chunks_for_document(opensearch_client, doc_id)
        opensearch_client.delete(index=settings.documents_index, id=doc_id, refresh=True)
