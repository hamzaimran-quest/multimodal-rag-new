"""Integration tests for XLSX upload and ingestion via API."""

from __future__ import annotations

from pathlib import Path

import pytest
from opensearchpy import OpenSearch

from app.config import settings
from app.ingestion.pipeline import run_ingestion
from app.opensearch.chunks import delete_chunks_for_document
from tests.conftest import requires_opensearch
from tests.xlsx_fixtures import build_sample_xlsx


@pytest.fixture
async def authed_user_id(api_client_with_opensearch) -> int:
    response = await api_client_with_opensearch.get("/auth/me")
    assert response.status_code == 200
    return int(response.json()["id"])


@pytest.mark.asyncio
@requires_opensearch
async def test_upload_xlsx_indexes_and_exposes_spreadsheet_api(
    api_client_with_opensearch,
    opensearch_client: OpenSearch,
    authed_user_id: int,
    tmp_path: Path,
):
    xlsx_path = build_sample_xlsx(tmp_path / "upload.xlsx")
    doc_id: str | None = None

    try:
        with xlsx_path.open("rb") as handle:
            response = await api_client_with_opensearch.post(
                "/documents/upload",
                files={
                    "file": (
                        "upload.xlsx",
                        handle,
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                },
            )

        assert response.status_code == 200
        payload = response.json()
        doc_id = payload["doc_id"]
        assert payload["status"] == "processing"
        assert payload["filename"] == "upload.xlsx"

        run_ingestion(opensearch_client, authed_user_id, doc_id, "upload.xlsx")

        status = await api_client_with_opensearch.get(f"/documents/{doc_id}/status")
        assert status.status_code == 200
        body = status.json()
        assert body["ingestion_status"] == "indexed"
        assert body["ingestion_progress"] == 100
        assert body["chunk_count"] >= 2
        assert body["page_count"] == 2

        metadata = await api_client_with_opensearch.get(f"/documents/{doc_id}/spreadsheet")
        assert metadata.status_code == 200
        meta = metadata.json()
        assert meta["sheet_count"] == 2
        assert [sheet["name"] for sheet in meta["sheets"]] == ["Revenue", "Bands"]

        sheet = await api_client_with_opensearch.get(f"/documents/{doc_id}/spreadsheet/sheets/Revenue")
        assert sheet.status_code == 200
        grid = sheet.json()
        assert grid["rows"][0] == ["Year", "Revenue"]
        assert grid["rows"][2] == ["2024", "120"]

        deleted = await api_client_with_opensearch.delete(f"/documents/{doc_id}")
        assert deleted.status_code == 200
        assert deleted.json()["deleted_chunks"] >= 2
        doc_id = None
    finally:
        if doc_id:
            delete_chunks_for_document(opensearch_client, doc_id)
            opensearch_client.delete(
                index=settings.documents_index,
                id=doc_id,
                refresh=True,
                ignore=[404],
            )
