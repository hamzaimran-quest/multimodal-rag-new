"""Tests for debug table inspection endpoint."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_debug_table_inspect_requires_identity(api_client_with_opensearch, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "debug_endpoints_enabled", True)
    response = await api_client_with_opensearch.get(
        "/debug/table-inspect",
        params={"page_number": 9},
    )
    assert response.status_code == 400
    assert "filename or doc_id" in response.text

