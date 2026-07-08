"""API health endpoint tests."""

import pytest


@pytest.mark.asyncio
async def test_health_returns_ok(api_client):
    response = await api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ready_when_opensearch_up(api_client_with_opensearch):
    response = await api_client_with_opensearch.get("/ready")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is True
    assert payload["checks"]["chunks_index"] is True
    assert payload["checks"]["documents_index"] is True
    assert payload["checks"]["hybrid_pipeline"] is True
    assert payload["checks"]["opensearch"] in {"green", "yellow"}
