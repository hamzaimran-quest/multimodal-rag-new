"""Per-user rate limiting tests."""

from __future__ import annotations

import io

import pytest

from app.auth.rate_limit import reset_rate_limits
from app.config import settings


@pytest.fixture(autouse=True)
def _clear_rate_limits():
    reset_rate_limits()
    yield
    reset_rate_limits()


@pytest.mark.asyncio
async def test_upload_rate_limit_returns_429(api_client_with_opensearch, monkeypatch):
    monkeypatch.setattr(settings, "upload_rate_limit_per_minute", 2)

    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"
    for i in range(2):
        response = await api_client_with_opensearch.post(
            "/documents/upload",
            files={"file": (f"doc{i}.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        assert response.status_code == 200, response.text

    blocked = await api_client_with_opensearch.post(
        "/documents/upload",
        files={"file": ("doc3.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert blocked.status_code == 429


@pytest.mark.asyncio
async def test_query_rate_limit_returns_429(api_client_with_opensearch, monkeypatch):
    from app.llm.agent import AgentTurnResult

    monkeypatch.setattr(settings, "query_rate_limit_per_minute", 2)

    async def fake_iter_agent_turn(*args, **kwargs):
        yield {
            "type": "complete",
            "result": AgentTurnResult(
                direct_answer="ok",
                tools_used=[],
            ),
        }

    monkeypatch.setattr("app.api.query.iter_agent_turn", fake_iter_agent_turn)

    for _ in range(2):
        async with api_client_with_opensearch.stream(
            "POST",
            "/query/stream",
            json={"query": "hello"},
        ) as response:
            assert response.status_code == 200

    blocked = await api_client_with_opensearch.post("/query/stream", json={"query": "blocked"})
    assert blocked.status_code == 429
