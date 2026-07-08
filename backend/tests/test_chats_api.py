"""Chat history API tests."""

from __future__ import annotations

import pytest

from app.chat import service
from app.db.session import get_db
from app.main import app


@pytest.mark.asyncio
async def test_create_and_list_chats(api_client_with_opensearch):
    created = await api_client_with_opensearch.post("/chats")
    assert created.status_code == 201
    body = created.json()
    assert body["title"] == "New chat"
    session_id = body["id"]

    listed = await api_client_with_opensearch.get("/chats")
    assert listed.status_code == 200
    sessions = listed.json()["sessions"]
    assert any(s["id"] == session_id for s in sessions)


@pytest.mark.asyncio
async def test_get_chat_returns_messages(api_client_with_opensearch):
    created = await api_client_with_opensearch.post("/chats")
    session_id = created.json()["id"]

    # Append messages through the service layer (simulates query persistence).
    db_gen = app.dependency_overrides[get_db]()
    db = next(db_gen)
    try:
        chat = service.get_session_for_user(db, session_id, user_id=1)
        assert chat is not None
        service.append_user_message(db, chat, "What is revenue?")
        service.append_assistant_message(
            db,
            chat,
            "Revenue grew year-over-year.",
            sources=[{"chunk_id": "c1", "filename": "a.pdf", "page_number": 1, "chunk_type": "text", "snippet": "rev", "score": 0.9}],
            charts=[],
        )
        db.commit()
    finally:
        db_gen.close()

    detail = await api_client_with_opensearch.get(f"/chats/{session_id}")
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["id"] == session_id
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][1]["role"] == "assistant"
    assert payload["messages"][1]["sources"][0]["chunk_id"] == "c1"


@pytest.mark.asyncio
async def test_delete_chat(api_client_with_opensearch):
    created = await api_client_with_opensearch.post("/chats")
    session_id = created.json()["id"]

    deleted = await api_client_with_opensearch.delete(f"/chats/{session_id}")
    assert deleted.status_code == 200

    missing = await api_client_with_opensearch.get(f"/chats/{session_id}")
    assert missing.status_code == 404


@pytest.mark.asyncio
async def test_user_cannot_access_other_users_chat(
    api_client_with_opensearch,
    second_authed_client,
):
    created = await api_client_with_opensearch.post("/chats")
    session_id = created.json()["id"]

    denied = await second_authed_client.get(f"/chats/{session_id}")
    assert denied.status_code == 404

    delete_denied = await second_authed_client.delete(f"/chats/{session_id}")
    assert delete_denied.status_code == 404
