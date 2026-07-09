"""Chat history helper tests."""

from __future__ import annotations

from app.chat import service as chat_service
from app.auth import service as auth_service


def test_history_for_llm_excludes_last_user(auth_db_session_factory) -> None:
    db = auth_db_session_factory()
    try:
        user = auth_service.create_user(db, "history@test.com", "supersecret1")
        db.commit()
        chat = chat_service.create_session(db, user.id)
        chat_service.append_user_message(db, chat, "first question")
        chat_service.append_assistant_message(db, chat, "first answer")
        chat_service.append_user_message(db, chat, "follow up")
        db.commit()

        history = chat_service.history_for_llm(db, chat, max_turns=6, exclude_last_user=True)
        assert history == [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
        ]
    finally:
        db.close()
