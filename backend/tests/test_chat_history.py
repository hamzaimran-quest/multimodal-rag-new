"""Chat history helper tests."""

from __future__ import annotations

from app.chat import service as chat_service
from app.auth import service as auth_service


def test_prior_user_queries_excludes_last_user_and_answers(auth_db_session_factory) -> None:
    db = auth_db_session_factory()
    try:
        user = auth_service.create_user(db, "history@test.com", "supersecret1")
        db.commit()
        chat = chat_service.create_session(db, user.id)
        chat_service.append_user_message(db, chat, "first question")
        chat_service.append_assistant_message(db, chat, "first answer")
        chat_service.append_user_message(db, chat, "follow up")
        db.commit()

        queries = chat_service.prior_user_queries(db, chat, max_turns=6, exclude_last_user=True)
        assert queries == ["first question"]
    finally:
        db.close()


def test_latest_assistant_reply_returns_most_recent_truncated(auth_db_session_factory) -> None:
    db = auth_db_session_factory()
    try:
        user = auth_service.create_user(db, "reply@test.com", "supersecret1")
        db.commit()
        chat = chat_service.create_session(db, user.id)
        chat_service.append_user_message(db, chat, "first question")
        chat_service.append_assistant_message(db, chat, "first answer")
        chat_service.append_user_message(db, chat, "second question")
        chat_service.append_assistant_message(db, chat, "second answer is longer")
        chat_service.append_user_message(db, chat, "follow up")
        db.commit()

        reply = chat_service.latest_assistant_reply(
            db,
            chat,
            max_chars=10,
            exclude_last_user=True,
        )
        assert reply == "second ans"
    finally:
        db.close()


def test_latest_assistant_reply_none_on_first_turn(auth_db_session_factory) -> None:
    db = auth_db_session_factory()
    try:
        user = auth_service.create_user(db, "reply2@test.com", "supersecret1")
        db.commit()
        chat = chat_service.create_session(db, user.id)
        chat_service.append_user_message(db, chat, "only question")
        db.commit()

        assert chat_service.latest_assistant_reply(db, chat) is None
    finally:
        db.close()


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
