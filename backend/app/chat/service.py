"""Chat history service layer."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ChatMessage, ChatSession

DEFAULT_TITLE = "New chat"
_TITLE_MAX_LEN = 60


def title_from_message(text: str) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return DEFAULT_TITLE
    if len(cleaned) <= _TITLE_MAX_LEN:
        return cleaned
    return cleaned[: _TITLE_MAX_LEN - 1].rstrip() + "…"


def _touch_session(session: ChatSession) -> None:
    session.updated_at = datetime.now(UTC)


def create_session(db: Session, user_id: int, *, title: str = DEFAULT_TITLE) -> ChatSession:
    chat = ChatSession(user_id=user_id, title=title)
    db.add(chat)
    db.flush()
    return chat


def list_sessions(db: Session, user_id: int) -> list[ChatSession]:
    stmt = (
        select(ChatSession)
        .where(ChatSession.user_id == user_id)
        .order_by(ChatSession.updated_at.desc())
    )
    return list(db.scalars(stmt).all())


def get_session_for_user(db: Session, session_id: int, user_id: int) -> ChatSession | None:
    stmt = select(ChatSession).where(
        ChatSession.id == session_id,
        ChatSession.user_id == user_id,
    )
    return db.scalars(stmt).first()


def delete_session(db: Session, session_id: int, user_id: int) -> bool:
    chat = get_session_for_user(db, session_id, user_id)
    if chat is None:
        return False
    db.delete(chat)
    return True


def resolve_session_for_query(
    db: Session,
    *,
    user_id: int,
    session_id: int | None,
    first_message: str,
) -> ChatSession:
    if session_id is not None:
        chat = get_session_for_user(db, session_id, user_id)
        if chat is None:
            raise LookupError("session_not_found")
        return chat

    return create_session(db, user_id, title=title_from_message(first_message))


def recent_history(
    db: Session, chat: ChatSession, *, limit: int, exclude_last_user: bool = True
) -> list[dict[str, str]]:
    """Recent conversation turns (oldest first) for follow-up context.

    ``exclude_last_user`` drops the just-appended current user message so it is
    not duplicated when passed alongside the live query.
    """
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == chat.id)
        .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
        .limit(limit + 1)
    )
    messages = list(db.scalars(stmt).all())
    messages.reverse()
    if exclude_last_user and messages and messages[-1].role == "user":
        messages = messages[:-1]
    trimmed = messages[-limit:] if limit > 0 else []
    return [{"role": m.role, "content": m.content} for m in trimmed]


def append_user_message(db: Session, chat: ChatSession, content: str) -> ChatMessage:
    if chat.title == DEFAULT_TITLE:
        chat.title = title_from_message(content)
    message = ChatMessage(session_id=chat.id, role="user", content=content)
    db.add(message)
    _touch_session(chat)
    db.flush()
    return message


def append_assistant_message(
    db: Session,
    chat: ChatSession,
    content: str,
    *,
    sources: list[dict[str, Any]] | None = None,
    charts: list[dict[str, Any]] | None = None,
) -> ChatMessage:
    message = ChatMessage(
        session_id=chat.id,
        role="assistant",
        content=content,
        sources=sources or [],
        charts=charts or [],
    )
    db.add(message)
    _touch_session(chat)
    db.flush()
    return message
