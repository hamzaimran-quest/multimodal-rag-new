"""Chat history service layer."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
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


def history_for_llm(
    db: Session,
    chat: ChatSession,
    *,
    max_turns: int = 6,
    exclude_last_user: bool = True,
) -> list[dict[str, str]]:
    """Return recent user/assistant turns for the agent (no sources payload)."""
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == chat.id)
        .order_by(ChatMessage.created_at.asc())
    )
    rows = list(db.scalars(stmt).all())
    if exclude_last_user and rows and rows[-1].role == "user":
        rows = rows[:-1]

    messages: list[dict[str, str]] = []
    for row in rows:
        if row.role not in {"user", "assistant"}:
            continue
        messages.append({"role": row.role, "content": row.content})

    if max_turns > 0 and len(messages) > max_turns * 2:
        messages = messages[-(max_turns * 2) :]
    return messages


def prior_user_queries(
    db: Session,
    chat: ChatSession,
    *,
    max_turns: int = 6,
    exclude_last_user: bool = True,
) -> list[str]:
    """Return recent user questions only — for query rewrite (no assistant answers)."""
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == chat.id)
        .order_by(ChatMessage.created_at.asc())
    )
    rows = list(db.scalars(stmt).all())
    if exclude_last_user and rows and rows[-1].role == "user":
        rows = rows[:-1]

    queries: list[str] = []
    max_query_chars = max(0, settings.chat_history_query_max_chars)
    for row in rows:
        if row.role != "user":
            continue
        text = str(row.content or "").strip()
        if not text:
            continue
        if max_query_chars and len(text) > max_query_chars:
            text = text[:max_query_chars]
        queries.append(text)

    if max_turns > 0 and len(queries) > max_turns:
        queries = queries[-max_turns:]
    return queries


def _rows_before_current_user(
    db: Session,
    chat: ChatSession,
    *,
    exclude_last_user: bool = True,
) -> list[ChatMessage]:
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == chat.id)
        .order_by(ChatMessage.created_at.asc())
    )
    rows = list(db.scalars(stmt).all())
    if exclude_last_user and rows and rows[-1].role == "user":
        rows = rows[:-1]
    return rows


def _latest_assistant_row(rows: list[ChatMessage]) -> ChatMessage | None:
    for row in reversed(rows):
        if row.role == "assistant":
            return row
    return None


def latest_assistant_reply(
    db: Session,
    chat: ChatSession,
    *,
    max_chars: int = 800,
    exclude_last_user: bool = True,
) -> str | None:
    """Return the most recent assistant message before the current user turn (truncated)."""
    row = _latest_assistant_row(_rows_before_current_user(db, chat, exclude_last_user=exclude_last_user))
    if row is None:
        return None
    text = str(row.content or "").strip()
    if not text:
        return None
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars]
    return text


def prior_table_chunk_ids(
    db: Session,
    chat: ChatSession,
    *,
    max_assistant_messages: int = 3,
    exclude_last_user: bool = True,
) -> list[str]:
    """Table chunk ids from recent assistant sources — for chart follow-ups."""
    rows = _rows_before_current_user(db, chat, exclude_last_user=exclude_last_user)
    assistant_rows = [row for row in reversed(rows) if row.role == "assistant"]
    if max_assistant_messages > 0:
        assistant_rows = assistant_rows[:max_assistant_messages]

    chunk_ids: list[str] = []
    seen: set[str] = set()
    for row in reversed(assistant_rows):
        for source in row.sources or []:
            if str(source.get("chunk_type", "")).strip().lower() != "table":
                continue
            chunk_id = str(source.get("chunk_id", "")).strip()
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            chunk_ids.append(chunk_id)
    return chunk_ids
