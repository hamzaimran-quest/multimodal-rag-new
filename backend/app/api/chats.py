"""Chat history API: list, create, load, and delete conversation sessions."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.chat import service
from app.chat.schemas import (
    ChatListResponse,
    ChatMessageResponse,
    ChatSessionDetail,
    ChatSessionSummary,
    DeleteChatResponse,
)
from app.db.models import User
from app.db.session import get_db

router = APIRouter(prefix="/chats", tags=["chats"])


def _session_summary(chat) -> ChatSessionSummary:
    return ChatSessionSummary(
        id=chat.id,
        title=chat.title,
        created_at=chat.created_at,
        updated_at=chat.updated_at,
    )


def _message_response(message) -> ChatMessageResponse:
    return ChatMessageResponse(
        id=message.id,
        role=message.role,  # type: ignore[arg-type]
        content=message.content,
        sources=message.sources or [],
        charts=message.charts or [],
        sql_meta=message.sql_meta,
        created_at=message.created_at,
    )


@router.get("", response_model=ChatListResponse)
def list_chats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChatListResponse:
    sessions = service.list_sessions(db, current_user.id)
    return ChatListResponse(sessions=[_session_summary(s) for s in sessions])


@router.post("", response_model=ChatSessionSummary, status_code=status.HTTP_201_CREATED)
def create_chat(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChatSessionSummary:
    chat = service.create_session(db, current_user.id)
    db.commit()
    db.refresh(chat)
    return _session_summary(chat)


@router.get("/{session_id}", response_model=ChatSessionDetail)
def get_chat(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChatSessionDetail:
    chat = service.get_session_for_user(db, session_id, current_user.id)
    if chat is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")

    return ChatSessionDetail(
        id=chat.id,
        title=chat.title,
        created_at=chat.created_at,
        updated_at=chat.updated_at,
        messages=[_message_response(m) for m in chat.messages],
    )


@router.delete("/{session_id}", response_model=DeleteChatResponse)
def delete_chat(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DeleteChatResponse:
    deleted = service.delete_session(db, session_id, current_user.id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
    db.commit()
    return DeleteChatResponse(session_id=session_id, status="deleted")
