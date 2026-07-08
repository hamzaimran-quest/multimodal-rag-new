"""Pydantic schemas for chat history API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatSessionSummary(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime


class ChatMessageResponse(BaseModel):
    id: int
    role: Literal["user", "assistant"]
    content: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    charts: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime


class ChatSessionDetail(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[ChatMessageResponse]


class ChatListResponse(BaseModel):
    sessions: list[ChatSessionSummary]


class DeleteChatResponse(BaseModel):
    session_id: int
    status: str
