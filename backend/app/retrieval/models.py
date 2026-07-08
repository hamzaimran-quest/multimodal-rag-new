"""Retrieval result models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.config import settings


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=settings.default_top_k, ge=1, le=50)
    doc_id: str | None = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    chunk_type: str
    content: str
    score: float
    image_url: str | None = None
    extraction_method: str | None = None
    bbox: list[float] | None = None
    extra_metadata: dict[str, Any] | None = None


class SearchResponse(BaseModel):
    query: str
    top_k: int
    doc_id: str | None = None
    total: int
    results: list[RetrievedChunk] = Field(default_factory=list)
