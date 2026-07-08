"""Hybrid search retrieval API (Phase 3)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.auth.dependencies import get_current_user
from app.config import settings
from app.db.models import User
from app.opensearch.documents import get_document_for_user
from app.retrieval.models import SearchRequest, SearchResponse
from app.retrieval.request_log import log_retrieval_request
from app.retrieval.service import hybrid_retrieve

router = APIRouter(prefix="/search", tags=["search"])
logger = logging.getLogger(__name__)


@router.get("", response_model=SearchResponse)
async def search_get(
    request: Request,
    query: str = Query(..., min_length=1, description="Natural language search query"),
    top_k: int = Query(default=settings.default_top_k, ge=1, le=50),
    doc_id: str | None = Query(default=None, description="Optional document scope filter"),
    current_user: User = Depends(get_current_user),
) -> SearchResponse:
    return _run_search(request, query=query, top_k=top_k, doc_id=doc_id, user=current_user)


@router.post("", response_model=SearchResponse)
async def search_post(
    request: Request,
    body: SearchRequest,
    current_user: User = Depends(get_current_user),
) -> SearchResponse:
    return _run_search(
        request,
        query=body.query,
        top_k=body.top_k,
        doc_id=body.doc_id,
        user=current_user,
    )


def _run_search(
    request: Request,
    *,
    query: str,
    top_k: int,
    doc_id: str | None,
    user: User,
) -> SearchResponse:
    client = request.app.state.opensearch
    if doc_id is not None:
        owned = get_document_for_user(client, doc_id, user.id)
        if owned is None:
            raise HTTPException(status_code=404, detail="Document not found")

    try:
        response = hybrid_retrieve(client, query, user_id=user.id, top_k=top_k, doc_id=doc_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc

    log_retrieval_request(
        endpoint="/search",
        query=query,
        top_k=top_k,
        doc_id=doc_id,
        chunks=response.results,
        charts=[],
    )
    return response
