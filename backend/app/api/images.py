"""Authenticated image serving for extracted PDF chart crops."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from opensearchpy import OpenSearch

from app.auth.dependencies import get_current_user
from app.config import settings
from app.db.models import User
from app.opensearch.documents import get_document_for_user

router = APIRouter(prefix="/images", tags=["images"])


def _get_opensearch(request: Request) -> OpenSearch:
    return request.app.state.opensearch


@router.get("/{doc_id}/{filename}")
async def serve_image(
    doc_id: str,
    filename: str,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> FileResponse:
    client = _get_opensearch(request)
    record = get_document_for_user(client, doc_id, current_user.id)
    if record is None:
        raise HTTPException(status_code=404, detail="Image not found")

    image_path = settings.resolved_images_dir / str(current_user.id) / doc_id / filename
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    resolved = image_path.resolve()
    allowed_root = (settings.resolved_images_dir / str(current_user.id)).resolve()
    if not str(resolved).startswith(str(allowed_root)):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(resolved)
