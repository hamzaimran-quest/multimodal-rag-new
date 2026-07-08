"""Authenticated PDF file serving with HTTP byte-range support.

Separate from /images (which serves extracted crops). Range support is what makes
the windowed viewer cheap: PDF.js fetches only the byte ranges for pages the user is
actually looking at, never the whole file up front.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from opensearchpy import OpenSearch

from app.auth.dependencies import get_current_user
from app.db.models import User
from app.ingestion.pipeline import find_pdf_path
from app.opensearch.documents import get_document_for_user

router = APIRouter(prefix="/documents", tags=["documents"])

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
_CHUNK_SIZE = 64 * 1024


def _get_opensearch(request: Request) -> OpenSearch:
    return request.app.state.opensearch


def _resolve_owned_pdf(request: Request, doc_id: str, user_id: int) -> Path:
    client = _get_opensearch(request)
    # Same ownership boundary as documents/chunks/images: unknown or non-owned -> 404.
    record = get_document_for_user(client, doc_id, user_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")

    pdf_path = find_pdf_path(user_id, doc_id)
    if pdf_path is None or not pdf_path.is_file():
        raise HTTPException(status_code=404, detail="Document not found")
    return pdf_path


def _parse_range(range_header: str, file_size: int) -> tuple[int, int] | None:
    """Return (start, end) inclusive byte offsets for a single-range request, or None."""
    match = _RANGE_RE.fullmatch(range_header.strip())
    if not match:
        return None

    start_raw, end_raw = match.group(1), match.group(2)
    if start_raw == "" and end_raw == "":
        return None

    if start_raw == "":
        # Suffix range: last N bytes.
        length = int(end_raw)
        if length <= 0:
            return None
        start = max(0, file_size - length)
        end = file_size - 1
    else:
        start = int(start_raw)
        end = int(end_raw) if end_raw != "" else file_size - 1

    end = min(end, file_size - 1)
    if start > end or start >= file_size:
        return None
    return start, end


def _stream_file_range(path: Path, start: int, end: int):
    remaining = end - start + 1
    with path.open("rb") as handle:
        handle.seek(start)
        while remaining > 0:
            data = handle.read(min(_CHUNK_SIZE, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data


@router.get("/{doc_id}/file")
async def serve_pdf(
    doc_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    range_header: str | None = Header(default=None, alias="Range"),
) -> Response:
    pdf_path = _resolve_owned_pdf(request, doc_id, current_user.id)
    file_size = pdf_path.stat().st_size
    common_headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": f'inline; filename="{pdf_path.name}"',
    }

    if range_header:
        parsed = _parse_range(range_header, file_size)
        if parsed is None:
            # Unsatisfiable range.
            raise HTTPException(
                status_code=416,
                detail="Requested range not satisfiable",
                headers={"Content-Range": f"bytes */{file_size}"},
            )
        start, end = parsed
        headers = {
            **common_headers,
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(end - start + 1),
        }
        return StreamingResponse(
            _stream_file_range(pdf_path, start, end),
            status_code=206,
            media_type="application/pdf",
            headers=headers,
        )

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        headers=common_headers,
    )
