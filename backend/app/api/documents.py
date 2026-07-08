"""Document upload and management endpoints (authenticated, user-scoped)."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, UploadFile
from opensearchpy import OpenSearch

from app.api.schemas import (
    DeleteDocumentResponse,
    DocumentListResponse,
    DocumentStatusResponse,
    UploadResponse,
)
from app.auth.dependencies import get_current_user
from app.auth.rate_limit import rate_limit_upload
from app.db.models import User
from app.ingestion.pipeline import (
    remove_document_files,
    run_ingestion,
    save_upload_file,
)
from app.opensearch.chunks import count_chunks_for_document, delete_chunks_for_document
from app.opensearch.documents import (
    create_document_record,
    delete_document_record,
    get_document_for_user,
    list_document_records,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

SUPPORTED_UPLOAD_EXTENSIONS = (".pdf", ".docx")


def _get_opensearch(request: Request) -> OpenSearch:
    return request.app.state.opensearch


def _schedule_ingestion(
    client: OpenSearch,
    user_id: int,
    doc_id: str,
    filename: str,
) -> None:
    try:
        run_ingestion(client, user_id, doc_id, filename)
    except Exception:
        logger.exception("Background ingestion failed for doc_id=%s user_id=%s", doc_id, user_id)


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile,
    current_user: User = Depends(rate_limit_upload),
) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(SUPPORTED_UPLOAD_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    doc_id = str(uuid.uuid4())
    client = _get_opensearch(request)

    save_upload_file(current_user.id, doc_id, file.filename, file_bytes)
    create_document_record(client, doc_id, file.filename, user_id=current_user.id, status="processing")

    background_tasks.add_task(_schedule_ingestion, client, current_user.id, doc_id, file.filename)

    return UploadResponse(doc_id=doc_id, filename=file.filename, status="processing")


@router.get("/{doc_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    doc_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> DocumentStatusResponse:
    client = _get_opensearch(request)
    record = get_document_for_user(client, doc_id, current_user.id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentStatusResponse(
        doc_id=record["doc_id"],
        filename=record["filename"],
        ingestion_status=record["ingestion_status"],
        ingestion_progress=record.get("ingestion_progress", 0),
        progress_message=record.get("progress_message"),
        upload_timestamp=record.get("upload_timestamp"),
        chunk_count=record.get("chunk_count", 0),
        page_count=record.get("page_count", 0),
        error_message=record.get("error_message"),
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> DocumentListResponse:
    client = _get_opensearch(request)
    records = list_document_records(client, current_user.id)
    documents = [
        DocumentStatusResponse(
            doc_id=r["doc_id"],
            filename=r["filename"],
            ingestion_status=r["ingestion_status"],
            ingestion_progress=r.get("ingestion_progress", 0),
            progress_message=r.get("progress_message"),
            upload_timestamp=r.get("upload_timestamp"),
            chunk_count=r.get("chunk_count", 0),
            page_count=r.get("page_count", 0),
            error_message=r.get("error_message"),
        )
        for r in records
    ]
    return DocumentListResponse(documents=documents)


@router.delete("/{doc_id}", response_model=DeleteDocumentResponse)
async def delete_document(
    doc_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> DeleteDocumentResponse:
    client = _get_opensearch(request)
    record = get_document_for_user(client, doc_id, current_user.id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")

    deleted_chunks = delete_chunks_for_document(client, doc_id)
    expected = count_chunks_for_document(client, doc_id)
    if expected != 0:
        raise HTTPException(
            status_code=500,
            detail=f"Delete incomplete: {expected} chunks remain for doc_id={doc_id}",
        )

    delete_document_record(client, doc_id)
    remove_document_files(current_user.id, doc_id)

    return DeleteDocumentResponse(doc_id=doc_id, deleted_chunks=deleted_chunks)
