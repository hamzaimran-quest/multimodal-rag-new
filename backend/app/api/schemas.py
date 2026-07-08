"""API request/response schemas."""

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    status: str = Field(description="Ingestion status, typically 'processing'")


class DocumentStatusResponse(BaseModel):
    doc_id: str
    filename: str
    ingestion_status: str
    ingestion_progress: float = 0
    progress_message: str | None = None
    upload_timestamp: str | None = None
    chunk_count: int = 0
    page_count: int = 0
    error_message: str | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentStatusResponse]


class DeleteDocumentResponse(BaseModel):
    doc_id: str
    deleted_chunks: int
    status: str = "deleted"
