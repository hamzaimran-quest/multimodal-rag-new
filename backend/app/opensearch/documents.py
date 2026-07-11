"""OpenSearch document registry operations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from opensearchpy import OpenSearch

from app.config import settings

INGESTION_STATUSES = ("pending", "processing", "indexed", "failed")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _user_id_str(user_id: int) -> str:
    return str(user_id)


def create_document_record(
    client: OpenSearch,
    doc_id: str,
    filename: str,
    user_id: int,
    status: str = "processing",
) -> None:
    client.index(
        index=settings.documents_index,
        id=doc_id,
        body={
            "doc_id": doc_id,
            "user_id": _user_id_str(user_id),
            "filename": filename,
            "ingestion_status": status,
            "ingestion_progress": 1.0,
            "progress_message": "Queued",
            "upload_timestamp": _now_iso(),
            "chunk_count": 0,
            "page_count": 0,
        },
        refresh=True,
    )


def update_document_record(
    client: OpenSearch,
    doc_id: str,
    *,
    status: str | None = None,
    ingestion_progress: float | None = None,
    progress_message: str | None = None,
    chunk_count: int | None = None,
    page_count: int | None = None,
    error_message: str | None = None,
    workbook_schema: dict[str, Any] | None = None,
) -> None:
    updates: dict[str, Any] = {}
    if status is not None:
        updates["ingestion_status"] = status
    if ingestion_progress is not None:
        updates["ingestion_progress"] = max(0.0, min(100.0, float(ingestion_progress)))
    if progress_message is not None:
        updates["progress_message"] = progress_message
    if chunk_count is not None:
        updates["chunk_count"] = chunk_count
    if page_count is not None:
        updates["page_count"] = page_count
    if error_message is not None:
        updates["error_message"] = error_message
    if workbook_schema is not None:
        updates["workbook_schema"] = workbook_schema

    if not updates:
        return

    client.update(
        index=settings.documents_index,
        id=doc_id,
        body={"doc": updates},
        refresh=True,
        ignore=[404],
    )


def get_document_record(client: OpenSearch, doc_id: str) -> dict[str, Any] | None:
    try:
        result = client.get(index=settings.documents_index, id=doc_id)
        return result["_source"]
    except Exception:
        return None


def get_document_for_user(client: OpenSearch, doc_id: str, user_id: int) -> dict[str, Any] | None:
    record = get_document_record(client, doc_id)
    if record is None:
        return None
    if record.get("user_id") != _user_id_str(user_id):
        return None
    return record


def list_document_records(client: OpenSearch, user_id: int) -> list[dict[str, Any]]:
    result = client.search(
        index=settings.documents_index,
        body={
            "size": 1000,
            "sort": [{"upload_timestamp": {"order": "desc"}}],
            "query": {"term": {"user_id": _user_id_str(user_id)}},
        },
    )
    return [hit["_source"] for hit in result["hits"]["hits"]]


def delete_document_record(client: OpenSearch, doc_id: str) -> None:
    client.delete(index=settings.documents_index, id=doc_id, refresh=True, ignore=[404])
