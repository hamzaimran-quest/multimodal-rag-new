"""Document scope resolution — the UI scope tab is the single source of truth."""

from __future__ import annotations

from fastapi import HTTPException
from opensearchpy import OpenSearch

from app.config import settings
from app.opensearch.documents import get_document_for_user
from app.retrieval.models import RetrievedChunk


def merge_scope_doc_ids(
    *,
    doc_ids: list[str] | None = None,
    doc_id: str | None = None,
) -> list[str] | None:
    """Normalize request fields. None means search the user's full library."""
    merged: list[str] = []
    seen: set[str] = set()
    for raw in doc_ids or []:
        value = str(raw).strip()
        if value and value not in seen:
            seen.add(value)
            merged.append(value)
    if doc_id:
        value = str(doc_id).strip()
        if value and value not in seen:
            merged.append(value)
    return merged if merged else None


def validate_scope_doc_ids(
    client: OpenSearch,
    *,
    user_id: int,
    doc_ids: list[str] | None = None,
    doc_id: str | None = None,
) -> list[str] | None:
    """Validate scoped doc ids belong to the user. Returns None for full-library scope."""
    merged = merge_scope_doc_ids(doc_ids=doc_ids, doc_id=doc_id)
    if merged is None:
        return None
    for scoped_id in merged:
        if get_document_for_user(client, scoped_id, user_id) is None:
            raise HTTPException(status_code=404, detail=f"Document not found: {scoped_id}")
    return merged


def scope_filenames(
    client: OpenSearch,
    *,
    user_id: int,
    scope_doc_ids: list[str],
) -> list[str]:
    names: list[str] = []
    for scoped_id in scope_doc_ids:
        record = get_document_for_user(client, scoped_id, user_id)
        if record is not None:
            names.append(str(record.get("filename") or scoped_id))
    return names


def scope_hint_for_agent(
    scope_doc_ids: list[str] | None,
    *,
    scoped_filenames: list[str] | None = None,
) -> str:
    """Inject UI scope into the agent router prompt."""
    if scope_doc_ids is None:
        return (
            "\n\nDocument scope: all uploaded documents (selected in the UI). "
            "Do not ask which document to use — search within this scope."
        )
    if len(scope_doc_ids) == 1:
        label = (scoped_filenames or [None])[0] or scope_doc_ids[0]
        return (
            f"\n\nDocument scope: restricted to {label!r} (UI selection). "
            "Do not ask which document to use — search only this file."
        )
    labels = scoped_filenames or scope_doc_ids
    joined = ", ".join(repr(name) for name in labels)
    return (
        f"\n\nDocument scope: restricted to {len(scope_doc_ids)} UI-selected documents "
        f"({joined}). Do not ask which document to use — search only within this set."
    )


def is_xlsx_filename(filename: str | None) -> bool:
    return str(filename or "").lower().endswith(".xlsx")


def scope_is_xlsx_only(
    client: OpenSearch,
    *,
    user_id: int,
    scope_doc_ids: list[str] | None,
) -> bool:
    """True when every scoped document is an .xlsx file."""
    if not scope_doc_ids:
        return False
    for scoped_id in scope_doc_ids:
        record = get_document_for_user(client, scoped_id, user_id)
        if record is None or not is_xlsx_filename(record.get("filename")):
            return False
    return True


def resolve_search_top_k(
    client: OpenSearch,
    *,
    user_id: int,
    scope_doc_ids: list[str] | None,
    top_k: int | None = None,
) -> int:
    """Use a smaller top_k when the search scope is Excel-only."""
    if scope_is_xlsx_only(client, user_id=user_id, scope_doc_ids=scope_doc_ids):
        return max(1, settings.excel_top_k)
    requested = top_k or settings.default_top_k
    return max(1, min(50, int(requested)))


def limit_xlsx_chunks(
    chunks: list[RetrievedChunk],
    *,
    limit: int | None = None,
) -> list[RetrievedChunk]:
    """Cap XLSX chunks in a mixed result set; other formats are unchanged."""
    cap = limit if limit is not None else settings.excel_top_k
    if cap <= 0:
        return chunks

    xlsx_chunks: list[RetrievedChunk] = []
    other_chunks: list[RetrievedChunk] = []
    for chunk in chunks:
        source_format = (chunk.extra_metadata or {}).get("source_format")
        if source_format == "xlsx":
            xlsx_chunks.append(chunk)
        else:
            other_chunks.append(chunk)

    if len(xlsx_chunks) <= cap:
        return chunks

    xlsx_chunks = sorted(xlsx_chunks, key=lambda chunk: chunk.score, reverse=True)[:cap]
    return sorted(xlsx_chunks + other_chunks, key=lambda chunk: chunk.score, reverse=True)
