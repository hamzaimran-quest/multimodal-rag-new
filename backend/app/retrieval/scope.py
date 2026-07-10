"""Document scope resolution — the UI scope tab is the single source of truth."""

from __future__ import annotations

from fastapi import HTTPException
from opensearchpy import OpenSearch

from app.opensearch.documents import get_document_for_user


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
