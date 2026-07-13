"""Document scope resolution — the UI scope tab is the single source of truth."""

from __future__ import annotations

from fastapi import HTTPException
from opensearchpy import OpenSearch

from app.config import settings
from app.opensearch.documents import get_document_for_user
from app.retrieval.table_query_signal import numeric_comparison_query_detected
from app.retrieval.models import RetrievedChunk
from app.retrieval.xlsx_expand import chunk_covers_anchor


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
            "Do not ask which document to use — search only this file. "
            "Treat factual questions as being about this document: call "
            "`search_documents` rather than asking generic clarification."
        )
    labels = scoped_filenames or scope_doc_ids
    joined = ", ".join(repr(name) for name in labels)
    return (
        f"\n\nDocument scope: restricted to {len(scope_doc_ids)} UI-selected documents "
        f"({joined}). Do not ask which document to use — search only within this set. "
        "Treat factual questions as being about these documents: call "
        "`search_documents` rather than asking generic clarification."
    )


def is_xlsx_filename(filename: str | None) -> bool:
    return str(filename or "").lower().endswith(".xlsx")


def is_pdf_filename(filename: str | None) -> bool:
    return str(filename or "").lower().endswith(".pdf")


def scope_is_pdf_only(
    client: OpenSearch,
    *,
    user_id: int,
    scope_doc_ids: list[str] | None,
) -> bool:
    """True when every scoped document is a PDF."""
    if not scope_doc_ids:
        return False
    for scoped_id in scope_doc_ids:
        record = get_document_for_user(client, scoped_id, user_id)
        if record is None or not is_pdf_filename(record.get("filename")):
            return False
    return True


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
    query: str | None = None,
) -> int:
    """Use format-specific top_k when scope or query signals need table-heavy retrieval."""
    if scope_is_xlsx_only(client, user_id=user_id, scope_doc_ids=scope_doc_ids):
        return max(1, settings.excel_top_k)
    if scope_is_pdf_only(client, user_id=user_id, scope_doc_ids=scope_doc_ids):
        return max(1, settings.pdf_top_k)
    if query and numeric_comparison_query_detected(query):
        return max(1, settings.pdf_top_k)
    requested = top_k or settings.default_top_k
    return max(1, min(50, int(requested)))


def _xlsx_sheet_key(chunk: RetrievedChunk) -> tuple[str, str]:
    extra = chunk.extra_metadata or {}
    return (chunk.doc_id, str(extra.get("sheet_name") or ""))


def _xlsx_sheet_role_rank(chunk: RetrievedChunk) -> int:
    role = str((chunk.extra_metadata or {}).get("sheet_role") or "primary")
    return {"standalone": 0, "satellite": 1, "primary": 2}.get(role, 2)


def _chunk_selection_rank(
    chunk: RetrievedChunk,
    *,
    anchor_keys: set[str],
) -> tuple[int, float, int]:
    """Higher sort key is better when used with reverse=True."""
    anchor_priority = 0
    if anchor_keys:
        if any(chunk_covers_anchor(chunk, anchor_key) for anchor_key in anchor_keys):
            anchor_priority = 2
        elif (chunk.extra_metadata or {}).get("xlsx_anchor_expanded"):
            anchor_priority = 1
    return (anchor_priority, chunk.score, -len(chunk.content or ""))


def _pick_best_xlsx_chunk_per_sheet(
    chunks: list[RetrievedChunk],
    *,
    anchor_keys: set[str] | None = None,
) -> list[RetrievedChunk]:
    keys = anchor_keys or set()
    best_by_sheet: dict[tuple[str, str], RetrievedChunk] = {}
    for chunk in chunks:
        sheet_key = _xlsx_sheet_key(chunk)
        current = best_by_sheet.get(sheet_key)
        if current is None:
            best_by_sheet[sheet_key] = chunk
            continue
        if _chunk_selection_rank(chunk, anchor_keys=keys) > _chunk_selection_rank(
            current,
            anchor_keys=keys,
        ):
            best_by_sheet[sheet_key] = chunk
    return list(best_by_sheet.values())


def _prioritize_xlsx_chunks(
    chunks: list[RetrievedChunk],
    cap: int,
    *,
    anchor_keys: set[str] | None = None,
) -> list[RetrievedChunk]:
    """Prefer linked satellite/standalone sheets over duplicate primary row bands."""
    keys = anchor_keys or set()
    representatives = _pick_best_xlsx_chunk_per_sheet(chunks, anchor_keys=keys)
    if len(representatives) <= cap:
        return representatives

    non_primary = [chunk for chunk in representatives if _xlsx_sheet_role_rank(chunk) < 2]
    primary = [chunk for chunk in representatives if _xlsx_sheet_role_rank(chunk) >= 2]
    non_primary.sort(key=lambda chunk: (-chunk.score, len(chunk.content or "")))
    primary.sort(key=lambda chunk: (-chunk.score, len(chunk.content or "")))

    selected: list[RetrievedChunk] = []
    primary_slots = 1 if primary else 0
    selected.extend(non_primary[: max(0, cap - primary_slots)])
    remaining = cap - len(selected)
    if remaining > 0:
        selected.extend(primary[:remaining])
    return selected


def limit_xlsx_chunks(
    chunks: list[RetrievedChunk],
    *,
    limit: int | None = None,
    anchor_keys: set[str] | None = None,
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

    xlsx_chunks = _prioritize_xlsx_chunks(xlsx_chunks, cap, anchor_keys=anchor_keys)
    return sorted(xlsx_chunks + other_chunks, key=lambda chunk: chunk.score, reverse=True)
