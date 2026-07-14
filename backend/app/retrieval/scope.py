"""Document scope resolution — the UI scope tab is the single source of truth."""

from __future__ import annotations

from fastapi import HTTPException
from opensearchpy import OpenSearch

from app.config import settings
from app.opensearch.documents import get_document_for_user, list_document_records
from app.retrieval.doc_digest import digest_text_from_record
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


def _indexed_document_records(
    client: OpenSearch,
    *,
    user_id: int,
    scope_doc_ids: list[str] | None = None,
) -> list[dict]:
    records = list_document_records(client, user_id)
    if scope_doc_ids is not None:
        allowed = set(scope_doc_ids)
        records = [record for record in records if record.get("doc_id") in allowed]
    return [
        record
        for record in records
        if record.get("ingestion_status") == "indexed"
        and int(record.get("chunk_count") or 0) > 0
    ]


def count_indexed_documents_in_scope(
    client: OpenSearch,
    *,
    user_id: int,
    scope_doc_ids: list[str] | None = None,
) -> int:
    return len(
        _indexed_document_records(
            client,
            user_id=user_id,
            scope_doc_ids=scope_doc_ids,
        )
    )


def _document_inventory_line(record: dict) -> str:
    filename = str(record.get("filename") or record.get("doc_id") or "unknown")
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else "file"
    pages = int(record.get("page_count") or 0)
    chunks = int(record.get("chunk_count") or 0)
    return f"- {filename} ({suffix}, {pages} pages, {chunks} chunks, indexed)"


def doc_inventory_hint_for_agent(
    client: OpenSearch,
    *,
    user_id: int,
    scope_doc_ids: list[str] | None = None,
    max_unscoped: int = 8,
) -> str:
    """Summarize indexed uploads available for retrieval (router context)."""
    indexed = _indexed_document_records(
        client,
        user_id=user_id,
        scope_doc_ids=scope_doc_ids,
    )
    if not indexed:
        return "\n\nDocument library: no indexed documents are available for search."

    lines = [_document_inventory_line(record) for record in indexed]
    if scope_doc_ids is not None:
        header = "Documents in scope (indexed):"
        body = "\n".join(lines)
        return f"\n\n{header}\n{body}"

    shown = lines[:max_unscoped]
    body = "\n".join(shown)
    extra = ""
    if len(lines) > max_unscoped:
        extra = f"\n({len(lines) - max_unscoped} more indexed documents in the library.)"
    return f"\n\nIndexed documents available (recent):\n{body}{extra}"


def doc_digest_hint_for_agent(
    client: OpenSearch,
    *,
    user_id: int,
    scope_doc_ids: list[str] | None = None,
    max_unscoped: int = 2,
) -> str:
    """Inject compact document digests for router context (prefer scoped)."""
    indexed = _indexed_document_records(
        client,
        user_id=user_id,
        scope_doc_ids=scope_doc_ids,
    )
    if not indexed:
        return ""

    # Unscoped: digests only for a couple recent docs; prefer inventory alone when many files.
    if scope_doc_ids is None and len(indexed) > max_unscoped:
        selected = indexed[:max_unscoped]
    elif scope_doc_ids is not None:
        selected = indexed
    else:
        selected = indexed[:max_unscoped]

    lines: list[str] = []
    for record in selected:
        digest_text = digest_text_from_record(record, for_router=True)
        if digest_text:
            lines.append(f"- {digest_text}")

    if not lines:
        return ""

    header = "Document outlines:" if scope_doc_ids is not None else "Recent document outlines:"
    return f"\n\n{header}\n" + "\n".join(lines)


def sources_hint_for_agent(
    client: OpenSearch,
    *,
    user_id: int,
    scope_doc_ids: list[str] | None = None,
    scoped_filenames: list[str] | None = None,
    sql_display_name: str | None = None,
    sql_description: str | None = None,
    sql_tables: list[str] | None = None,
) -> str:
    """One compact sources block for the router (no full SQL schema)."""
    scope_part = scope_hint_for_agent(scope_doc_ids, scoped_filenames=scoped_filenames).strip()
    indexed = _indexed_document_records(
        client,
        user_id=user_id,
        scope_doc_ids=scope_doc_ids,
    )
    indexed_count = len(indexed)

    lines: list[str] = ["\n\n## Available sources"]
    if scope_part:
        # Drop leading newlines from scope hint; keep one line summary when possible.
        lines.append(scope_part.replace("\n\n", "\n").strip())

    if sql_display_name:
        db_bits = [f"Database: {sql_display_name}"]
        desc = (sql_description or "").strip()
        if desc:
            db_bits.append(desc[:160] + ("…" if len(desc) > 160 else ""))
        tables = [name for name in (sql_tables or []) if name][:12]
        if tables:
            db_bits.append("tables: " + ", ".join(tables))
        lines.append(" | ".join(db_bits))
        if indexed_count > 0:
            lines.append(
                "Default: call `query_database` and `search_documents` together in the same turn "
                "unless the user restricts the source. Pass the full user question to `search_documents`."
            )
        else:
            lines.append("Use `query_database` for live PostgreSQL facts.")

    if indexed_count == 0:
        lines.append("Documents: none indexed.")
    else:
        # Scoped: inventory + compact outlines. Unscoped: short inventory, outlines only if few files.
        inv_lines = [_document_inventory_line(record) for record in indexed]
        if scope_doc_ids is not None:
            lines.append("Documents in scope:")
            lines.extend(inv_lines)
            for record in indexed:
                outline = digest_text_from_record(record, for_router=True)
                if outline:
                    lines.append(f"  outline: {outline}")
        else:
            shown = inv_lines[:5]
            lines.append(f"Indexed documents ({indexed_count}):")
            lines.extend(shown)
            if indexed_count > 5:
                lines.append(f"  ({indexed_count - 5} more omitted)")
            if indexed_count <= 3:
                for record in indexed[:2]:
                    outline = digest_text_from_record(record, for_router=True)
                    if outline:
                        lines.append(f"  outline: {outline}")

    return "\n".join(lines)


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
