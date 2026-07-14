"""Build and format cached document digests for router hints."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from app.ingestion.models import ExtractedChunk

_MAX_SECTIONS = 8
_MAX_SHEETS = 12
_MAX_DIGEST_CHARS = 900
_MAX_ROUTER_DIGEST_CHARS = 450
_MAX_ROUTER_SECTIONS = 5


def _source_format(filename: str, chunks: list[ExtractedChunk]) -> str:
    for chunk in chunks:
        fmt = (chunk.extra_metadata or {}).get("source_format")
        if fmt:
            return str(fmt)
    suffix = Path(filename).suffix.lower().lstrip(".")
    return suffix or "file"


def _ordered_unique(values: list[str], *, limit: int) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        cleaned = value.strip()
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(cleaned)
        if len(ordered) >= limit:
            break
    return ordered


def _collect_sections(chunks: list[ExtractedChunk]) -> list[str]:
    labels: list[str] = []
    for chunk in sorted(
        chunks,
        key=lambda item: (
            int(item.page_number or 0),
            int((item.extra_metadata or {}).get("block_index") or 0),
        ),
    ):
        meta = chunk.extra_metadata or {}
        for key in ("section", "subsection"):
            label = str(meta.get(key) or "").strip()
            if label:
                labels.append(label)
    return _ordered_unique(labels, limit=_MAX_SECTIONS)


def _sheet_names_from_schema(workbook_schema: dict[str, Any] | None) -> list[str]:
    if not workbook_schema:
        return []
    names: list[str] = []
    for sheet in workbook_schema.get("standalone_sheets") or []:
        if sheet:
            names.append(str(sheet))
    for cluster in workbook_schema.get("clusters") or []:
        primary = cluster.get("primary_sheet")
        if primary:
            names.append(str(primary))
        for satellite in cluster.get("satellites") or []:
            sheet = satellite.get("sheet")
            if sheet:
                names.append(str(sheet))
    return _ordered_unique(names, limit=_MAX_SHEETS)


def _sheet_names_from_chunks(chunks: list[ExtractedChunk]) -> list[str]:
    names: list[str] = []
    for chunk in sorted(chunks, key=lambda item: int(item.page_number or 0)):
        sheet = (chunk.extra_metadata or {}).get("sheet_name")
        if sheet:
            names.append(str(sheet))
    return _ordered_unique(names, limit=_MAX_SHEETS)


def _chunk_type_counts(chunks: list[ExtractedChunk]) -> dict[str, int]:
    counts = Counter(chunk.chunk_type or "other" for chunk in chunks)
    return dict(sorted(counts.items()))


def _format_digest_text(
    *,
    filename: str,
    source_format: str,
    page_count: int | None,
    chunk_types: dict[str, int],
    sections: list[str],
    sheet_names: list[str],
) -> str:
    lines = [f"File: {filename} ({source_format})"]
    if page_count and page_count > 0:
        unit = "sheets" if source_format == "xlsx" else "pages"
        lines.append(f"Size: {page_count} {unit}")
    if chunk_types:
        parts = ", ".join(f"{name}={count}" for name, count in chunk_types.items())
        lines.append(f"Chunk types: {parts}")
    if sheet_names:
        lines.append("Sheets: " + ", ".join(sheet_names))
    if sections:
        lines.append("Sections:")
        lines.extend(f"- {label}" for label in sections)
    text = "\n".join(lines)
    if len(text) > _MAX_DIGEST_CHARS:
        text = text[: _MAX_DIGEST_CHARS].rsplit("\n", 1)[0].rstrip() + "\n..."
    return text


def build_doc_digest_from_chunks(
    *,
    filename: str,
    chunks: list[ExtractedChunk],
    page_count: int | None = None,
    workbook_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact digest payload from extracted chunks."""
    source_format = _source_format(filename, chunks)
    sections = _collect_sections(chunks) if source_format in {"pdf", "docx"} else []
    sheet_names = _sheet_names_from_schema(workbook_schema)
    if not sheet_names and source_format == "xlsx":
        sheet_names = _sheet_names_from_chunks(chunks)
    chunk_types = _chunk_type_counts(chunks)
    if page_count is None and chunks:
        page_count = max(int(chunk.page_number or 0) for chunk in chunks)

    digest_text = _format_digest_text(
        filename=filename,
        source_format=source_format,
        page_count=page_count,
        chunk_types=chunk_types,
        sections=sections,
        sheet_names=sheet_names,
    )
    return {
        "digest": digest_text,
        "source_format": source_format,
        "sections": sections,
        "sheet_names": sheet_names,
        "chunk_types": chunk_types,
        "page_count": page_count or 0,
    }


def digest_text_from_record(record: dict[str, Any], *, for_router: bool = False) -> str:
    """Return digest text for a document registry record."""
    payload = record.get("doc_digest")
    if not isinstance(payload, dict):
        return ""

    if for_router:
        filename = str(record.get("filename") or payload.get("source_format") or "document")
        source_format = str(payload.get("source_format") or "file")
        sections = list(payload.get("sections") or [])[:_MAX_ROUTER_SECTIONS]
        sheets = list(payload.get("sheet_names") or [])[:_MAX_SHEETS]
        page_count = int(payload.get("page_count") or record.get("page_count") or 0)
        parts = [f"{filename} ({source_format}"]
        if page_count > 0:
            unit = "sheets" if source_format == "xlsx" else "pages"
            parts[0] += f", {page_count} {unit}"
        parts[0] += ")"
        if sheets:
            parts.append("sheets: " + ", ".join(sheets[:8]))
        if sections:
            parts.append("sections: " + "; ".join(sections))
        text = " | ".join(parts)
        if len(text) > _MAX_ROUTER_DIGEST_CHARS:
            return text[:_MAX_ROUTER_DIGEST_CHARS].rstrip() + "…"
        return text

    text = str(payload.get("digest") or "").strip()
    return text
