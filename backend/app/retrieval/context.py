"""Build and trim grounded-answer context for the LLM."""

from __future__ import annotations

import logging

from app.config import settings
from app.ingestion.xlsx_serialize import format_chunk_content_for_llm
from app.retrieval.models import RetrievedChunk
from app.security.injection import wrap_document_excerpt

logger = logging.getLogger(__name__)

_TEXT_TABLE_TYPES = frozenset({"text", "table"})
_SHEET_ROLE_RANK = {"standalone": 0, "satellite": 1, "primary": 2}


def _llm_chunk_priority(chunk: RetrievedChunk) -> tuple[int, float, int]:
    extra = chunk.extra_metadata or {}
    role = str(extra.get("sheet_role") or "primary")
    role_rank = _SHEET_ROLE_RANK.get(role, 2)
    return (role_rank, -chunk.score, len(chunk.content or ""))


def _format_source_block(chunk: RetrievedChunk, index: int) -> str:
    content = format_chunk_content_for_llm(chunk.content, chunk.extra_metadata)
    extra = chunk.extra_metadata or {}
    lines = [
        f"--- Source {index} ---",
        f"Document: {chunk.filename}",
    ]
    if extra.get("source_format") == "docx":
        section = (extra.get("section") or "").strip()
        if section:
            lines.append(f"Section: {section}")
    else:
        lines.append(f"Page: {chunk.page_number}")
    lines.extend([f"Type: {chunk.chunk_type}", f"Content:\n{wrap_document_excerpt(content)}"])
    return "\n".join(lines)


def build_llm_context(chunks: list[RetrievedChunk]) -> str:
    text_table_chunks = [chunk for chunk in chunks if chunk.chunk_type in _TEXT_TABLE_TYPES]
    if not text_table_chunks:
        return "No textual/table context retrieved."

    parts = [_format_source_block(chunk, index) for index, chunk in enumerate(text_table_chunks, start=1)]
    return "\n\n".join(parts)


def select_chunks_for_llm_context(
    chunks: list[RetrievedChunk],
    *,
    max_chars: int | None = None,
) -> list[RetrievedChunk]:
    """Keep highest-priority chunks whose formatted context fits within the char budget."""
    cap = max_chars if max_chars is not None else settings.llm_context_max_chars
    if cap <= 0:
        return chunks

    text_table = [chunk for chunk in chunks if chunk.chunk_type in _TEXT_TABLE_TYPES]
    if not text_table:
        return chunks

    ordered = sorted(text_table, key=_llm_chunk_priority)
    selected: list[RetrievedChunk] = []
    for chunk in ordered:
        candidate = [*selected, chunk]
        if len(build_llm_context(candidate)) <= cap:
            selected.append(chunk)

    if len(selected) < len(text_table):
        logger.info(
            "LLM_CONTEXT_TRIM kept=%s dropped=%s max_chars=%s",
            len(selected),
            len(text_table) - len(selected),
            cap,
        )

    non_text = [chunk for chunk in chunks if chunk.chunk_type not in _TEXT_TABLE_TYPES]
    if not selected:
        return chunks
    return sorted(selected + non_text, key=lambda chunk: chunk.score, reverse=True)
