"""DOCX image attachment via block-index proximity (no bbox / no PDF overlap).

Complements ``image_attach.py`` (PDF bbox proximity). DOCX text/table anchors
carry ``block_index`` in ``extra_metadata``; image chunks on nearby blocks attach
when horizontal layout metadata is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

from opensearchpy import OpenSearch

from app.config import settings
from app.retrieval.image_attach import _image_from_hit
from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


def _docx_block_index(chunk: RetrievedChunk) -> int | None:
    extra = chunk.extra_metadata or {}
    if extra.get("source_format") != "docx":
        return None
    raw = extra.get("block_index", chunk.page_number)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _select_docx_anchors(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    anchors = [
        c
        for c in chunks
        if c.chunk_type in {"text", "table"} and _docx_block_index(c) is not None
    ]
    if not anchors:
        return []
    top_score = anchors[0].score
    ratio = settings.image_proximity_score_ratio
    threshold = top_score * ratio if top_score > 0 else float("-inf")
    gated = [c for c in anchors if c.score >= threshold]
    return gated[: settings.image_proximity_anchor_count]


def _block_proximity_score(anchor_block: int, image_block: int, *, radius: int) -> float:
    gap = abs(anchor_block - image_block)
    if gap > radius:
        return 0.0
    return 1.0 - (gap / radius) if radius > 0 else 1.0


def _fetch_docx_image_candidates(
    client: OpenSearch,
    *,
    user_id: int,
    doc_id: str,
    min_block: int,
    max_block: int,
) -> list[dict[str, Any]]:
    body = {
        "size": max((max_block - min_block + 1) * 4, 16),
        "query": {
            "bool": {
                "filter": [
                    {"term": {"user_id": str(user_id)}},
                    {"term": {"doc_id": doc_id}},
                    {"term": {"chunk_type": "image"}},
                    {"term": {"extra_metadata.source_format": "docx"}},
                    {"range": {"page_number": {"gte": min_block, "lte": max_block}}},
                ]
            }
        },
    }
    response = client.search(index=settings.chunks_index, body=body)
    return [hit["_source"] for hit in response.get("hits", {}).get("hits", [])]


def resolve_docx_proximity_attachments(
    client: OpenSearch, chunks: list[RetrievedChunk], *, user_id: int
) -> dict[str, list[dict[str, Any]]]:
    """Map anchor chunk_id -> attached DOCX image dicts (fail-closed on error)."""
    if not settings.image_attach_enabled:
        return {}
    try:
        anchors = _select_docx_anchors(chunks)
        if not anchors:
            return {}

        radius = settings.docx_image_proximity_block_radius
        by_doc: dict[str, list[RetrievedChunk]] = {}
        for anchor in anchors:
            by_doc.setdefault(anchor.doc_id, []).append(anchor)

        min_score = settings.image_min_attachment_score
        attachments: dict[str, list[dict[str, Any]]] = {}

        for doc_id, doc_anchors in by_doc.items():
            anchor_blocks = [_docx_block_index(a) for a in doc_anchors]
            anchor_blocks = [b for b in anchor_blocks if b is not None]
            if not anchor_blocks:
                continue
            min_block = min(anchor_blocks) - radius
            max_block = max(anchor_blocks) + radius
            candidates = _fetch_docx_image_candidates(
                client,
                user_id=user_id,
                doc_id=doc_id,
                min_block=min_block,
                max_block=max_block,
            )
            if not candidates:
                continue

            for anchor in doc_anchors:
                anchor_block = _docx_block_index(anchor)
                if anchor_block is None:
                    continue
                attached: list[dict[str, Any]] = []
                for source in candidates:
                    extra = source.get("extra_metadata") or {}
                    try:
                        image_block = int(extra.get("block_index", source.get("page_number")))
                    except (TypeError, ValueError):
                        continue
                    score = _block_proximity_score(anchor_block, image_block, radius=radius)
                    if score < min_score:
                        continue
                    image = _image_from_hit(source, score, reason="proximity")
                    if image is not None:
                        attached.append(image)
                if attached:
                    attached.sort(key=lambda img: img["score"], reverse=True)
                    attachments[anchor.chunk_id] = attached
        return attachments
    except Exception:
        logger.warning("DOCX proximity image attachment failed; continuing without", exc_info=True)
        return {}
