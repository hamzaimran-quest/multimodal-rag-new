"""DOCX image attachment: intent retrieval + block-index proximity.

Completely separate from ``image_attach.py`` (PDF bbox proximity + intent).
"""

from __future__ import annotations

import logging
from typing import Any

from opensearchpy import OpenSearch

from app.config import settings
from app.opensearch.search import hybrid_search
from app.retrieval.image_attach import _gate_intent_images
from app.retrieval.models import RetrievedChunk
from app.retrieval.service import image_path_to_url

logger = logging.getLogger(__name__)


def _image_from_hit(source: dict[str, Any], score: float, reason: str) -> dict[str, Any] | None:
    image_url = image_path_to_url(source.get("image_path"))
    if image_url is None:
        return None
    extra = source.get("extra_metadata") or {}
    return {
        "image_chunk_id": source["chunk_id"],
        "doc_id": source.get("doc_id"),
        "filename": source.get("filename"),
        "page_number": source.get("page_number"),
        "image_url": image_url,
        "bbox": None,
        "caption": extra.get("image_caption") or "",
        "score": round(float(score), 4),
        "reason": reason,
        "source_format": "docx",
        "section": extra.get("section"),
    }


def _block_distance_score(anchor_block: int, image_block: int, *, window: int) -> float:
    """Proximity in [0, 1] by block ordinal distance (same block = 1.0)."""
    distance = abs(anchor_block - image_block)
    if distance > window:
        return 0.0
    if distance == 0:
        return 1.0
    return max(0.0, 1.0 - (distance / (window + 1)))


def _select_docx_anchors(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Top DOCX text/table hits, score-gated relative to the top DOCX hit."""
    anchors = [
        c
        for c in chunks
        if c.chunk_type in {"text", "table"}
        and (c.extra_metadata or {}).get("source_format") == "docx"
    ]
    if not anchors:
        return []
    top_score = anchors[0].score
    ratio = settings.image_proximity_score_ratio
    threshold = top_score * ratio if top_score > 0 else float("-inf")
    gated = [c for c in anchors if c.score >= threshold]
    return gated[: settings.image_proximity_anchor_count]


def _attachment_keys_for_anchor(anchor: RetrievedChunk, *, window: int) -> set[str]:
    block_index = anchor.page_number
    return {
        f"{anchor.doc_id}:{block_index + offset}"
        for offset in range(-window, window + 1)
        if block_index + offset > 0
    }


def _fetch_docx_candidate_images(
    client: OpenSearch, keys: set[str], user_id: int
) -> dict[str, list[dict[str, Any]]]:
    if not keys:
        return {}
    body = {
        "size": max(len(keys) * 6, 24),
        "query": {
            "bool": {
                "filter": [
                    {"term": {"user_id": str(user_id)}},
                    {"term": {"chunk_type": "image"}},
                    {"term": {"extra_metadata.source_format": "docx"}},
                    {"terms": {"extra_metadata.attachment_key": sorted(keys)}},
                ]
            }
        },
    }
    response = client.search(index=settings.chunks_index, body=body)
    by_key: dict[str, list[dict[str, Any]]] = {}
    for hit in response.get("hits", {}).get("hits", []):
        source = hit["_source"]
        extra = source.get("extra_metadata") or {}
        key = extra.get("attachment_key")
        if not key:
            key = f"{source.get('doc_id')}:{source.get('page_number')}"
        by_key.setdefault(key, []).append(source)
    return by_key


def resolve_docx_proximity_attachments(
    client: OpenSearch, chunks: list[RetrievedChunk], *, user_id: int
) -> dict[str, list[dict[str, Any]]]:
    """Map DOCX anchor chunk_id -> images on the same or adjacent block ordinals."""
    if not settings.image_attach_enabled:
        return {}
    try:
        anchors = _select_docx_anchors(chunks)
        if not anchors:
            return {}

        window = settings.docx_proximity_block_window
        keys: set[str] = set()
        for anchor in anchors:
            keys.update(_attachment_keys_for_anchor(anchor, window=window))
        candidates_by_key = _fetch_docx_candidate_images(client, keys, user_id)
        if not candidates_by_key:
            return {}

        min_score = settings.image_min_attachment_score
        attachments: dict[str, list[dict[str, Any]]] = {}
        for anchor in anchors:
            attached: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            for key in _attachment_keys_for_anchor(anchor, window=window):
                for source in candidates_by_key.get(key, []):
                    image_id = source.get("chunk_id")
                    if not image_id or image_id in seen_ids:
                        continue
                    image_block = int(source.get("page_number") or 0)
                    block_score = _block_distance_score(
                        anchor.page_number, image_block, window=window
                    )
                    if block_score <= 0:
                        continue
                    score = anchor.score * block_score
                    if score < min_score:
                        continue
                    image = _image_from_hit(source, score, reason="proximity")
                    if image is not None:
                        seen_ids.add(image_id)
                        attached.append(image)
            if attached:
                attached.sort(key=lambda img: img["score"], reverse=True)
                attachments[anchor.chunk_id] = attached
        return attachments
    except Exception:
        logger.warning("DOCX proximity image attachment failed; continuing without", exc_info=True)
        return {}


def retrieve_docx_intent_images(
    client: OpenSearch,
    query: str,
    query_vector: list[float],
    *,
    user_id: int,
    doc_id: str | None = None,
) -> list[dict[str, Any]]:
    """Image-only hybrid search restricted to DOCX embedded image chunks."""
    if not settings.image_attach_enabled:
        return []
    try:
        response = hybrid_search(
            client,
            query_text=query,
            query_vector=query_vector,
            k=settings.image_intent_top_k,
            user_id=user_id,
            doc_id=doc_id,
            chunk_type="image",
            metadata_filters={"source_format": "docx"},
        )
        images: list[dict[str, Any]] = []
        for hit in response.get("hits", {}).get("hits", []):
            image = _image_from_hit(hit["_source"], hit.get("_score") or 0.0, reason="intent")
            if image is not None:
                images.append(image)
        return _gate_intent_images(images)
    except Exception:
        logger.warning("DOCX intent image retrieval failed; continuing without", exc_info=True)
        return []
