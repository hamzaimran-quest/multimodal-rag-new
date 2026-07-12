"""Image attachment: surface relevant images beside text answers (PDF only).

Two complementary tracks, both additive and never touching the LLM context:

* Track B (proximity) — for implicit relevance ("who is the chairman"): after
  hybrid retrieval, attach image chunks that sit next to a high-ranked text/table
  anchor on the same page. A hard column (x-overlap) gate rejects same-height but
  different-column content on multi-column layouts; vertical proximity then scores
  the survivors. Fail-closed below a minimum score.

* Track A (intent) — for explicit requests ("show me the chart"): a separate
  image-only retrieval pass, merged with priority over proximity.

Everything here is PDF-only for bbox proximity: DOCX uses block-index proximity in
``docx_image_attach.py`` instead.
"""

from __future__ import annotations

import logging
from typing import Any

from opensearchpy import OpenSearch

from app.config import settings
from app.opensearch.search import hybrid_search
from app.retrieval.models import RetrievedChunk
from app.retrieval.service import image_path_to_url

logger = logging.getLogger(__name__)

Bbox = list[float]


def _valid_bbox(bbox: Any) -> Bbox | None:
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            return [float(v) for v in bbox]
        except (TypeError, ValueError):
            return None
    return None


def _column_overlap_ratio(a: Bbox, b: Bbox) -> float:
    """Horizontal overlap normalized by the narrower box width (column awareness)."""
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    narrower = min(a[2] - a[0], b[2] - b[0])
    if narrower <= 0:
        return 0.0
    return overlap / narrower


def _vertical_gap(a: Bbox, b: Bbox) -> float:
    """Vertical gap between two boxes (0 if they overlap vertically). Top-origin."""
    return max(0.0, max(a[1], b[1]) - min(a[3], b[3]))


def bbox_iou(a: Bbox, b: Bbox) -> float:
    """Intersection-over-union of two boxes (used for image dedup)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _anchor_boxes(chunk: RetrievedChunk) -> list[Bbox]:
    """Per-line boxes when available (avoids the multi-column union gutter), else
    the chunk's union bbox."""
    extra = chunk.extra_metadata or {}
    lines = extra.get("line_bboxes")
    boxes: list[Bbox] = []
    if isinstance(lines, list):
        for line in lines:
            valid = _valid_bbox(line)
            if valid:
                boxes.append(valid)
    if boxes:
        return boxes
    union = _valid_bbox(chunk.bbox)
    return [union] if union else []


def _proximity_score(anchor: RetrievedChunk, image_bbox: Bbox) -> float:
    """Combined column-overlap × vertical-proximity score in [0, 1].

    Column gate is hard: any anchor line with insufficient x-overlap is skipped
    entirely, so a right-column image cannot attach to left-column text merely by
    sitting at the same height.
    """
    margin = settings.image_proximity_margin_px
    min_overlap = settings.image_proximity_column_overlap_min
    best = 0.0
    for box in _anchor_boxes(anchor):
        overlap = _column_overlap_ratio(box, image_bbox)
        if overlap < min_overlap:
            continue
        gap = _vertical_gap(box, image_bbox)
        if gap > margin:
            continue
        vertical_score = 1.0 - (gap / margin) if margin > 0 else 1.0
        score = overlap * vertical_score
        if score > best:
            best = score
    return best


def _image_from_hit(source: dict[str, Any], score: float, reason: str) -> dict[str, Any] | None:
    bbox = _valid_bbox(source.get("bbox"))
    image_url = image_path_to_url(source.get("image_path"))
    if image_url is None:
        return None
    extra = source.get("extra_metadata") or {}
    caption = extra.get("image_caption") or ""
    return {
        "image_chunk_id": source["chunk_id"],
        "doc_id": source.get("doc_id"),
        "filename": source.get("filename"),
        "page_number": source.get("page_number"),
        "image_url": image_url,
        "bbox": bbox,
        "caption": caption,
        "score": round(float(score), 4),
        "reason": reason,
        "source_format": extra.get("source_format", "pdf"),
        "section": extra.get("section"),
    }


def _image_from_chunk(chunk: RetrievedChunk, score: float, reason: str) -> dict[str, Any] | None:
    if not chunk.image_url:
        return None
    extra = chunk.extra_metadata or {}
    return {
        "image_chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "filename": chunk.filename,
        "page_number": chunk.page_number,
        "image_url": chunk.image_url,
        "bbox": _valid_bbox(chunk.bbox),
        "caption": extra.get("image_caption") or "",
        "score": round(float(score), 4),
        "reason": reason,
    }


def _select_anchors(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Top text/table hits with a bbox, score-gated relative to the top hit."""
    anchors = [
        c
        for c in chunks
        if c.chunk_type in {"text", "table"}
        and _valid_bbox(c.bbox) is not None
        and (c.extra_metadata or {}).get("source_format", "pdf") == "pdf"
    ]
    if not anchors:
        return []
    top_score = anchors[0].score
    ratio = settings.image_proximity_score_ratio
    threshold = top_score * ratio if top_score > 0 else float("-inf")
    gated = [c for c in anchors if c.score >= threshold]
    return gated[: settings.image_proximity_anchor_count]


def _fetch_candidate_images(
    client: OpenSearch, keys: set[str], user_id: int
) -> dict[str, list[dict[str, Any]]]:
    """One batched terms query: all image chunks sharing a page with any anchor."""
    if not keys:
        return {}
    body = {
        "size": max(len(keys) * 6, 24),
        "query": {
            "bool": {
                "filter": [
                    {"term": {"user_id": str(user_id)}},
                    {"term": {"chunk_type": "image"}},
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


def resolve_proximity_attachments(
    client: OpenSearch, chunks: list[RetrievedChunk], *, user_id: int
) -> dict[str, list[dict[str, Any]]]:
    """Map anchor chunk_id -> attached image dicts (fail-closed on any error)."""
    if not settings.image_attach_enabled:
        return {}
    try:
        anchors = _select_anchors(chunks)
        if not anchors:
            return {}

        keys = {f"{a.doc_id}:{a.page_number}" for a in anchors}
        candidates_by_key = _fetch_candidate_images(client, keys, user_id)
        if not candidates_by_key:
            return {}

        min_score = settings.image_min_attachment_score
        attachments: dict[str, list[dict[str, Any]]] = {}
        for anchor in anchors:
            key = f"{anchor.doc_id}:{anchor.page_number}"
            attached: list[dict[str, Any]] = []
            for source in candidates_by_key.get(key, []):
                image_bbox = _valid_bbox(source.get("bbox"))
                if image_bbox is None:
                    continue
                score = _proximity_score(anchor, image_bbox)
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
        logger.warning("Proximity image attachment failed; continuing without", exc_info=True)
        return {}


def retrieve_intent_images(
    client: OpenSearch,
    query: str,
    query_vector: list[float],
    *,
    user_id: int,
    doc_id: str | None = None,
    doc_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Image-only retrieval for the agent search_images tool."""
    effective_doc_ids = list(doc_ids) if doc_ids else ([doc_id] if doc_id else None)
    try:
        response = hybrid_search(
            client,
            query_text=query,
            query_vector=query_vector,
            k=settings.image_intent_top_k,
            user_id=user_id,
            doc_ids=effective_doc_ids,
            chunk_type="image",
        )
        images: list[dict[str, Any]] = []
        for hit in response.get("hits", {}).get("hits", []):
            image = _image_from_hit(hit["_source"], hit.get("_score") or 0.0, reason="intent")
            if image is not None:
                images.append(image)
        return images
    except Exception:
        logger.warning("Intent image retrieval failed; continuing without", exc_info=True)
        return []


def _is_duplicate(candidate: dict[str, Any], selected: list[dict[str, Any]], iou_thresh: float) -> bool:
    for existing in selected:
        if existing["image_chunk_id"] == candidate["image_chunk_id"]:
            return True
        same_page = (
            existing.get("doc_id") == candidate.get("doc_id")
            and existing.get("page_number") == candidate.get("page_number")
        )
        if same_page and existing.get("bbox") and candidate.get("bbox"):
            if bbox_iou(existing["bbox"], candidate["bbox"]) > iou_thresh:
                return True
    return False


def build_display_images(
    intent_images: list[dict[str, Any]],
    proximity_attachments: dict[str, list[dict[str, Any]]],
    *,
    max_display: int | None = None,
) -> list[dict[str, Any]]:
    """Deduped, capped hero list. Intent images win; proximity fills remainder."""
    cap = max_display if max_display is not None else settings.image_max_display
    iou_thresh = settings.image_dedup_iou
    selected: list[dict[str, Any]] = []

    for image in sorted(intent_images, key=lambda i: i["score"], reverse=True):
        if len(selected) >= cap:
            break
        if not _is_duplicate(image, selected, iou_thresh):
            selected.append(image)

    proximity_flat = [img for images in proximity_attachments.values() for img in images]
    for image in sorted(proximity_flat, key=lambda i: i["score"], reverse=True):
        if len(selected) >= cap:
            break
        if not _is_duplicate(image, selected, iou_thresh):
            selected.append(image)

    return selected
