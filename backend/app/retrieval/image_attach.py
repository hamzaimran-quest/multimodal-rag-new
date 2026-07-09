"""Image attachment: surface relevant images beside text answers (PDF only).

Two complementary tracks, both additive and never touching the LLM context:

* Track B (proximity) — for implicit relevance ("who is the chairman"): after
  hybrid retrieval, attach image chunks that sit next to a high-ranked text/table
  anchor on the same page. A hard column (x-overlap) gate rejects same-height but
  different-column content on multi-column layouts; vertical proximity then scores
  the survivors. Fail-closed below a minimum score.

* Track A (intent) — for explicit requests ("show me the chart"): a separate
  image-only retrieval pass, merged with priority over proximity.

Everything here is PDF-only: DOCX Phase 1 indexes no image chunks and no bbox, so
there is nothing to attach for DOCX documents.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from opensearchpy import OpenSearch

from app.config import settings
from app.opensearch.search import hybrid_search
from app.retrieval.models import RetrievedChunk
from app.retrieval.service import image_path_to_url

logger = logging.getLogger(__name__)

Bbox = list[float]

_TOKEN_RE = re.compile(r"[a-zA-Z]{4,}")
# PDF points² — portraits are moderate; full-page covers are much larger.
_PORTRAIT_MAX_AREA_PDF_PT2 = 200_000.0


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
    strict = _proximity_score_strict(anchor, image_bbox)
    if strict > 0:
        return strict
    return _proximity_score_portrait_relaxed(anchor, image_bbox)


def _proximity_score_strict(anchor: RetrievedChunk, image_bbox: Bbox) -> float:
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


def _proximity_score_portrait_relaxed(anchor: RetrievedChunk, image_bbox: Bbox) -> float:
    """Same-page portrait in another column: vertical alignment only, reduced score."""
    width = max(0.0, image_bbox[2] - image_bbox[0])
    height = max(0.0, image_bbox[3] - image_bbox[1])
    if height <= width * 1.1 or height * width > _PORTRAIT_MAX_AREA_PDF_PT2:
        return 0.0

    margin = settings.image_proximity_margin_px
    best = 0.0
    for box in _anchor_boxes(anchor):
        gap = _vertical_gap(box, image_bbox)
        if gap > margin:
            continue
        vertical_score = 1.0 - (gap / margin) if margin > 0 else 1.0
        score = vertical_score * 0.5
        if score > best:
            best = score
    return best


def _spread_page_score(anchor: RetrievedChunk, image_page: int) -> float:
    """Decay score for images on nearby pages of the same document (letter spreads)."""
    spread = settings.image_proximity_spread_pages
    if spread <= 0:
        return 0.0
    dist = abs(int(anchor.page_number) - int(image_page))
    if dist == 0 or dist > spread:
        return 0.0
    decay = 1.0 - dist / (spread + 1)
    anchor_weight = max(settings.image_min_attachment_score, min(float(anchor.score), 1.0))
    return decay * anchor_weight


def _attachment_keys_for_anchors(anchors: list[RetrievedChunk]) -> set[str]:
    """Same-page plus ±N spread pages per anchor for batched image lookup."""
    spread = settings.image_proximity_spread_pages
    keys: set[str] = set()
    for anchor in anchors:
        page = int(anchor.page_number)
        doc_id = anchor.doc_id
        for offset in range(spread + 1):
            if offset == 0:
                candidates = [page]
            else:
                candidates = [page - offset, page + offset]
            for candidate in candidates:
                if candidate > 0:
                    keys.add(f"{doc_id}:{candidate}")
    return keys


def _content_tokens(text: str) -> set[str]:
    """Alphabetic tokens length ≥ 4 for cross-chunk overlap (no curated vocab)."""
    return {token.lower() for token in _TOKEN_RE.findall(text or "")}


def _token_overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _layout_score_adjustment(bbox: Bbox | None) -> float:
    """Geometry-only tie-breaker: penalize thin strips, boost moderate portraits."""
    if bbox is None:
        return 0.0
    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])
    if width <= 0 or height <= 0:
        return 0.0
    area = width * height
    aspect = width / height
    adjustment = 0.0
    if aspect >= settings.image_intent_banner_aspect_threshold:
        adjustment -= settings.image_intent_banner_aspect_penalty
    elif height > width * 1.1 and area <= _PORTRAIT_MAX_AREA_PDF_PT2:
        adjustment += settings.image_intent_portrait_aspect_boost
    return adjustment


def rerank_intent_images(
    images: list[dict[str, Any]],
    *,
    query: str,
    text_chunks: list[RetrievedChunk] | None = None,
) -> list[dict[str, Any]]:
    """Fuse hybrid image scores with text-retrieval context and layout signals."""
    if not images:
        return []

    text_chunks = text_chunks or []
    context_parts = [query]
    anchors = [
        chunk
        for chunk in text_chunks
        if chunk.chunk_type in {"text", "table"} and (chunk.content or "").strip()
    ]
    context_parts.extend(chunk.content for chunk in anchors[:5])
    context_tokens = _content_tokens(" ".join(context_parts))

    reranked: list[dict[str, Any]] = []
    for image in images:
        score = float(image.get("score", 0.0))
        image_tokens = _content_tokens(
            " ".join(
                part
                for part in (image.get("caption") or "", image.get("page_context") or "")
                if part
            )
        )
        if context_tokens and image_tokens:
            score += _token_overlap_ratio(image_tokens, context_tokens) * settings.image_intent_text_boost_max

        image_page = image.get("page_number")
        if image_page is not None:
            page_boost = 0.0
            for anchor in anchors[: settings.image_proximity_anchor_count]:
                if anchor.doc_id != image.get("doc_id"):
                    continue
                spread = settings.image_proximity_spread_pages
                dist = abs(int(anchor.page_number) - int(image_page))
                if dist > spread:
                    continue
                decay = 1.0 if dist == 0 else 1.0 - dist / (spread + 1)
                page_boost = max(
                    page_boost,
                    decay * min(float(anchor.score), 1.0) * settings.image_intent_page_boost_max,
                )
            score += page_boost

        score += _layout_score_adjustment(_valid_bbox(image.get("bbox")))
        updated = dict(image)
        updated["score"] = round(max(0.0, score), 4)
        reranked.append(updated)

    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked


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
        "page_context": extra.get("page_context") or "",
        "score": round(float(score), 4),
        "reason": reason,
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
        "page_context": extra.get("page_context") or "",
        "score": round(float(score), 4),
        "reason": reason,
    }


def _select_anchors(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Top text/table hits with a bbox, score-gated relative to the top hit."""
    anchors = [
        c
        for c in chunks
        if c.chunk_type in {"text", "table"} and _valid_bbox(c.bbox) is not None
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
                    {"terms": {"extra_metadata.attachment_key.keyword": sorted(keys)}},
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

        keys = _attachment_keys_for_anchors(anchors)
        candidates_by_key = _fetch_candidate_images(client, keys, user_id)
        if not candidates_by_key:
            return {}

        min_score = settings.image_min_attachment_score
        attachments: dict[str, list[dict[str, Any]]] = {}
        for anchor in anchors:
            attached: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            for key, sources in candidates_by_key.items():
                key_doc, _, key_page_str = key.partition(":")
                if key_doc != anchor.doc_id:
                    continue
                try:
                    key_page = int(key_page_str)
                except ValueError:
                    continue
                for source in sources:
                    chunk_id = source.get("chunk_id")
                    if not chunk_id or chunk_id in seen_ids:
                        continue
                    image_bbox = _valid_bbox(source.get("bbox"))
                    image_page = int(source.get("page_number") or key_page)
                    if image_page == int(anchor.page_number) and image_bbox is not None:
                        score = _proximity_score(anchor, image_bbox)
                    else:
                        score = _spread_page_score(anchor, image_page)
                    if score < min_score:
                        continue
                    image = _image_from_hit(source, score, reason="proximity")
                    if image is not None:
                        attached.append(image)
                        seen_ids.add(chunk_id)
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
    text_chunks: list[RetrievedChunk] | None = None,
) -> list[dict[str, Any]]:
    """Track A: image-only retrieval pass for explicit visual-intent queries."""
    try:
        response = hybrid_search(
            client,
            query_text=query,
            query_vector=query_vector,
            k=settings.image_intent_top_k,
            user_id=user_id,
            doc_id=doc_id,
            chunk_type="image",
            exclude_metadata={"source_format": "docx"},
        )
        images: list[dict[str, Any]] = []
        for hit in response.get("hits", {}).get("hits", []):
            image = _image_from_hit(hit["_source"], hit.get("_score") or 0.0, reason="intent")
            if image is not None:
                images.append(image)
        images = rerank_intent_images(images, query=query, text_chunks=text_chunks)
        return _gate_intent_images(images)
    except Exception:
        logger.warning("Intent image retrieval failed; continuing without", exc_info=True)
        return []


def _gate_intent_images(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop loosely-matching images: keep only those near the best score, and only
    if the best clears an absolute floor. Fail-closed (empty) when nothing qualifies."""
    if not images:
        return []
    best = max(img["score"] for img in images)
    if best < settings.image_intent_min_score:
        return []
    threshold = best * settings.image_intent_score_ratio
    gated = [img for img in images if img["score"] >= threshold]
    gated.sort(key=lambda item: item["score"], reverse=True)
    return gated


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
) -> list[dict[str, Any]]:
    """Deduped, capped hero list. Intent images win; proximity fills remainder."""
    cap = settings.image_max_display
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
