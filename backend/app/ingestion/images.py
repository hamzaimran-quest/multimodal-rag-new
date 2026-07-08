"""Image extraction for image-chunk indexing (MVP OCR-proxy path)."""

from __future__ import annotations

import logging
from pathlib import Path

import pdfplumber

from app.config import settings
from app.ingestion.chunking import normalize_whitespace
from app.ingestion.models import ExtractedChunk

logger = logging.getLogger(__name__)

MIN_IMAGE_AREA = 8_000.0
OCR_PREVIEW_LIMIT = 500
NEARBY_TEXT_WORD_LIMIT = 80
VECTOR_OBJ_MIN_COUNT = 20
VECTOR_CLUSTER_MIN_AREA = 20_000.0
VECTOR_CLUSTER_MARGIN = 8.0


def _clamp_bbox(page: pdfplumber.page.Page, bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x0, top, x1, bottom = bbox
    x0 = max(0.0, min(float(page.width), float(x0)))
    x1 = max(0.0, min(float(page.width), float(x1)))
    top = max(0.0, min(float(page.height), float(top)))
    bottom = max(0.0, min(float(page.height), float(bottom)))
    if x1 < x0:
        x0, x1 = x1, x0
    if bottom < top:
        top, bottom = bottom, top
    return (x0, top, x1, bottom)


def _nearby_text_for_bbox(page: pdfplumber.page.Page, bbox: tuple[float, float, float, float]) -> str:
    x0, top, x1, bottom = bbox
    margin = 80.0
    words = []
    for word in page.extract_words() or []:
        wx0 = float(word.get("x0", 0.0))
        wx1 = float(word.get("x1", 0.0))
        wtop = float(word.get("top", 0.0))
        wbottom = float(word.get("bottom", 0.0))

        horizontally_close = wx1 >= (x0 - margin) and wx0 <= (x1 + margin)
        vertically_close = wbottom >= (top - margin) and wtop <= (bottom + margin)
        if horizontally_close and vertically_close:
            text = str(word.get("text", "")).strip()
            if text:
                words.append(text)
    if not words:
        return ""
    return normalize_whitespace(" ".join(words[:NEARBY_TEXT_WORD_LIMIT]))


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _expand_bbox(bbox: tuple[float, float, float, float], margin: float) -> tuple[float, float, float, float]:
    return (bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin)


def _bbox_intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _bbox_union(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _cluster_vector_bboxes(
    bboxes: list[tuple[float, float, float, float]],
) -> list[tuple[tuple[float, float, float, float], int]]:
    if not bboxes:
        return []
    visited = [False] * len(bboxes)
    clusters: list[tuple[tuple[float, float, float, float], int]] = []

    for i in range(len(bboxes)):
        if visited[i]:
            continue
        visited[i] = True
        cluster_bbox = bboxes[i]
        count = 1
        stack = [i]
        while stack:
            idx = stack.pop()
            target = _expand_bbox(bboxes[idx], VECTOR_CLUSTER_MARGIN)
            for j in range(len(bboxes)):
                if visited[j]:
                    continue
                if _bbox_intersects(target, _expand_bbox(bboxes[j], VECTOR_CLUSTER_MARGIN)):
                    visited[j] = True
                    stack.append(j)
                    cluster_bbox = _bbox_union(cluster_bbox, bboxes[j])
                    count += 1
        clusters.append((cluster_bbox, count))
    return clusters


def _collect_vector_chart_regions(
    page: pdfplumber.page.Page,
    *,
    excluded_bboxes: list[tuple[float, float, float, float]],
    taken_bboxes: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    objects: list[tuple[float, float, float, float]] = []
    for src in (page.rects or []):
        bbox = _clamp_bbox(page, (float(src.get("x0", 0.0)), float(src.get("top", 0.0)), float(src.get("x1", 0.0)), float(src.get("bottom", 0.0))))
        if _bbox_area(bbox) > 1.0:
            objects.append(bbox)
    for src in (page.lines or []):
        bbox = _clamp_bbox(page, (float(src.get("x0", 0.0)), float(src.get("top", 0.0)), float(src.get("x1", 0.0)), float(src.get("bottom", 0.0))))
        # Lines have near-zero area; inflate slightly so clustering works.
        bbox = _expand_bbox(bbox, 1.5)
        objects.append(_clamp_bbox(page, bbox))
    for src in (page.curves or []):
        bbox = _clamp_bbox(page, (float(src.get("x0", 0.0)), float(src.get("top", 0.0)), float(src.get("x1", 0.0)), float(src.get("bottom", 0.0))))
        if _bbox_area(bbox) > 1.0:
            objects.append(bbox)

    regions: list[tuple[float, float, float, float]] = []
    for cluster_bbox, count in _cluster_vector_bboxes(objects):
        area = _bbox_area(cluster_bbox)
        width = cluster_bbox[2] - cluster_bbox[0]
        height = cluster_bbox[3] - cluster_bbox[1]
        if count < VECTOR_OBJ_MIN_COUNT:
            continue
        if area < VECTOR_CLUSTER_MIN_AREA or width < 80 or height < 60:
            continue
        if any(_bbox_intersects(cluster_bbox, ex) for ex in excluded_bboxes):
            continue
        if any(_bbox_intersects(cluster_bbox, tk) for tk in taken_bboxes):
            continue
        regions.append(cluster_bbox)
    return regions


def _ocr_image_text(image_path: Path) -> str:
    """Best-effort OCR; returns empty string when OCR tooling is unavailable."""
    try:
        import pytesseract
        from PIL import Image
    except Exception:
        return ""

    try:
        text = pytesseract.image_to_string(Image.open(image_path))
    except Exception:
        logger.debug("OCR failed for %s", image_path, exc_info=True)
        return ""
    return normalize_whitespace(text)[:OCR_PREVIEW_LIMIT]


def _build_image_chunk_content(*, nearby_text: str, ocr_text: str, page_number: int) -> str:
    parts: list[str] = [f"Page {page_number} image/chart context."]
    if nearby_text:
        parts.append(f"Nearby text: {nearby_text}")
    if ocr_text:
        parts.append(f"OCR text: {ocr_text}")
    return "\n".join(parts)


def extract_image_chunks_for_page(
    page: pdfplumber.page.Page,
    *,
    page_number: int,
    doc_id: str | None,
    user_id: int | None = None,
    excluded_bboxes: list[tuple[float, float, float, float]] | None = None,
) -> list[ExtractedChunk]:
    if not doc_id or user_id is None:
        return []

    out_dir = settings.resolved_images_dir / str(user_id) / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[ExtractedChunk] = []
    excluded = excluded_bboxes or []
    taken_bboxes: list[tuple[float, float, float, float]] = []

    for idx, image in enumerate(page.images or []):
        x0 = float(image.get("x0", 0.0))
        x1 = float(image.get("x1", 0.0))
        top = float(image.get("top", 0.0))
        bottom = float(image.get("bottom", 0.0))
        bbox = _clamp_bbox(page, (x0, top, x1, bottom))
        width = max(0.0, bbox[2] - bbox[0])
        height = max(0.0, bbox[3] - bbox[1])
        area = width * height
        if area < MIN_IMAGE_AREA:
            continue
        if any(_bbox_intersects(bbox, ex) for ex in excluded):
            continue

        image_filename = f"page{page_number}_img{idx}.png"
        image_path = out_dir / image_filename
        try:
            cropped = page.crop(bbox)
            page_img = cropped.to_image(resolution=150)
            page_img.original.save(image_path)
        except Exception:
            logger.debug("Failed to persist image crop page=%s idx=%s", page_number, idx, exc_info=True)
            continue

        nearby_text = _nearby_text_for_bbox(page, bbox)
        ocr_text = _ocr_image_text(image_path)
        if not nearby_text and not ocr_text:
            # Skip decorative images with no retrievable signal.
            continue

        content = _build_image_chunk_content(
            nearby_text=nearby_text,
            ocr_text=ocr_text,
            page_number=page_number,
        )
        if not content.strip():
            continue

        relative_image_path = f"data/images/{user_id}/{doc_id}/{image_filename}"
        chunks.append(
            ExtractedChunk(
                content=content,
                page_number=page_number,
                chunk_type="image",
                extraction_method="ocr_proxy",
                bbox=[bbox[0], bbox[1], bbox[2], bbox[3]],
                image_path=relative_image_path,
                extra_metadata={
                    "image_caption": nearby_text[:200] if nearby_text else "",
                    "ocr_available": bool(ocr_text),
                },
            )
        )
        taken_bboxes.append(bbox)

    vector_regions = _collect_vector_chart_regions(
        page,
        excluded_bboxes=excluded,
        taken_bboxes=taken_bboxes,
    )
    for idx, bbox in enumerate(vector_regions):
        image_filename = f"page{page_number}_vec{idx}.png"
        image_path = out_dir / image_filename
        try:
            cropped = page.crop(bbox)
            page_img = cropped.to_image(resolution=150)
            page_img.original.save(image_path)
        except Exception:
            logger.debug("Failed to persist vector chart crop page=%s idx=%s", page_number, idx, exc_info=True)
            continue

        nearby_text = _nearby_text_for_bbox(page, bbox)
        ocr_text = _ocr_image_text(image_path)
        if not nearby_text and not ocr_text:
            continue

        content = _build_image_chunk_content(
            nearby_text=nearby_text,
            ocr_text=ocr_text,
            page_number=page_number,
        )
        relative_image_path = f"data/images/{user_id}/{doc_id}/{image_filename}"
        chunks.append(
            ExtractedChunk(
                content=content,
                page_number=page_number,
                chunk_type="image",
                extraction_method="ocr_proxy",
                bbox=[bbox[0], bbox[1], bbox[2], bbox[3]],
                image_path=relative_image_path,
                extra_metadata={
                    "image_caption": nearby_text[:200] if nearby_text else "",
                    "ocr_available": bool(ocr_text),
                    "vector_graphics_region": True,
                },
            )
        )

    return chunks
