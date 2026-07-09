"""DOCX embedded image extraction (separate from the PDF image pipeline).

Extracts inline and anchored raster drawings from paragraph blocks. Images share
the host paragraph's block ordinal and never receive viewer PDF geometry.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

from app.config import settings
from app.ingestion.chunking import normalize_whitespace
from app.ingestion.models import ExtractedChunk

logger = logging.getLogger(__name__)

OCR_PREVIEW_LIMIT = 500
NEARBY_TEXT_CHAR_LIMIT = 400
CONTENT_TYPE_EXT = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/gif": "gif",
    "image/bmp": "bmp",
    "image/tiff": "tif",
    "image/x-emf": "emf",
    "image/x-wmf": "wmf",
}


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


def _drawing_layout(drawing) -> str | None:
    if drawing.find(qn("wp:inline")) is not None:
        return "inline"
    if drawing.find(qn("wp:anchor")) is not None:
        return "anchored"
    return None


def _embedded_image_rel_ids(paragraph: Paragraph) -> list[tuple[str, str]]:
    """Return (relationship_id, layout) for raster blips in inline/anchored drawings."""
    rels: list[tuple[str, str]] = []
    seen: set[str] = set()
    for drawing in paragraph._element.findall(".//" + qn("w:drawing")):
        layout = _drawing_layout(drawing)
        if layout is None:
            continue
        for blip in drawing.findall(".//" + qn("a:blip")):
            rel_id = blip.get(qn("r:embed"))
            if not rel_id or rel_id in seen:
                continue
            seen.add(rel_id)
            rels.append((rel_id, layout))
    return rels


def _image_extension(content_type: str, partname: str) -> str:
    ext = CONTENT_TYPE_EXT.get(content_type.lower().strip())
    if ext:
        return ext
    suffix = Path(partname).suffix.lstrip(".").lower()
    return suffix or "png"


def _image_pixel_area(image_path: Path) -> int:
    try:
        from PIL import Image
    except Exception:
        return settings.docx_min_image_pixels
    try:
        with Image.open(image_path) as image:
            width, height = image.size
        return int(width) * int(height)
    except Exception:
        return 0


def _build_image_chunk_content(
    *,
    block_index: int,
    section: str | None,
    host_text: str,
    prev_text: str,
    next_text: str,
    ocr_text: str,
) -> str:
    parts: list[str] = [f"Part {block_index} image context."]
    if section:
        parts.append(f"Section: {section}")
    if host_text:
        parts.append(f"Host text: {host_text[:NEARBY_TEXT_CHAR_LIMIT]}")
    if prev_text:
        parts.append(f"Previous text: {prev_text[:NEARBY_TEXT_CHAR_LIMIT]}")
    if next_text:
        parts.append(f"Next text: {next_text[:NEARBY_TEXT_CHAR_LIMIT]}")
    if ocr_text:
        parts.append(f"OCR text: {ocr_text}")
    return "\n".join(parts)


def _base_metadata(block_index: int, section: str | None, drawing_layout: str) -> dict:
    extra: dict = {
        "source_format": "docx",
        "block_index": block_index,
        "docx_drawing_layout": drawing_layout,
    }
    if section:
        extra["section"] = section
    return extra


def extract_inline_image_chunks_for_paragraph(
    paragraph: Paragraph,
    *,
    block_index: int,
    section: str | None,
    prev_paragraph_text: str,
    next_paragraph_text: str,
    doc_id: str,
    user_id: int,
) -> list[ExtractedChunk]:
    """Extract embedded raster images from one paragraph block."""
    rels = _embedded_image_rel_ids(paragraph)
    if not rels:
        return []

    out_dir = settings.resolved_images_dir / str(user_id) / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[ExtractedChunk] = []
    part = paragraph.part
    host_text = normalize_whitespace(paragraph.text)

    for img_idx, (rel_id, drawing_layout) in enumerate(rels):
        try:
            image_part = part.related_parts[rel_id]
        except KeyError:
            logger.debug("Missing image relationship %s in doc_id=%s", rel_id, doc_id)
            continue

        content_type = getattr(image_part, "content_type", "") or ""
        ext = _image_extension(content_type, getattr(image_part, "partname", ""))
        image_filename = f"block{block_index}_img{img_idx}.{ext}"
        image_path = out_dir / image_filename

        try:
            image_path.write_bytes(image_part.blob)
        except Exception:
            logger.debug(
                "Failed to persist DOCX image block=%s idx=%s",
                block_index,
                img_idx,
                exc_info=True,
            )
            continue

        if _image_pixel_area(image_path) < settings.docx_min_image_pixels:
            image_path.unlink(missing_ok=True)
            continue

        ocr_text = _ocr_image_text(image_path)
        nearby_text = normalize_whitespace(
            " ".join(filter(None, [host_text, prev_paragraph_text, next_paragraph_text]))
        )
        if not nearby_text and not ocr_text:
            image_path.unlink(missing_ok=True)
            continue
        content = _build_image_chunk_content(
            block_index=block_index,
            section=section,
            host_text=host_text,
            prev_text=prev_paragraph_text,
            next_text=next_paragraph_text,
            ocr_text=ocr_text,
        )
        if not content.strip():
            image_path.unlink(missing_ok=True)
            continue

        caption_source = host_text or nearby_text or ocr_text
        relative_image_path = f"data/images/{user_id}/{doc_id}/{image_filename}"
        chunks.append(
            ExtractedChunk(
                content=content,
                page_number=block_index,
                chunk_type="image",
                extraction_method="docx_embedded",
                image_path=relative_image_path,
                extra_metadata={
                    **_base_metadata(block_index, section, drawing_layout),
                    "image_caption": caption_source[:200],
                    "ocr_available": bool(ocr_text),
                },
            )
        )

    return chunks
