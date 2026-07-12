"""DOCX embedded image extraction (OCR-proxy indexing).

Body-order inline images only — table-cell images are skipped in v1. Keeps the
PDF image pipeline in ``images.py`` untouched; shares only private OCR helpers.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.config import settings
from app.ingestion.chunking import normalize_whitespace
from app.ingestion.docx_extract import _is_heading, _iter_block_items, _style_name
from app.ingestion.images import NEARBY_TEXT_WORD_LIMIT, _build_image_chunk_content, _ocr_image_text
from app.ingestion.models import ExtractedChunk

logger = logging.getLogger(__name__)

MIN_IMAGE_PIXELS = 2_500
NEARBY_BLOCK_RADIUS = 1


def _paragraph_image_blobs(paragraph: Paragraph) -> list[bytes]:
    """Return unique embedded image blobs from a body paragraph (inline drawings)."""
    blobs: list[bytes] = []
    seen_embeds: set[str] = set()
    for blip in paragraph._element.xpath('.//*[local-name()="blip"]'):
        embed = blip.get(qn("r:embed"))
        if not embed or embed in seen_embeds:
            continue
        seen_embeds.add(embed)
        try:
            blobs.append(paragraph.part.related_parts[embed].blob)
        except KeyError:
            logger.debug("DOCX image embed missing from related_parts: %s", embed)
    return blobs


def _save_image_blob(blob: bytes, image_path: Path) -> bool:
    try:
        from PIL import Image
    except Exception:
        return False
    try:
        with Image.open(io.BytesIO(blob)) as img:
            width, height = img.size
            if width * height < MIN_IMAGE_PIXELS:
                return False
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            img.save(image_path, format="PNG")
        return True
    except Exception:
        logger.debug("Failed to persist DOCX image %s", image_path, exc_info=True)
        return False


def _nearby_text(block_index: int, block_texts: dict[int, str], *, radius: int) -> str:
    parts: list[str] = []
    for idx in range(block_index - radius, block_index + radius + 1):
        text = block_texts.get(idx, "").strip()
        if text:
            parts.append(text)
    if not parts:
        return ""
    return normalize_whitespace(" ".join(parts))[:NEARBY_TEXT_WORD_LIMIT]


def _base_metadata(block_index: int, section: str | None) -> dict:
    extra: dict = {"source_format": "docx", "block_index": block_index}
    if section:
        extra["section"] = section
    return extra


def extract_docx_image_chunks(
    docx_path: str,
    *,
    doc_id: str | None = None,
    user_id: int | None = None,
) -> list[ExtractedChunk]:
    """Extract body-order embedded images as OCR-proxy image chunks."""
    if not doc_id or user_id is None:
        return []

    document = Document(docx_path)
    blocks = list(_iter_block_items(document))
    block_texts: dict[int, str] = {}
    block_index = 0
    current_section: str | None = None

    for block in blocks:
        block_index += 1
        if isinstance(block, Paragraph):
            if _is_heading(_style_name(block)):
                heading = normalize_whitespace(block.text)
                if heading:
                    current_section = heading
            text = normalize_whitespace(block.text)
            if text:
                block_texts[block_index] = text
        elif isinstance(block, Table):
            rows = [[normalize_whitespace(cell.text) for cell in row.cells] for row in block.rows]
            table_text = normalize_whitespace(" ".join(" ".join(row) for row in rows))
            if table_text:
                block_texts[block_index] = table_text

    out_dir = settings.resolved_images_dir / str(user_id) / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[ExtractedChunk] = []
    block_index = 0
    current_section = None

    for block in blocks:
        block_index += 1
        if isinstance(block, Paragraph):
            if _is_heading(_style_name(block)):
                heading = normalize_whitespace(block.text)
                if heading:
                    current_section = heading
        else:
            continue

        blobs = _paragraph_image_blobs(block)
        if not blobs:
            continue

        nearby = _nearby_text(block_index, block_texts, radius=NEARBY_BLOCK_RADIUS)
        for img_idx, blob in enumerate(blobs):
            image_filename = f"block{block_index}_img{img_idx}.png"
            image_path = out_dir / image_filename
            if not _save_image_blob(blob, image_path):
                continue

            ocr_text = _ocr_image_text(image_path)
            if not nearby and not ocr_text:
                image_path.unlink(missing_ok=True)
                continue

            content = _build_image_chunk_content(
                nearby_text=nearby,
                ocr_text=ocr_text,
                page_number=block_index,
            )
            relative_image_path = f"data/images/{user_id}/{doc_id}/{image_filename}"
            extra = _base_metadata(block_index, current_section)
            extra["image_caption"] = nearby[:200] if nearby else ""
            extra["ocr_available"] = bool(ocr_text)
            chunks.append(
                ExtractedChunk(
                    content=content,
                    page_number=block_index,
                    chunk_type="image",
                    extraction_method="docx_ocr_proxy",
                    image_path=relative_image_path,
                    extra_metadata=extra,
                )
            )

    logger.info("Extracted %s image chunks from DOCX doc_id=%s", len(chunks), doc_id)
    return chunks
