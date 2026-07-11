"""End-to-end document ingestion orchestration."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pdfplumber
from opensearchpy import OpenSearch

from app.config import settings
from app.ingestion.docx_bbox_lookup import locate_chunks_in_viewer_pdf
from app.ingestion.docx_extract import extract_docx_chunks
from app.ingestion.docx_render import render_docx_to_pdf
from app.ingestion.embeddings import embed_texts
from app.ingestion.text import extract_page_chunks
from app.opensearch.chunks import delete_chunks_for_document, index_chunks
from app.opensearch.documents import update_document_record
from app.ingestion.models import ExtractedChunk
from app.ingestion.xlsx_enrich import extract_xlsx_workbook
from app.ingestion.xlsx_extract import count_visible_sheets

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".xlsx")
VIEWER_PDF_NAME = "__viewer.pdf"


def document_upload_dir(user_id: int, doc_id: str) -> Path:
    return settings.resolved_uploads_dir / str(user_id) / doc_id


def viewer_pdf_path(user_id: int, doc_id: str) -> Path:
    return document_upload_dir(user_id, doc_id) / VIEWER_PDF_NAME


def save_upload_file(user_id: int, doc_id: str, filename: str, file_bytes: bytes) -> Path:
    dest_dir = document_upload_dir(user_id, doc_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    dest_path.write_bytes(file_bytes)
    return dest_path


def find_viewer_pdf_path(user_id: int, doc_id: str) -> Path | None:
    path = viewer_pdf_path(user_id, doc_id)
    return path if path.is_file() else None


def find_pdf_path(user_id: int, doc_id: str) -> Path | None:
    """Return the PDF served by the citation viewer (native upload or DOCX render)."""
    viewer = find_viewer_pdf_path(user_id, doc_id)
    if viewer is not None:
        return viewer

    dest_dir = document_upload_dir(user_id, doc_id)
    if not dest_dir.exists():
        return None
    pdfs = sorted(p for p in dest_dir.glob("*.pdf") if p.name != VIEWER_PDF_NAME)
    return pdfs[0] if pdfs else None


def find_document_path(user_id: int, doc_id: str, filename: str | None = None) -> Path | None:
    """Locate the uploaded source file for ingestion, not the viewer render."""
    dest_dir = document_upload_dir(user_id, doc_id)
    if not dest_dir.exists():
        return None

    if filename:
        candidate = dest_dir / filename
        if candidate.is_file():
            return candidate

    for ext in SUPPORTED_EXTENSIONS:
        matches = sorted(
            p
            for p in dest_dir.glob(f"*{ext}")
            if p.name != VIEWER_PDF_NAME and not p.name.endswith(".viewer.pdf")
        )
        if matches:
            return matches[0]
    return None


def remove_document_files(user_id: int, doc_id: str) -> None:
    upload_dir = document_upload_dir(user_id, doc_id)
    if upload_dir.exists():
        shutil.rmtree(upload_dir, ignore_errors=True)

    image_dir = settings.resolved_images_dir / str(user_id) / doc_id
    if image_dir.exists():
        shutil.rmtree(image_dir, ignore_errors=True)


def _extract_pdf_chunks(
    client: OpenSearch,
    doc_id: str,
    user_id: int,
    pdf_path: Path,
) -> list[ExtractedChunk]:
    extracted: list[ExtractedChunk] = []
    update_document_record(
        client,
        doc_id,
        ingestion_progress=5,
        progress_message="Parsing PDF pages",
    )
    with pdfplumber.open(str(pdf_path)) as pdf:
        total_pages = max(len(pdf.pages), 1)
        update_document_record(client, doc_id, page_count=len(pdf.pages))
        for index, page in enumerate(pdf.pages, start=1):
            page_chunks = extract_page_chunks(
                page, index, str(pdf_path), doc_id=doc_id, user_id=user_id
            )
            extracted.extend(page_chunks)
            parse_progress = 5 + (index / total_pages) * 50
            update_document_record(
                client,
                doc_id,
                ingestion_progress=parse_progress,
                progress_message=f"Parsed page {index}/{total_pages}",
            )
    return extracted


def _extract_docx_chunks(
    client: OpenSearch,
    doc_id: str,
    user_id: int,
    docx_path: Path,
) -> list[ExtractedChunk]:
    update_document_record(
        client,
        doc_id,
        ingestion_progress=8,
        progress_message="Parsing document",
    )
    extracted = extract_docx_chunks(str(docx_path), doc_id=doc_id, user_id=user_id)
    update_document_record(
        client,
        doc_id,
        ingestion_progress=25,
        progress_message="Parsed document",
    )
    return extracted


def _extract_xlsx_chunks(
    client: OpenSearch,
    doc_id: str,
    user_id: int,
    xlsx_path: Path,
) -> list[ExtractedChunk]:
    update_document_record(
        client,
        doc_id,
        ingestion_progress=8,
        progress_message="Parsing spreadsheet",
    )
    sheet_count = count_visible_sheets(str(xlsx_path))
    update_document_record(client, doc_id, page_count=sheet_count)
    update_document_record(
        client,
        doc_id,
        ingestion_progress=12,
        progress_message="Analyzing workbook schema",
    )
    extracted, schema = extract_xlsx_workbook(str(xlsx_path), doc_id=doc_id, user_id=user_id)
    update_document_record(
        client,
        doc_id,
        workbook_schema=schema.to_document_metadata(),
    )
    update_document_record(
        client,
        doc_id,
        ingestion_progress=55,
        progress_message="Parsed spreadsheet",
    )
    return extracted


def _prepare_docx_viewer(
    client: OpenSearch,
    doc_id: str,
    user_id: int,
    docx_path: Path,
    extracted: list[ExtractedChunk],
) -> None:
    update_document_record(
        client,
        doc_id,
        ingestion_progress=30,
        progress_message="Rendering preview PDF",
    )
    output_path = viewer_pdf_path(user_id, doc_id)
    rendered = render_docx_to_pdf(docx_path, output_path)
    if not rendered or not output_path.is_file():
        update_document_record(
            client,
            doc_id,
            ingestion_progress=45,
            progress_message="Preview PDF unavailable (LibreOffice not installed)",
        )
        return

    with pdfplumber.open(str(output_path)) as pdf:
        page_count = len(pdf.pages)
    update_document_record(
        client,
        doc_id,
        page_count=page_count,
        ingestion_progress=45,
        progress_message="Rendered preview PDF",
    )

    update_document_record(
        client,
        doc_id,
        ingestion_progress=48,
        progress_message="Locating citations in preview",
    )
    total_blocks = max((chunk.page_number for chunk in extracted), default=1)
    locate_chunks_in_viewer_pdf(
        extracted,
        output_path,
        total_blocks=total_blocks,
    )

    matched = sum(
        1
        for chunk in extracted
        if (chunk.extra_metadata.get("viewer_location") or {}).get("match_status") == "ok"
    )
    failed = len(extracted) - matched
    if failed:
        logger.warning(
            "DOCX viewer bbox lookup: %s/%s chunks matched for doc_id=%s",
            matched,
            len(extracted),
            doc_id,
        )

    update_document_record(
        client,
        doc_id,
        ingestion_progress=60,
        progress_message=f"Located citations in preview ({matched}/{len(extracted)} matched)",
    )


def run_ingestion(client: OpenSearch, user_id: int, doc_id: str, filename: str) -> int:
    """Extract, embed, and index a previously uploaded document. Returns chunk count."""
    source_path = find_document_path(user_id, doc_id, filename)
    if source_path is None:
        raise FileNotFoundError(f"No supported document found for doc_id={doc_id}")

    suffix = source_path.suffix.lower()

    update_document_record(
        client,
        doc_id,
        status="processing",
        ingestion_progress=2,
        progress_message="Queued",
        error_message="",
    )

    try:
        if suffix == ".pdf":
            extracted = _extract_pdf_chunks(client, doc_id, user_id, source_path)
        elif suffix == ".docx":
            extracted = _extract_docx_chunks(client, doc_id, user_id, source_path)
            _prepare_docx_viewer(client, doc_id, user_id, source_path, extracted)
        elif suffix == ".xlsx":
            extracted = _extract_xlsx_chunks(client, doc_id, user_id, source_path)
        else:
            raise ValueError(f"Unsupported document format: {suffix}")

        if not extracted:
            raise ValueError("No text or table content extracted from document")

        update_document_record(
            client,
            doc_id,
            ingestion_progress=65,
            progress_message="Generating embeddings",
        )
        embeddings = embed_texts([chunk.content for chunk in extracted])
        update_document_record(
            client,
            doc_id,
            ingestion_progress=85,
            progress_message="Indexing chunks",
        )
        delete_chunks_for_document(client, doc_id)
        count = index_chunks(
            client,
            doc_id=doc_id,
            user_id=user_id,
            filename=filename,
            chunks=extracted,
            embeddings=embeddings,
        )
        update_document_record(
            client,
            doc_id,
            status="indexed",
            ingestion_progress=100,
            progress_message="Completed",
            chunk_count=count,
        )
        logger.info("Indexed %s chunks for doc_id=%s", count, doc_id)
        return count
    except Exception as exc:
        logger.exception("Ingestion failed for doc_id=%s", doc_id)
        update_document_record(
            client,
            doc_id,
            status="failed",
            ingestion_progress=100,
            progress_message="Failed",
            error_message=str(exc),
        )
        raise
