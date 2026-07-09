"""Unit tests for DOCX embedded image extraction."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from docx import Document
from PIL import Image

from app.ingestion.docx_extract import extract_docx_chunks
from app.ingestion.docx_images import (
    _embedded_image_rel_ids,
    extract_inline_image_chunks_for_paragraph,
)


def _write_test_png(path: Path, *, width: int = 120, height: int = 120) -> None:
    Image.new("RGB", (width, height), color=(30, 90, 180)).save(path, format="PNG")


def _build_docx_with_image(
    docx_path: Path,
    image_path: Path,
    *,
    include_neighbors: bool = True,
) -> None:
    document = Document()
    if include_neighbors:
        document.add_paragraph(
            "Previous paragraph about the company chairman and leadership team "
            "with enough words here for retrieval context."
        )
    paragraph = document.add_paragraph()
    run = paragraph.add_run()
    run.add_picture(str(image_path))
    if include_neighbors:
        document.add_paragraph(
            "Next paragraph describes quarterly revenue results and financial "
            "performance metrics clearly for investors and analysts."
        )
    document.save(docx_path)


@pytest.fixture
def sample_image_path(tmp_path: Path) -> Path:
    path = tmp_path / "sample.png"
    _write_test_png(path)
    return path


def test_embedded_image_rel_ids_finds_inline_drawing(sample_image_path: Path, tmp_path: Path) -> None:
    docx_path = tmp_path / "sample.docx"
    _build_docx_with_image(docx_path, sample_image_path)

    document = Document(docx_path)
    paragraph = document.paragraphs[1]
    rels = _embedded_image_rel_ids(paragraph)
    assert len(rels) == 1
    assert rels[0][1] == "inline"


def test_extract_inline_image_chunks_persists_image(
    sample_image_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docx_path = tmp_path / "sample.docx"
    _build_docx_with_image(docx_path, sample_image_path)
    images_dir = tmp_path / "images"
    monkeypatch.setattr("app.ingestion.docx_images.settings.images_dir", images_dir)

    document = Document(docx_path)
    paragraph = document.paragraphs[1]
    chunks = extract_inline_image_chunks_for_paragraph(
        paragraph,
        block_index=2,
        section="Leadership",
        prev_paragraph_text="Chairman portrait section with enough words.",
        next_paragraph_text="Quarterly revenue growth exceeded analyst expectations.",
        doc_id="doc-1",
        user_id=7,
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_type == "image"
    assert chunk.extraction_method == "docx_embedded"
    assert chunk.page_number == 2
    assert chunk.bbox is None
    assert chunk.extra_metadata["source_format"] == "docx"
    assert chunk.extra_metadata["block_index"] == 2
    assert chunk.extra_metadata["section"] == "Leadership"
    assert chunk.image_path is not None
    assert chunk.image_path.endswith("block2_img0.png")
    assert (images_dir / "7" / "doc-1" / "block2_img0.png").is_file()
    assert "Chairman portrait" in chunk.content
    assert "Quarterly revenue" in chunk.content


def test_extract_inline_image_chunks_includes_host_paragraph_text(
    sample_image_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docx_path = tmp_path / "arrow.docx"
    document = Document()
    document.add_paragraph("Previous paragraph with enough words for retrieval context here.")
    paragraph = document.add_paragraph("An image of a left and right arrow is like:")
    paragraph.add_run().add_picture(str(sample_image_path))
    document.add_paragraph("Next paragraph with enough words for retrieval context here.")
    document.save(docx_path)

    images_dir = tmp_path / "images"
    monkeypatch.setattr("app.ingestion.docx_images.settings.images_dir", images_dir)

    document = Document(docx_path)
    paragraph = document.paragraphs[1]
    chunks = extract_inline_image_chunks_for_paragraph(
        paragraph,
        block_index=2,
        section="Images",
        prev_paragraph_text="Previous paragraph with enough words for retrieval context here.",
        next_paragraph_text="Next paragraph with enough words for retrieval context here.",
        doc_id="doc-arrow",
        user_id=1,
    )

    assert len(chunks) == 1
    assert "Host text:" in chunks[0].content
    assert "left and right arrow" in chunks[0].content


def test_extract_inline_image_chunks_skips_decorative_without_signal(
    sample_image_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docx_path = tmp_path / "isolated.docx"
    _build_docx_with_image(docx_path, sample_image_path, include_neighbors=False)
    images_dir = tmp_path / "images"
    monkeypatch.setattr("app.ingestion.docx_images.settings.images_dir", images_dir)
    monkeypatch.setattr("app.ingestion.docx_images._ocr_image_text", lambda _path: "")

    document = Document(docx_path)
    chunks = extract_inline_image_chunks_for_paragraph(
        document.paragraphs[0],
        block_index=1,
        section=None,
        prev_paragraph_text="",
        next_paragraph_text="",
        doc_id="doc-2",
        user_id=1,
    )
    assert chunks == []
    assert not list((images_dir / "1" / "doc-2").glob("*"))


def test_extract_docx_small_inline_image_below_pdf_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """32x32 inline icons (e.g. green dots) are below PDF's 8000px² gate but valid for DOCX."""
    image_path = tmp_path / "dot.png"
    Image.new("RGBA", (32, 32), color=(0, 200, 0, 255)).save(image_path, format="PNG")

    docx_path = tmp_path / "dot.docx"
    document = Document()
    document.add_paragraph(
        "Previous paragraph with enough words about inline icons and formatting markers."
    )
    paragraph = document.add_paragraph()
    paragraph.add_run().add_picture(str(image_path))
    document.add_paragraph(
        "Next paragraph explains that the green dot marks a special formatting case."
    )
    document.save(docx_path)

    images_dir = tmp_path / "images"
    monkeypatch.setattr("app.ingestion.docx_images.settings.images_dir", images_dir)
    monkeypatch.setattr("app.ingestion.docx_images.settings.docx_min_image_pixels", 256)
    monkeypatch.setattr("app.ingestion.docx_images._ocr_image_text", lambda _path: "")

    chunks = extract_docx_chunks(str(docx_path), doc_id="doc-dot", user_id=1)
    image_chunks = [chunk for chunk in chunks if chunk.chunk_type == "image"]
    assert len(image_chunks) == 1
    assert image_chunks[0].page_number == 2


def test_extract_docx_chunks_includes_image_without_advancing_block_index(
    sample_image_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docx_path = tmp_path / "sample.docx"
    _build_docx_with_image(docx_path, sample_image_path)
    images_dir = tmp_path / "images"
    monkeypatch.setattr("app.ingestion.docx_images.settings.images_dir", images_dir)

    chunks = extract_docx_chunks(str(docx_path), doc_id="doc-3", user_id=3)
    image_chunks = [chunk for chunk in chunks if chunk.chunk_type == "image"]
    text_chunks = [chunk for chunk in chunks if chunk.chunk_type == "text"]

    assert len(image_chunks) == 1
    assert image_chunks[0].page_number == 2
    assert {chunk.page_number for chunk in text_chunks} == {1, 3}
    assert max(chunk.page_number for chunk in chunks) == 3
