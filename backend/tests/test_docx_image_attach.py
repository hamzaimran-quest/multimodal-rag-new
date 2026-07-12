"""Unit tests for DOCX block-index image proximity."""

from __future__ import annotations

from app.retrieval.docx_image_attach import (
    _block_proximity_score,
    _select_docx_anchors,
)
from app.retrieval.models import RetrievedChunk


def _docx_chunk(
    chunk_id: str,
    *,
    chunk_type: str,
    block_index: int,
    score: float,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="d1",
        filename="report.docx",
        page_number=block_index,
        chunk_type=chunk_type,
        content="Sample content for testing block proximity attachment logic.",
        score=score,
        extra_metadata={"source_format": "docx", "block_index": block_index},
    )


def test_block_proximity_score_within_radius() -> None:
    assert _block_proximity_score(5, 5, radius=2) == 1.0
    assert _block_proximity_score(5, 6, radius=2) == 0.5
    assert _block_proximity_score(5, 8, radius=2) == 0.0


def test_select_docx_anchors_score_gated() -> None:
    chunks = [
        _docx_chunk("t1", chunk_type="text", block_index=3, score=1.0),
        _docx_chunk("t2", chunk_type="table", block_index=4, score=0.85),
        _docx_chunk("t3", chunk_type="text", block_index=10, score=0.3),
    ]
    anchors = _select_docx_anchors(chunks)
    assert [a.chunk_id for a in anchors] == ["t1", "t2"]


def test_select_docx_anchors_ignores_pdf_chunks() -> None:
    pdf_chunk = RetrievedChunk(
        chunk_id="p1",
        doc_id="d1",
        filename="report.pdf",
        page_number=1,
        chunk_type="text",
        content="PDF text",
        score=1.0,
        bbox=[0, 0, 100, 20],
        extra_metadata={"source_format": "pdf"},
    )
    assert _select_docx_anchors([pdf_chunk]) == []
