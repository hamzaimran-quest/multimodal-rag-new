"""Tests for query source payload mapping."""

from __future__ import annotations

from app.api.query import (
    _build_sources,
    _enrich_docx_image_sources_with_viewer_page,
    _image_source_snippet,
)
from app.retrieval.models import RetrievedChunk


def test_build_sources_maps_docx_viewer_location() -> None:
    chunk = RetrievedChunk(
        chunk_id="c1",
        doc_id="d1",
        filename="report.docx",
        page_number=12,
        chunk_type="text",
        content="Sample content for citation mapping.",
        score=1.0,
        extra_metadata={
            "source_format": "docx",
            "section": "Highlights",
            "viewer_location": {
                "match_status": "ok",
                "viewer_page": 4,
                "bbox": [10.0, 20.0, 100.0, 40.0],
                "line_bboxes": [[10.0, 20.0, 100.0, 30.0]],
            },
        },
    )

    sources = _build_sources([chunk], page_counts={"d1": 18})
    assert len(sources) == 1
    source = sources[0]
    assert source["page_number"] == 12
    assert source["viewer_page"] == 4
    assert source["section"] == "Highlights"
    assert source["bbox"] == [10.0, 20.0, 100.0, 40.0]
    assert source["page_count"] == 18


def test_build_sources_omits_bbox_when_docx_match_failed() -> None:
    chunk = RetrievedChunk(
        chunk_id="c2",
        doc_id="d1",
        filename="report.docx",
        page_number=3,
        chunk_type="text",
        content="Unmatched chunk text.",
        score=0.5,
        extra_metadata={
            "source_format": "docx",
            "viewer_location": {"match_status": "failed"},
        },
    )

    source = _build_sources([chunk])[0]
    assert source["viewer_page"] is None
    assert source["bbox"] is None
    assert source["line_bboxes"] is None


def test_image_source_snippet_uses_caption_not_part_prefix() -> None:
    chunk = RetrievedChunk(
        chunk_id="img-1",
        doc_id="d1",
        filename="demo.docx",
        page_number=51,
        chunk_type="image",
        content="Part 51 image context.\nSection: Images\nHost text: An image of a left and right arrow is like:",
        score=0.8,
        image_url="/images/d1/block51_img0.png",
        extra_metadata={
            "source_format": "docx",
            "image_caption": "An image of a left and right arrow is like:",
        },
    )

    assert _image_source_snippet(chunk) == "An image of a left and right arrow is like:"

    sources = _build_sources([chunk])
    assert sources[0]["snippet"] == "An image of a left and right arrow is like:"
    assert "Part 51" not in sources[0]["snippet"]


def test_enrich_docx_image_sources_copies_viewer_page_from_neighbor_block() -> None:
    sources = [
        {
            "chunk_id": "t1",
            "doc_id": "d1",
            "page_number": 51,
            "chunk_type": "text",
            "source_format": "docx",
            "viewer_page": 4,
        },
        {
            "chunk_id": "img-1",
            "doc_id": "d1",
            "page_number": 51,
            "chunk_type": "image",
            "source_format": "docx",
            "viewer_page": None,
        },
    ]

    _enrich_docx_image_sources_with_viewer_page(sources)
    assert sources[1]["viewer_page"] == 4
