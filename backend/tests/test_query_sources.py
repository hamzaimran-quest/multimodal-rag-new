"""Tests for query source payload mapping."""

from __future__ import annotations

from app.api.query import _build_sources, _merge_intent_image_sources, _resolve_page_counts
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


def test_resolve_page_counts_includes_intent_doc_ids(monkeypatch) -> None:
    def fake_lookup(client, doc_id, user_id):
        return {"page_count": 42}

    monkeypatch.setattr("app.api.query.get_document_for_user", fake_lookup)

    counts = _resolve_page_counts(object(), [], user_id=1, extra_doc_ids={"doc-1"})
    assert counts == {"doc-1": 42}


def test_merge_intent_image_sources_gets_page_count() -> None:
    sources: list[dict] = []
    _merge_intent_image_sources(
        sources,
        [{"image_chunk_id": "img1", "doc_id": "d1", "filename": "a.pdf", "page_number": 1}],
        page_counts={"d1": 24},
    )
    assert len(sources) == 1
    assert sources[0]["page_count"] == 24
