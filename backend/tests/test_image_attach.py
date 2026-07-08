"""Unit tests for image attachment geometry, dedup, and merge logic."""

from __future__ import annotations

from app.retrieval.image_attach import (
    _column_overlap_ratio,
    _proximity_score,
    _select_anchors,
    _vertical_gap,
    bbox_iou,
    build_display_images,
)
from app.retrieval.models import RetrievedChunk


def _text_chunk(chunk_id: str, score: float, bbox, line_bboxes=None) -> RetrievedChunk:
    extra = {"source_format": "pdf"}
    if line_bboxes is not None:
        extra["line_bboxes"] = line_bboxes
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="d1",
        filename="report.pdf",
        page_number=3,
        chunk_type="text",
        content="Chairman of the Board is Jane Doe.",
        score=score,
        bbox=bbox,
        extra_metadata=extra,
    )


def test_column_overlap_and_vertical_gap() -> None:
    left = [50.0, 100.0, 250.0, 130.0]
    right = [300.0, 100.0, 500.0, 130.0]
    assert _column_overlap_ratio(left, right) == 0.0
    # Same column, stacked vertically.
    below = [60.0, 140.0, 240.0, 220.0]
    assert _column_overlap_ratio(left, below) > 0.5
    assert _vertical_gap(left, below) == 10.0


def test_proximity_rejects_other_column_same_height() -> None:
    # Left-column chairman text; right-column portrait at the same height.
    anchor = _text_chunk("t1", 1.0, [50.0, 100.0, 250.0, 200.0])
    right_column_image = [300.0, 100.0, 480.0, 260.0]
    assert _proximity_score(anchor, right_column_image) == 0.0


def test_proximity_accepts_same_column_nearby() -> None:
    anchor = _text_chunk("t1", 1.0, [50.0, 100.0, 250.0, 200.0])
    same_column_image = [60.0, 205.0, 240.0, 360.0]  # just below, overlapping x
    score = _proximity_score(anchor, same_column_image)
    assert score > 0.0


def test_select_anchors_score_gated() -> None:
    chunks = [
        _text_chunk("t1", 1.0, [0, 0, 100, 20]),
        _text_chunk("t2", 0.9, [0, 30, 100, 50]),
        _text_chunk("t3", 0.4, [0, 60, 100, 80]),  # below 0.7 * top -> excluded
    ]
    anchors = _select_anchors(chunks)
    ids = [a.chunk_id for a in anchors]
    assert ids == ["t1", "t2"]


def test_bbox_iou() -> None:
    a = [0.0, 0.0, 10.0, 10.0]
    b = [0.0, 0.0, 10.0, 10.0]
    assert bbox_iou(a, b) == 1.0
    c = [100.0, 100.0, 110.0, 110.0]
    assert bbox_iou(a, c) == 0.0


def _image(chunk_id: str, score: float, reason: str, bbox=None, page=3) -> dict:
    return {
        "image_chunk_id": chunk_id,
        "doc_id": "d1",
        "filename": "report.pdf",
        "page_number": page,
        "image_url": f"/images/d1/{chunk_id}.png",
        "bbox": bbox,
        "caption": "",
        "score": score,
        "reason": reason,
    }


def test_build_display_images_intent_wins_and_caps(monkeypatch) -> None:
    monkeypatch.setattr("app.retrieval.image_attach.settings.image_max_display", 2)
    intent = [_image("i1", 0.9, "intent"), _image("i2", 0.8, "intent")]
    proximity = {"t1": [_image("p1", 0.95, "proximity")]}
    result = build_display_images(intent, proximity)
    assert [img["image_chunk_id"] for img in result] == ["i1", "i2"]


def test_build_display_images_dedup_by_iou(monkeypatch) -> None:
    monkeypatch.setattr("app.retrieval.image_attach.settings.image_max_display", 3)
    monkeypatch.setattr("app.retrieval.image_attach.settings.image_dedup_iou", 0.6)
    intent = [_image("i1", 0.9, "intent", bbox=[0, 0, 100, 100])]
    # Same page, heavily overlapping crop of the same region -> treated as duplicate.
    proximity = {"t1": [_image("p1", 0.95, "proximity", bbox=[5, 5, 100, 100])]}
    result = build_display_images(intent, proximity)
    assert [img["image_chunk_id"] for img in result] == ["i1"]


def test_build_display_images_proximity_fills_remainder(monkeypatch) -> None:
    monkeypatch.setattr("app.retrieval.image_attach.settings.image_max_display", 2)
    intent = [_image("i1", 0.9, "intent", bbox=[0, 0, 10, 10])]
    proximity = {"t1": [_image("p1", 0.8, "proximity", bbox=[200, 200, 260, 260])]}
    result = build_display_images(intent, proximity)
    assert [img["image_chunk_id"] for img in result] == ["i1", "p1"]
