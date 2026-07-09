"""Unit tests for DOCX intent image attachment."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.api.query import _merge_intent_image_sources
from app.retrieval.docx_image_attach import (
    _block_distance_score,
    resolve_docx_proximity_attachments,
    retrieve_docx_intent_images,
)
from app.retrieval.models import RetrievedChunk


def test_merge_intent_image_sources_preserves_docx_format() -> None:
    sources: list[dict] = []
    intent_images = [
        {
            "image_chunk_id": "img-1",
            "doc_id": "d1",
            "filename": "report.docx",
            "page_number": 4,
            "image_url": "/images/d1/block4_img0.png",
            "bbox": None,
            "caption": "Org chart",
            "score": 0.91,
            "source_format": "docx",
            "section": "Structure",
        }
    ]

    _merge_intent_image_sources(sources, intent_images, page_counts={"d1": 12})

    assert len(sources) == 1
    source = sources[0]
    assert source["source_format"] == "docx"
    assert source["section"] == "Structure"
    assert source["viewer_page"] is None
    assert source["bbox"] is None
    assert source["attach_reason"] == "intent"
    assert source["page_count"] == 12


@patch("app.retrieval.docx_image_attach.hybrid_search")
def test_retrieve_docx_intent_images_filters_docx_metadata(mock_search: MagicMock) -> None:
    mock_search.return_value = {
        "hits": {
            "hits": [
                {
                    "_score": 1.2,
                    "_source": {
                        "chunk_id": "c-img",
                        "doc_id": "d1",
                        "filename": "report.docx",
                        "page_number": 5,
                        "image_path": "data/images/1/d1/block5_img0.png",
                        "extra_metadata": {
                            "source_format": "docx",
                            "image_caption": "Leadership photo",
                            "section": "Team",
                        },
                    },
                }
            ]
        }
    }

    images = retrieve_docx_intent_images(
        MagicMock(),
        "show leadership photo",
        [0.1] * 8,
        user_id=1,
        doc_id="d1",
    )

    mock_search.assert_called_once()
    assert mock_search.call_args.kwargs["metadata_filters"] == {"source_format": "docx"}
    assert len(images) == 1
    assert images[0]["source_format"] == "docx"
    assert images[0]["section"] == "Team"
    assert images[0]["bbox"] is None


def test_block_distance_score_same_and_adjacent() -> None:
    assert _block_distance_score(51, 51, window=1) == 1.0
    assert _block_distance_score(51, 50, window=1) > 0.0
    assert _block_distance_score(51, 52, window=1) > 0.0
    assert _block_distance_score(51, 45, window=1) == 0.0


def _docx_text_anchor(chunk_id: str, block: int, score: float, content: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="d1",
        filename="report.docx",
        page_number=block,
        chunk_type="text",
        content=content,
        score=score,
        extra_metadata={"source_format": "docx", "block_index": block},
    )


@patch("app.retrieval.docx_image_attach._fetch_docx_candidate_images")
def test_resolve_docx_proximity_attachments_same_block(mock_fetch: MagicMock) -> None:
    mock_fetch.return_value = {
        "d1:51": [
            {
                "chunk_id": "img-arrow",
                "doc_id": "d1",
                "filename": "report.docx",
                "page_number": 51,
                "image_path": "data/images/1/d1/block51_img0.png",
                "extra_metadata": {
                    "source_format": "docx",
                    "attachment_key": "d1:51",
                    "image_caption": "arrow",
                },
            }
        ],
        "d1:45": [
            {
                "chunk_id": "img-dot",
                "doc_id": "d1",
                "filename": "report.docx",
                "page_number": 45,
                "image_path": "data/images/1/d1/block45_img0.png",
                "extra_metadata": {
                    "source_format": "docx",
                    "attachment_key": "d1:45",
                    "image_caption": "green dot",
                },
            }
        ],
    }
    anchors = [
        _docx_text_anchor("t-arrow", 51, 1.0, "An image of a left and right arrow is like:"),
        _docx_text_anchor("t-other", 45, 0.2, "Below is a centered green dot."),
    ]

    attachments = resolve_docx_proximity_attachments(MagicMock(), anchors, user_id=1)

    assert "t-arrow" in attachments
    assert attachments["t-arrow"][0]["image_chunk_id"] == "img-arrow"
    assert attachments["t-arrow"][0]["reason"] == "proximity"
