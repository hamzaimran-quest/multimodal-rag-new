"""Build computed chart payloads from retrieved table chunks."""

from __future__ import annotations

from typing import Any

from app.charts.spec import validate_and_build_chart_spec
from app.retrieval.models import RetrievedChunk


def build_computed_charts(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    """
    Offer computed charts only for retrieved table chunks marked chartable at ingestion.

    Relevance is entirely determined by hybrid retrieval — never by query wording.
    """
    has_primary_image = any(chunk.chunk_type == "image" for chunk in chunks)
    charts: list[dict[str, Any]] = []

    for chunk in chunks:
        if chunk.chunk_type != "table":
            continue

        extra = chunk.extra_metadata or {}
        profile = extra.get("chart_profile")
        if not profile or not profile.get("chartable"):
            continue

        spec = validate_and_build_chart_spec(chunk.content, profile)
        if spec is None:
            continue

        charts.append(
            {
                **spec,
                "chunk_id": chunk.chunk_id,
                "filename": chunk.filename,
                "page_number": chunk.page_number,
                "doc_id": chunk.doc_id,
                "is_secondary": has_primary_image,
                "derivation": "computed",
                "citation": {
                    "chunk_id": chunk.chunk_id,
                    "filename": chunk.filename,
                    "page_number": chunk.page_number,
                    "chunk_type": "table",
                },
            }
        )

    return charts
