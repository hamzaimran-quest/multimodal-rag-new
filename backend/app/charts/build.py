"""Unified chart construction for tool-driven workflows (QuickChart + aux LLM)."""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from app.charts.llm_config import extract_chart_data_spec
from app.charts.quickchart import (
    build_quickchart_url,
    chartjs_config_from_data_spec,
    chart_title_from_config,
)
from app.charts.structural import build_chart_data_spec_from_structure
from app.ingestion.tables import table_to_markdown
from app.ingestion.xlsx_serialize import table_rows_from_chunk_content
from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)

ChartType = Literal["bar", "line"]
_VALID_CHART_TYPES = {"bar", "line"}


def chart_payload_from_chunk(
    chunk: RetrievedChunk,
    spec: dict[str, Any],
    *,
    derivation: str = "tool",
) -> dict[str, Any]:
    """Wrap a chart spec with chunk citation metadata for the API."""
    return {
        **spec,
        "chunk_id": chunk.chunk_id,
        "filename": chunk.filename,
        "page_number": chunk.page_number,
        "doc_id": chunk.doc_id,
        "derivation": derivation,
        "citation": {
            "chunk_id": chunk.chunk_id,
            "filename": chunk.filename,
            "page_number": chunk.page_number,
            "chunk_type": chunk.chunk_type,
        },
    }


def attempt_chart_from_chunk(
    chunk: RetrievedChunk,
    *,
    user_query: str,
    chart_type: ChartType | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Build a QuickChart URL from a table chunk: structural profiling first, LLM fallback.

    Returns (chart_payload, error_message). On success error_message is None.
    """
    if chunk.chunk_type != "table":
        return None, "The selected chunk is not a table."

    extra = chunk.extra_metadata or {}
    rows = table_rows_from_chunk_content(chunk.content, extra)
    markdown = (
        table_to_markdown(rows)
        if extra.get("content_format") == "slim_rows" and rows
        else chunk.content
    )
    if not markdown.strip():
        return None, "The table chunk has no readable content."

    logger.info(
        "CHART_BUILD input chunk_id=%s page=%s md_chars=%s md_preview=%r",
        chunk.chunk_id,
        chunk.page_number,
        len(markdown),
        markdown[:300],
    )

    data_spec = build_chart_data_spec_from_structure(
        markdown,
        user_query=user_query,
        chart_type=chart_type,
        extra_metadata=extra,
    )
    spec_source = "structural"
    llm_error: str | None = None
    if data_spec is None:
        spec_source = "llm"
        data_spec, llm_error = extract_chart_data_spec(
            user_query,
            markdown,
            chart_type=chart_type,
        )
    else:
        logger.info(
            "CHART_BUILD structural ok chunk_id=%s type=%s labels=%s series=%s spec=%s",
            chunk.chunk_id,
            data_spec.get("chart_type"),
            len(data_spec.get("labels", [])),
            len(data_spec.get("series", [])),
            json.dumps(data_spec, ensure_ascii=False)[:2000],
        )

    if data_spec is None:
        return None, llm_error or "Could not build a chart configuration from this table."

    logger.info("CHART_BUILD spec_source=%s chunk_id=%s", spec_source, chunk.chunk_id)

    config = chartjs_config_from_data_spec(data_spec, chart_type_hint=chart_type)
    logger.info(
        "CHART_BUILD chartjs chunk_id=%s config=%s",
        chunk.chunk_id,
        json.dumps(config, ensure_ascii=False)[:2000],
    )

    try:
        chart_url = build_quickchart_url(config)
    except Exception:
        logger.warning("QuickChart URL generation failed chunk_id=%s", chunk.chunk_id, exc_info=True)
        return None, "Failed to generate chart image from configuration."

    resolved_type = str(config.get("type") or chart_type or "bar").strip().lower()
    if resolved_type not in _VALID_CHART_TYPES:
        resolved_type = "bar"

    spec = {
        "chart_type": resolved_type,
        "chart_url": chart_url,
        "title": chart_title_from_config(config) or str(data_spec.get("title") or "").strip() or None,
        "periods": [str(label) for label in data_spec["labels"]],
        "series": [
            {"name": str(entry["name"]), "values": [float(value) for value in entry["values"]]}
            for entry in data_spec["series"]
        ],
    }
    return chart_payload_from_chunk(chunk, spec, derivation="tool"), None


def merge_chart_outputs(*chart_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge chart lists; later lists override earlier entries for the same chunk_id."""
    merged: dict[str, dict[str, Any]] = {}
    for charts in chart_lists:
        for chart in charts:
            merged[chart["chunk_id"]] = chart
    return list(merged.values())
