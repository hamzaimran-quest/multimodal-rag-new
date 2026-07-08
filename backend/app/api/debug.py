"""Debug endpoints for inspecting indexed chunks directly (no LLM)."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/debug", tags=["debug"])


class TableInspectResponse(BaseModel):
    total_chunks: int
    findings: list[dict[str, Any]]


def _parse_markdown_table(content: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
    return rows


@router.get("/table-inspect", response_model=TableInspectResponse)
async def inspect_table_chunk(
    request: Request,
    page_number: int = Query(..., ge=1),
    filename: str | None = Query(default=None),
    doc_id: str | None = Query(default=None),
    metric: str = Query(default="operating margin"),
) -> TableInspectResponse:
    """Inspect indexed table chunk(s) directly from OpenSearch and log alignment diagnostics."""
    if not filename and not doc_id:
        raise HTTPException(status_code=400, detail="Provide either filename or doc_id")

    filters: list[dict[str, Any]] = [
        {"term": {"page_number": page_number}},
        {"term": {"chunk_type": "table"}},
    ]
    if filename:
        filters.append({"term": {"filename": filename}})
    if doc_id:
        filters.append({"term": {"doc_id": doc_id}})

    client = request.app.state.opensearch
    result = client.search(
        index=settings.chunks_index,
        body={"size": 20, "query": {"bool": {"filter": filters}}},
    )
    hits = result["hits"]["hits"]
    findings: list[dict[str, Any]] = []

    for hit in hits:
        src = hit["_source"]
        chunk_id = src.get("chunk_id")
        content = src.get("content", "")
        rows = _parse_markdown_table(content)
        table_qa = (src.get("extra_metadata") or {}).get("table_qa") or {}
        has_empty_first_col = all((row[0] if row else "") == "" for row in rows[2:]) if len(rows) > 2 else False
        metric_rows = [r for r in rows if any(metric.lower() in c.lower() for c in r)]

        finding = {
            "chunk_id": chunk_id,
            "filename": src.get("filename"),
            "page_number": src.get("page_number"),
            "row_count": len(rows),
            "has_empty_first_column_after_headers": has_empty_first_col,
            "table_qa": table_qa,
            "metric_row_found": bool(metric_rows),
            "metric_rows": metric_rows,
            "content_preview": content[:2000],
        }
        findings.append(finding)

        logger.info(
            "TABLE_INSPECT chunk=%s page=%s metric=%s found=%s empty_first_col=%s rows=%s",
            chunk_id,
            src.get("page_number"),
            metric,
            bool(metric_rows),
            has_empty_first_col,
            len(rows),
        )
        if table_qa:
            logger.info(
                "TABLE_INSPECT_QA chunk=%s misalignment=%s merged_header=%s label_col=%s fallback=%s",
                chunk_id,
                table_qa.get("misalignment_ratio"),
                table_qa.get("merged_header_geometry_signal"),
                table_qa.get("label_column_index"),
                table_qa.get("fallback_triggered"),
            )
        if has_empty_first_col:
            logger.warning(
                "TABLE_INSPECT alignment issue chunk=%s: first column is empty in data rows (row labels likely lost)",
                chunk_id,
            )

    return TableInspectResponse(total_chunks=len(hits), findings=findings)

