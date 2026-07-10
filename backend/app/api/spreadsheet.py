"""Authenticated spreadsheet data API for XLSX documents.

Separate from /documents/{doc_id}/file (PDF viewer). Serves sheet metadata and
grid data for the native spreadsheet viewer without rendering through PDF.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from opensearchpy import OpenSearch

from app.auth.dependencies import get_current_user
from app.db.models import User
from app.ingestion.xlsx_data import find_xlsx_path, list_workbook_sheets, read_sheet_grid
from app.opensearch.documents import get_document_for_user

router = APIRouter(prefix="/documents", tags=["documents"])


def _get_opensearch(request: Request) -> OpenSearch:
    return request.app.state.opensearch


def _resolve_owned_xlsx(request: Request, doc_id: str, user_id: int) -> Path:
    client = _get_opensearch(request)
    record = get_document_for_user(client, doc_id, user_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")

    xlsx_path = find_xlsx_path(user_id, doc_id)
    if xlsx_path is None or not xlsx_path.is_file():
        raise HTTPException(status_code=404, detail="Spreadsheet not found")
    return xlsx_path


@router.get("/{doc_id}/spreadsheet")
async def get_spreadsheet_metadata(
    doc_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> dict:
    xlsx_path = _resolve_owned_xlsx(request, doc_id, current_user.id)
    sheets = list_workbook_sheets(xlsx_path)
    return {
        "doc_id": doc_id,
        "filename": xlsx_path.name,
        "sheet_count": len(sheets),
        "sheets": sheets,
    }


@router.get("/{doc_id}/spreadsheet/sheets/{sheet_name}")
async def get_spreadsheet_sheet(
    doc_id: str,
    sheet_name: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    row_start: int | None = Query(default=None, ge=1),
    row_end: int | None = Query(default=None, ge=1),
    col_start: int | None = Query(default=None, ge=1),
    col_end: int | None = Query(default=None, ge=1),
) -> dict:
    xlsx_path = _resolve_owned_xlsx(request, doc_id, current_user.id)
    try:
        return read_sheet_grid(
            xlsx_path,
            sheet_name,
            row_start=row_start,
            row_end=row_end,
            col_start=col_start,
            col_end=col_end,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Sheet not found") from None
