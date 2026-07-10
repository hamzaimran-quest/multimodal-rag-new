"""Read XLSX workbook data for the spreadsheet viewer API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openpyxl import load_workbook

from app.ingestion.xlsx_extract import is_sheet_visible, rows_from_range


def find_xlsx_path(user_id: int, doc_id: str) -> Path | None:
    from app.ingestion.pipeline import document_upload_dir

    dest_dir = document_upload_dir(user_id, doc_id)
    if not dest_dir.exists():
        return None
    matches = sorted(dest_dir.glob("*.xlsx"))
    return matches[0] if matches else None


def list_workbook_sheets(xlsx_path: Path) -> list[dict[str, Any]]:
    workbook = load_workbook(xlsx_path, read_only=True, data_only=True)
    sheets: list[dict[str, Any]] = []
    visible_index = 0
    try:
        for worksheet in workbook.worksheets:
            if not is_sheet_visible(worksheet):
                continue
            visible_index += 1
            sheets.append(
                {
                    "name": worksheet.title,
                    "index": visible_index,
                    "row_count": int(worksheet.max_row or 0),
                    "col_count": int(worksheet.max_column or 0),
                }
            )
    finally:
        workbook.close()
    return sheets


def read_sheet_grid(
    xlsx_path: Path,
    sheet_name: str,
    *,
    row_start: int | None = None,
    row_end: int | None = None,
    col_start: int | None = None,
    col_end: int | None = None,
) -> dict[str, Any]:
    workbook = load_workbook(xlsx_path, read_only=False, data_only=True)
    try:
        if sheet_name not in workbook.sheetnames:
            raise KeyError(sheet_name)
        worksheet = workbook[sheet_name]
        if not is_sheet_visible(worksheet):
            raise KeyError(sheet_name)

        visible_index = 0
        for ws in workbook.worksheets:
            if not is_sheet_visible(ws):
                continue
            visible_index += 1
            if ws.title == sheet_name:
                break

        min_row = row_start or 1
        max_row = row_end or int(worksheet.max_row or 1)
        min_col = col_start or 1
        max_col = col_end or int(worksheet.max_column or 1)
        rows = rows_from_range(worksheet, min_row, min_col, max_row, max_col)

        highlight = None
        if any(value is not None for value in (row_start, row_end, col_start, col_end)):
            highlight = {
                "row_start": min_row,
                "row_end": max_row,
                "col_start": min_col,
                "col_end": max_col,
            }

        return {
            "name": sheet_name,
            "index": visible_index,
            "rows": rows,
            "row_count": int(worksheet.max_row or 0),
            "col_count": int(worksheet.max_column or 0),
            "highlight": highlight,
        }
    finally:
        workbook.close()


def resolve_sheet_by_index(xlsx_path: Path, sheet_index: int) -> str | None:
    sheets = list_workbook_sheets(xlsx_path)
    for sheet in sheets:
        if sheet["index"] == sheet_index:
            return sheet["name"]
    return None
