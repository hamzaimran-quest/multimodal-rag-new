"""Table extraction via pdfplumber with quality-gated fallback."""

from __future__ import annotations

import logging
import statistics
from typing import TYPE_CHECKING

from app.charts.profile import analyze_table_chartability
from app.ingestion.models import ExtractedChunk
from app.ingestion.table_geometry import (
    reconstruct_table_geometry,
    reconstruct_table_text_lines,
)
from app.ingestion.tables import (
    MISALIGNMENT_THRESHOLD,
    clean_cell,
    column_misalignment_ratio,
    is_numeric_like,
    table_signature,
    table_to_markdown,
    validate_reconstructed_table,
)

if TYPE_CHECKING:
    import pdfplumber

logger = logging.getLogger(__name__)

MIN_TABLE_WORDS = 4
MERGED_HEADER_WIDTH_RATIO = 1.8


def _infer_label_column(rows: list[list[object | None]]) -> int | None:
    """Infer row-label column by type consistency, not fixed index."""
    cleaned = [[clean_cell(c) for c in row] for row in rows if any(clean_cell(c) for c in row)]
    if len(cleaned) < 2:
        return None
    width = max(len(r) for r in cleaned)
    scores: list[tuple[int, float]] = []

    data_rows = cleaned[1:] if len(cleaned) > 1 else cleaned
    for col in range(width):
        non_empty = 0
        text_like = 0
        numeric_like = 0
        for row in data_rows:
            val = row[col] if col < len(row) else ""
            if not val:
                continue
            non_empty += 1
            if is_numeric_like(val):
                numeric_like += 1
            else:
                text_like += 1
        if non_empty == 0:
            continue
        score = (text_like / non_empty) - (numeric_like / non_empty)
        scores.append((col, score))

    if not scores:
        return None
    best_col, best_score = max(scores, key=lambda x: x[1])
    return best_col if best_score > 0.3 else None


def _label_column_empty_ratio(rows: list[list[object | None]], label_col: int | None) -> float:
    if label_col is None:
        return 1.0

    cleaned = [[clean_cell(c) for c in row] for row in rows if any(clean_cell(c) for c in row)]
    if len(cleaned) < 2:
        return 0.0

    data_rows = cleaned[1:]
    if not data_rows:
        return 0.0

    empty = sum(1 for row in data_rows if not (row[label_col] if label_col < len(row) else ""))
    return empty / len(data_rows)


def _numeric_data_ratio(rows: list[list[object | None]]) -> float:
    cleaned = [[clean_cell(c) for c in row] for row in rows if any(clean_cell(c) for c in row)]
    if len(cleaned) < 2:
        return 0.0

    numeric = 0
    non_empty = 0
    for row in cleaned[1:]:
        for cell in row:
            if not cell:
                continue
            non_empty += 1
            if is_numeric_like(cell):
                numeric += 1

    if non_empty == 0:
        return 0.0
    return numeric / non_empty


def _semantic_label_loss_detected(
    rows: list[list[object | None]],
    label_col: int | None,
) -> tuple[bool, float, float]:
    label_empty_ratio = _label_column_empty_ratio(rows, label_col)
    numeric_ratio = _numeric_data_ratio(rows)
    cleaned = [[clean_cell(c) for c in row] for row in rows if any(clean_cell(c) for c in row)]
    data_row_count = max(0, len(cleaned) - 1)

    semantic_loss = data_row_count >= 4 and numeric_ratio >= 0.6 and label_empty_ratio >= 0.6
    return semantic_loss, label_empty_ratio, numeric_ratio


def _merged_header_geometry_signal(table: "pdfplumber.table.Table") -> bool:
    try:
        cell_widths = []
        for cell in table.cells or []:
            if not cell:
                continue
            x0, _, x1, _ = cell
            w = float(x1) - float(x0)
            if w > 0:
                cell_widths.append(w)
        if len(cell_widths) < 4:
            return False
        median_width = statistics.median(cell_widths)
        if median_width <= 0:
            return False
        return any(w >= median_width * MERGED_HEADER_WIDTH_RATIO for w in cell_widths)
    except Exception:
        return False


def _move_label_column_to_front(rows: list[list[object | None]], label_col: int | None) -> list[list[object | None]]:
    if label_col is None or label_col <= 0:
        return rows
    moved: list[list[object | None]] = []
    for row in rows:
        if label_col >= len(row):
            moved.append(row)
            continue
        new_row = [row[label_col], *row[:label_col], *row[label_col + 1 :]]
        moved.append(new_row)
    return moved


def _pdfplumber_tables(
    page: "pdfplumber.page.Page",
) -> list[tuple[list[list[object | None]], tuple[float, ...], dict]]:
    found: list[tuple[list[list[object | None]], tuple[float, ...], dict]] = []
    for table in page.find_tables() or []:
        rows = table.extract()
        if rows:
            misalignment_ratio = column_misalignment_ratio(rows)
            merged_header_signal = _merged_header_geometry_signal(table)
            label_col = _infer_label_column(rows)
            semantic_label_loss, label_empty_ratio, numeric_ratio = _semantic_label_loss_detected(rows, label_col)
            rows_reordered = _move_label_column_to_front(rows, label_col)
            fallback_triggered = (misalignment_ratio > MISALIGNMENT_THRESHOLD) or semantic_label_loss
            qa = {
                "misalignment_ratio": round(misalignment_ratio, 4),
                "merged_header_geometry_signal": merged_header_signal,
                "label_column_index": label_col,
                "label_empty_ratio": round(label_empty_ratio, 4),
                "numeric_data_ratio": round(numeric_ratio, 4),
                "semantic_label_loss": semantic_label_loss,
                "fallback_triggered": fallback_triggered,
            }
            found.append((rows_reordered, table.bbox, qa))
    return found


def _camelot_tables(pdf_path: str, page_number: int) -> list[list[list[object | None]]]:
    try:
        import camelot
    except ImportError:
        return []

    tables: list[list[list[object | None]]] = []
    for flavor in ("lattice", "stream"):
        try:
            result = camelot.read_pdf(pdf_path, pages=str(page_number), flavor=flavor)
        except Exception:
            logger.debug("Camelot %s failed for page %s", flavor, page_number, exc_info=True)
            continue
        for table in result:
            rows = table.df.values.tolist()
            if rows:
                tables.append(rows)
    return tables


def _rows_to_chunk(
    rows: list[list[object | None]],
    page_number: int,
    extraction_method: str,
    bbox: list[float] | None = None,
    qa: dict | None = None,
) -> ExtractedChunk | None:
    markdown = table_to_markdown(rows)
    if not markdown or len(markdown.split()) < MIN_TABLE_WORDS:
        return None

    _, headers = table_signature(rows)
    extra: dict = {}
    if headers:
        extra["table_headers"] = list(headers)
    if qa:
        extra["table_qa"] = qa

    chart_profile = analyze_table_chartability(rows)
    if chart_profile:
        extra["chart_profile"] = chart_profile

    return ExtractedChunk(
        content=markdown,
        page_number=page_number,
        chunk_type="table",
        extraction_method=extraction_method,
        bbox=bbox,
        extra_metadata=extra,
    )


def _qa_for_recovered_rows(rows: list[list[str]], recovery_method: str, base_qa: dict) -> dict:
    ok, validation = validate_reconstructed_table(rows)
    return {
        **base_qa,
        "recovery_method": recovery_method,
        "recovery_validated": ok,
        "recovery_validation": validation,
        "misalignment_ratio": validation.get("misalignment_ratio", base_qa.get("misalignment_ratio", 0.0)),
        "label_nonempty_rate": validation.get("label_nonempty_rate"),
        "semantic_label_loss": False,
        "fallback_triggered": True,
    }


def _attempt_table_recovery(
    page: "pdfplumber.page.Page",
    pdf_path: str,
    page_number: int,
    bbox: tuple[float, ...],
    base_qa: dict,
) -> tuple[list[list[str]], str, dict] | None:
    """Try geometry, then text-line fallback, then optional Camelot."""
    geometry_rows = reconstruct_table_geometry(page, bbox)
    if geometry_rows:
        ok, _ = validate_reconstructed_table(geometry_rows)
        if ok:
            logger.info("TABLE_QA page=%s recovered via geometry reconstruction", page_number)
            return geometry_rows, "geometry", _qa_for_recovered_rows(geometry_rows, "geometry", base_qa)
        logger.info("TABLE_QA page=%s geometry reconstruction failed validation", page_number)

    text_rows = reconstruct_table_text_lines(page, bbox)
    if text_rows:
        ok, _ = validate_reconstructed_table(text_rows)
        if ok:
            logger.info("TABLE_QA page=%s recovered via text-line fallback", page_number)
            return text_rows, "text_lines", _qa_for_recovered_rows(text_rows, "text_lines", base_qa)
        logger.info("TABLE_QA page=%s text-line fallback failed validation", page_number)

    for rows in _camelot_tables(pdf_path, page_number):
        label_col = _infer_label_column(rows)
        rows_reordered = _move_label_column_to_front(rows, label_col)
        normalized = [[clean_cell(c) for c in row] for row in rows_reordered]
        ok, _ = validate_reconstructed_table(normalized)
        if ok:
            logger.info("TABLE_QA page=%s recovered via optional camelot compare", page_number)
            return normalized, "camelot_optional", _qa_for_recovered_rows(normalized, "camelot_optional", base_qa)

    return None


def extract_tables_for_page(
    page: "pdfplumber.page.Page",
    page_number: int,
    pdf_path: str,
) -> tuple[list[ExtractedChunk], list[tuple[float, ...]]]:
    chunks: list[ExtractedChunk] = []
    bboxes: list[tuple[float, ...]] = []

    plumber_hits = _pdfplumber_tables(page)
    if not plumber_hits:
        for rows in _camelot_tables(pdf_path, page_number):
            label_col = _infer_label_column(rows)
            rows_reordered = _move_label_column_to_front(rows, label_col)
            normalized = [[clean_cell(c) for c in row] for row in rows_reordered]
            ok, validation = validate_reconstructed_table(normalized)
            if not ok:
                continue
            qa = {
                "misalignment_ratio": validation.get("misalignment_ratio", 0.0),
                "merged_header_geometry_signal": False,
                "label_column_index": label_col,
                "label_nonempty_rate": validation.get("label_nonempty_rate"),
                "semantic_label_loss": False,
                "fallback_triggered": False,
            }
            chunk = _rows_to_chunk(normalized, page_number, "camelot", None, qa)
            if chunk:
                chunks.append(chunk)
        return chunks, bboxes

    for rows, bbox, qa in plumber_hits:
        logger.info(
            "TABLE_QA page=%s misalignment=%.3f merged_header=%s label_col=%s label_empty=%.3f numeric_ratio=%.3f semantic_loss=%s",
            page_number,
            qa.get("misalignment_ratio", 0.0),
            qa.get("merged_header_geometry_signal", False),
            qa.get("label_column_index"),
            qa.get("label_empty_ratio", 0.0),
            qa.get("numeric_data_ratio", 0.0),
            qa.get("semantic_label_loss", False),
        )

        if qa.get("fallback_triggered"):
            logger.warning(
                "TABLE_QA page=%s fallback triggered: misalignment or semantic label-loss",
                page_number,
            )
            recovered = _attempt_table_recovery(page, pdf_path, page_number, bbox, qa)
            if recovered:
                recovered_rows, method, recovery_qa = recovered
                bboxes.append(bbox)
                chunk = _rows_to_chunk(recovered_rows, page_number, method, list(bbox), recovery_qa)
                if chunk:
                    chunks.append(chunk)
                continue

            logger.warning(
                "TABLE_QA page=%s skipping broken table chunk after recovery attempts failed",
                page_number,
            )
            continue

        bboxes.append(bbox)
        chunk = _rows_to_chunk(rows, page_number, "pdfplumber", list(bbox), qa)
        if chunk:
            chunks.append(chunk)

    return chunks, bboxes
