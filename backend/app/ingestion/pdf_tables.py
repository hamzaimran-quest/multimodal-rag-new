"""Table extraction via pdfplumber with quality-gated fallback."""

from __future__ import annotations

import logging
import statistics
from typing import TYPE_CHECKING

from app.ingestion.pdf_headings import nearest_heading_above_bbox
from app.ingestion.models import ExtractedChunk
from app.ingestion.pdf_coords import crop_offset, shift_bbox
from app.ingestion.pdf_text_repair import repair_pdf_text
from app.ingestion.table_geometry import (
    expand_table_bbox_for_labels,
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
LABEL_EMPTY_SHORTCIRCUIT = 0.6
_BBOX_DEDUPE_TOLERANCE = 2.0
_MIN_NARRATIVE_BAND_HEIGHT = 20.0
_NARRATIVE_ZONE_RIGHT_PAD = 8.0
_LEFT_TABLE_LAYOUT_RATIO = 0.55


def table_text_exclusion_zones(
    page: "pdfplumber.page.Page",
    table_bboxes: list[tuple[float, ...]],
) -> list[tuple[float, ...]]:
    """
    Regions to exclude from PDF text extraction around indexed tables.

    Includes the table bbox plus a right-adjacent row band where narrative prose
    often duplicates tabular facts (common in annual-report layouts).
    """
    zones: list[tuple[float, ...]] = [tuple(float(v) for v in bbox) for bbox in table_bboxes]
    if not table_bboxes:
        return zones

    px0, ptop, page_right, pbottom = page.bbox
    eligible = [
        bbox
        for bbox in table_bboxes
        if float(bbox[3]) - float(bbox[1]) >= _MIN_NARRATIVE_BAND_HEIGHT
        and float(bbox[2]) < page_right - _NARRATIVE_ZONE_RIGHT_PAD
    ]
    if not eligible:
        return zones

    band_top = min(float(bbox[1]) for bbox in eligible)
    band_bottom = max(float(bbox[3]) for bbox in eligible)
    band_right = max(float(bbox[2]) for bbox in eligible)
    if band_right < page.width * _LEFT_TABLE_LAYOUT_RATIO:
        band_bottom = pbottom
    zones.append((band_right, band_top, page_right, band_bottom))
    return zones


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
    # Justified prose can get shaped into a sparse 1-2 "column" grid by table detection
    # (wrapped sentence fragments split across rows). A real table always has a
    # meaningful share of numeric cells; near-zero numeric content means this is
    # narrative text, not tabular data, and should fall through to normal text
    # extraction instead of being indexed as a garbled table chunk.
    no_tabular_data = data_row_count >= 3 and numeric_ratio <= 0.05
    return semantic_loss or no_tabular_data, label_empty_ratio, numeric_ratio


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


def _bbox_is_duplicate(candidate: tuple[float, ...], seen: list[tuple[float, ...]]) -> bool:
    for existing in seen:
        if all(abs(float(a) - float(b)) <= _BBOX_DEDUPE_TOLERANCE for a, b in zip(candidate, existing)):
            return True
    return False


def _discover_table_candidates(
    page: "pdfplumber.page.Page",
) -> list[tuple["pdfplumber.table.Table", tuple[float, float, float, float]]]:
    """Use pdfplumber only to locate table regions on the page."""
    found: list[tuple["pdfplumber.table.Table", tuple[float, float, float, float]]] = []
    seen_bboxes: list[tuple[float, ...]] = []
    for table in page.find_tables() or []:
        bbox = tuple(float(v) for v in table.bbox)
        if _bbox_is_duplicate(bbox, seen_bboxes):
            continue
        seen_bboxes.append(bbox)
        found.append((table, bbox))
    return found


def _qa_from_extract_rows(
    table: "pdfplumber.table.Table",
    rows: list[list[object | None]],
) -> dict:
    misalignment_ratio = column_misalignment_ratio(rows)
    merged_header_signal = _merged_header_geometry_signal(table)
    label_col = _infer_label_column(rows)
    semantic_label_loss, label_empty_ratio, numeric_ratio = _semantic_label_loss_detected(rows, label_col)
    fallback_triggered = (misalignment_ratio > MISALIGNMENT_THRESHOLD) or semantic_label_loss
    return {
        "misalignment_ratio": round(misalignment_ratio, 4),
        "merged_header_geometry_signal": merged_header_signal,
        "label_column_index": label_col,
        "label_empty_ratio": round(label_empty_ratio, 4),
        "numeric_data_ratio": round(numeric_ratio, 4),
        "semantic_label_loss": semantic_label_loss,
        "fallback_triggered": fallback_triggered,
        "geometry_primary": False,
        "bbox_expanded": False,
    }


def _repair_row_text(rows: list[list[object | None]]) -> list[list[object | None]]:
    return [[repair_pdf_text(cell) if isinstance(cell, str) else cell for cell in row] for row in rows]


def _probe_pdfplumber_extract(
    table: "pdfplumber.table.Table",
) -> tuple[list[list[object | None]], dict] | None:
    rows = table.extract()
    if not rows:
        return None
    rows = _repair_row_text(rows)
    label_col = _infer_label_column(rows)
    rows_reordered = _move_label_column_to_front(rows, label_col)
    return rows_reordered, _qa_from_extract_rows(table, rows)


def _should_short_circuit_extract(qa: dict) -> bool:
    """Skip pdfplumber cell extract when geometry is the right tool (empty labels + numeric grid)."""
    if qa.get("misalignment_ratio", 0.0) > MISALIGNMENT_THRESHOLD:
        return True
    if qa.get("semantic_label_loss"):
        return True
    return (
        qa.get("label_empty_ratio", 1.0) >= LABEL_EMPTY_SHORTCIRCUIT
        and qa.get("numeric_data_ratio", 0.0) >= 0.6
    )


def _qa_for_geometry_rows(rows: list[list[str]], method: str, base_qa: dict, *, bbox_expanded: bool) -> dict:
    _, validation = validate_reconstructed_table(rows)
    return {
        **base_qa,
        "recovery_method": method,
        "recovery_validated": True,
        "recovery_validation": validation,
        "misalignment_ratio": validation.get("misalignment_ratio", base_qa.get("misalignment_ratio", 0.0)),
        "label_nonempty_rate": validation.get("label_nonempty_rate"),
        "semantic_label_loss": False,
        "fallback_triggered": False,
        "geometry_primary": True,
        "bbox_expanded": bbox_expanded,
    }


def _extract_via_geometry(
    page: "pdfplumber.page.Page",
    bbox: tuple[float, ...],
    page_number: int,
    base_qa: dict,
) -> tuple[list[list[str]], str, dict] | None:
    """Geometry-first reconstruction on a label-expanded bbox."""
    expanded = expand_table_bbox_for_labels(page, bbox)
    bbox_expanded = expanded != bbox
    for method, reconstruct in (
        ("geometry", reconstruct_table_geometry),
        ("text_lines", reconstruct_table_text_lines),
    ):
        rows = reconstruct(page, expanded)
        if not rows:
            continue
        ok, validation = validate_reconstructed_table(rows)
        if not ok:
            logger.debug(
                "TABLE_QA page=%s %s failed validation on expanded bbox: %s",
                page_number,
                method,
                validation.get("reason"),
            )
            continue
        logger.info(
            "TABLE_QA page=%s indexed via %s (geometry-first, expanded=%s, label_nonempty=%.3f)",
            page_number,
            method,
            bbox_expanded,
            validation.get("label_nonempty_rate", 0.0),
        )
        return rows, method, _qa_for_geometry_rows(rows, method, base_qa, bbox_expanded=bbox_expanded)
    return None


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


def _attempt_camelot_recovery(
    pdf_path: str,
    page_number: int,
    base_qa: dict,
) -> tuple[list[list[str]], dict] | None:
    for rows in _camelot_tables(pdf_path, page_number):
        label_col = _infer_label_column(rows)
        rows_reordered = _move_label_column_to_front(rows, label_col)
        normalized = [[clean_cell(c) for c in row] for row in rows_reordered]
        ok, validation = validate_reconstructed_table(normalized)
        if not ok:
            continue
        logger.info("TABLE_QA page=%s recovered via optional camelot compare", page_number)
        qa = {
            **base_qa,
            "recovery_method": "camelot_optional",
            "recovery_validated": True,
            "recovery_validation": validation,
            "label_nonempty_rate": validation.get("label_nonempty_rate"),
            "semantic_label_loss": False,
            "fallback_triggered": True,
            "geometry_primary": False,
        }
        return normalized, qa
    return None


def _rows_to_chunk(
    rows: list[list[object | None]],
    page_number: int,
    extraction_method: str,
    bbox: list[float] | None = None,
    qa: dict | None = None,
    *,
    heading: str | None = None,
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
    if heading:
        extra["table_heading"] = heading
        markdown = f"{heading}\n\n{markdown}"

    return ExtractedChunk(
        content=markdown,
        page_number=page_number,
        chunk_type="table",
        extraction_method=extraction_method,
        bbox=bbox,
        extra_metadata=extra,
    )


def _extract_table_candidate(
    page: "pdfplumber.page.Page",
    pdf_path: str,
    page_number: int,
    table: "pdfplumber.table.Table",
    bbox: tuple[float, ...],
) -> ExtractedChunk | None:
    # `bbox` stays in pdfplumber's native (MediaBox-anchored) coordinate space for
    # every internal page query below (page.within_bbox, geometry reconstruction,
    # heading lookup) -- only `output_bbox` (CropBox-relative) is ever stored on a
    # chunk, since that's the frame any PDF renderer (incl. the frontend viewer)
    # actually displays. See app.ingestion.pdf_coords for why this distinction matters.
    output_bbox = shift_bbox(bbox, crop_offset(page))
    probe = _probe_pdfplumber_extract(table)
    base_qa = probe[1] if probe else {
        "misalignment_ratio": 0.0,
        "merged_header_geometry_signal": False,
        "label_column_index": None,
        "label_empty_ratio": 1.0,
        "numeric_data_ratio": 0.0,
        "semantic_label_loss": True,
        "fallback_triggered": True,
        "geometry_primary": False,
        "bbox_expanded": False,
    }

    if probe:
        logger.info(
            "TABLE_QA page=%s misalignment=%.3f merged_header=%s label_col=%s label_empty=%.3f numeric_ratio=%.3f semantic_loss=%s",
            page_number,
            base_qa.get("misalignment_ratio", 0.0),
            base_qa.get("merged_header_geometry_signal", False),
            base_qa.get("label_column_index"),
            base_qa.get("label_empty_ratio", 0.0),
            base_qa.get("numeric_data_ratio", 0.0),
            base_qa.get("semantic_label_loss", False),
        )

    geometry_hit = _extract_via_geometry(page, bbox, page_number, base_qa)
    if geometry_hit:
        rows, method, qa = geometry_hit
        heading = nearest_heading_above_bbox(page, bbox)
        return _rows_to_chunk(rows, page_number, method, output_bbox, qa, heading=heading)

    if probe and not _should_short_circuit_extract(base_qa) and not base_qa.get("fallback_triggered"):
        rows, qa = probe
        normalized = [[clean_cell(c) for c in row] for row in rows]
        # Header-only "tables" (no data rows) are typically harmless decorative
        # artifacts (e.g. a section rule pdfplumber detects as a 1-row grid) -- the
        # stricter row/label validation below is about catching garbled *data*
        # reconstructions, not about requiring every detected grid to have data rows.
        has_data_rows = len([row for row in normalized if any(clean_cell(c) for c in row)]) >= 2
        ok, validation = (True, {}) if not has_data_rows else validate_reconstructed_table(normalized)
        if ok:
            qa = {**qa, "recovery_validation": validation}
            logger.info("TABLE_QA page=%s indexed via pdfplumber extract", page_number)
            heading = nearest_heading_above_bbox(page, bbox)
            return _rows_to_chunk(normalized, page_number, "pdfplumber", output_bbox, qa, heading=heading)
        logger.debug(
            "TABLE_QA page=%s pdfplumber extract failed validation: %s",
            page_number,
            validation.get("reason"),
        )

    if probe and _should_short_circuit_extract(base_qa):
        logger.info(
            "TABLE_QA page=%s skipping pdfplumber extract (label_empty=%.3f, semantic_loss=%s)",
            page_number,
            base_qa.get("label_empty_ratio", 1.0),
            base_qa.get("semantic_label_loss", False),
        )

    camelot_hit = _attempt_camelot_recovery(pdf_path, page_number, base_qa)
    if camelot_hit:
        rows, qa = camelot_hit
        heading = nearest_heading_above_bbox(page, bbox)
        return _rows_to_chunk(rows, page_number, "camelot_optional", output_bbox, qa, heading=heading)

    logger.warning("TABLE_QA page=%s skipping table at bbox after all extraction attempts failed", page_number)
    return None


def extract_tables_for_page(
    page: "pdfplumber.page.Page",
    page_number: int,
    pdf_path: str,
) -> tuple[list[ExtractedChunk], list[tuple[float, ...]]]:
    chunks: list[ExtractedChunk] = []
    bboxes: list[tuple[float, ...]] = []

    candidates = _discover_table_candidates(page)
    if not candidates:
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
                "geometry_primary": False,
            }
            chunk = _rows_to_chunk(normalized, page_number, "camelot", None, qa)
            if chunk:
                chunks.append(chunk)
        return chunks, bboxes

    for table, bbox in candidates:
        chunk = _extract_table_candidate(page, pdf_path, page_number, table, bbox)
        if chunk:
            chunks.append(chunk)
            bboxes.append(bbox)

    return chunks, bboxes
