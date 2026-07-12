"""Integration tests for geometry-first PDF table extraction."""

from __future__ import annotations

import pdfplumber
import pytest

from app.config import PROJECT_ROOT
from app.ingestion.pdf_tables import extract_tables_for_page, table_text_exclusion_zones
from app.ingestion.table_geometry import expand_table_bbox_for_labels
from app.ingestion.text import extract_page_chunks

HUAWEI_PDF = PROJECT_ROOT / "huawei.pdf"
TIMBERLAND_PDF = PROJECT_ROOT / "timberland.pdf"


def _table_rows(content: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in content.splitlines():
        if not line.startswith("|") or "---" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows.append(cells)
    return rows


def _first_data_label(content: str) -> str:
    rows = _table_rows(content)
    return rows[1][0] if len(rows) > 1 else ""


@pytest.mark.skipif(not HUAWEI_PDF.exists(), reason="huawei.pdf not found at project root")
def test_huawei_page9_geometry_first_financial_highlights():
    with pdfplumber.open(HUAWEI_PDF) as pdf:
        page = pdf.pages[8]
        chunks, _ = extract_tables_for_page(page, 9, str(HUAWEI_PDF))

    assert len(chunks) == 1
    chunk = chunks[0]
    qa = chunk.extra_metadata.get("table_qa", {})
    assert chunk.extraction_method == "geometry"
    assert qa.get("geometry_primary") is True
    assert qa.get("label_nonempty_rate", 0) >= 0.9

    rows = _table_rows(chunk.content)
    revenue = next(row for row in rows if row[0] == "Revenue")
    assert revenue[1] == "126,018"
    assert revenue[2] == "880,941"


@pytest.mark.skipif(not HUAWEI_PDF.exists(), reason="huawei.pdf not found at project root")
def test_huawei_page23_region_and_segment_tables_have_row_labels():
    with pdfplumber.open(HUAWEI_PDF) as pdf:
        page = pdf.pages[22]
        chunks, _ = extract_tables_for_page(page, 23, str(HUAWEI_PDF))

    assert len(chunks) == 2
    labels = {_first_data_label(chunk.content) for chunk in chunks}
    assert "Consumer" in labels or "ICT Infrastructure" in labels
    assert "EMEA" in labels or "China" in labels

    for chunk in chunks:
        qa = chunk.extra_metadata.get("table_qa", {})
        assert chunk.extraction_method == "geometry"
        assert qa.get("geometry_primary") is True
        assert qa.get("label_nonempty_rate", 0) >= 0.9


@pytest.mark.skipif(not HUAWEI_PDF.exists(), reason="huawei.pdf not found at project root")
def test_huawei_page14_non_financial_table_falls_back_to_pdfplumber():
    with pdfplumber.open(HUAWEI_PDF) as pdf:
        page = pdf.pages[13]
        chunks, _ = extract_tables_for_page(page, 14, str(HUAWEI_PDF))

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.extraction_method == "pdfplumber"
    assert "Connectivity" in chunk.content


@pytest.mark.skipif(not HUAWEI_PDF.exists(), reason="huawei.pdf not found at project root")
def test_huawei_page23_excludes_narrative_duplicate_text():
    with pdfplumber.open(HUAWEI_PDF) as pdf:
        page = pdf.pages[22]
        chunks = extract_page_chunks(page, 23, str(HUAWEI_PDF))

    text_chunks = [chunk for chunk in chunks if chunk.chunk_type == "text"]
    joined = "\n".join(chunk.content for chunk in text_chunks)
    assert "CNY50,113 million in 2025" not in joined
    assert "CNY37,184 million" not in joined

    table_chunks = [chunk for chunk in chunks if chunk.chunk_type == "table"]
    assert len(table_chunks) == 2
    region_table = next(
        chunk for chunk in table_chunks if "China" in chunk.content and "EMEA" in chunk.content
    )
    assert region_table.extra_metadata.get("table_heading") or "2025" in region_table.content


@pytest.mark.skipif(not TIMBERLAND_PDF.exists(), reason="timberland.pdf not found at project root")
@pytest.mark.parametrize("page_number", [17, 18, 29])
def test_timberland_financial_tables_prefer_geometry_with_labels(page_number: int):
    with pdfplumber.open(TIMBERLAND_PDF) as pdf:
        page = pdf.pages[page_number - 1]
        chunks, _ = extract_tables_for_page(page, page_number, str(TIMBERLAND_PDF))

    assert chunks, f"expected at least one table chunk on page {page_number}"
    labeled = [
        chunk
        for chunk in chunks
        if chunk.extra_metadata.get("table_qa", {}).get("label_nonempty_rate", 0) >= 0.5
        or _first_data_label(chunk.content)
    ]
    assert labeled, f"expected labeled rows on page {page_number}"


def test_expand_table_bbox_for_labels_extends_left():
    class _Page:
        width = 600.0
        bbox = (0.0, 0.0, 600.0, 800.0)

    expanded = expand_table_bbox_for_labels(_Page(), (120.0, 10.0, 400.0, 200.0))
    assert expanded[0] < 120.0
    assert expanded[1:] == (10.0, 400.0, 200.0)


def test_table_text_exclusion_zones_adds_right_adjacent_band():
    class _Page:
        width = 600.0
        bbox = (0.0, 0.0, 600.0, 800.0)

    zones = table_text_exclusion_zones(_Page(), [(50.0, 100.0, 250.0, 300.0)])
    assert len(zones) == 2
    assert zones[1] == (250.0, 100.0, 600.0, 800.0)
