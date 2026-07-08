"""Tests for geometry-based table reconstruction."""

from __future__ import annotations

from pathlib import Path

import pdfplumber
import pytest

from app.config import PROJECT_ROOT
from app.ingestion.table_geometry import reconstruct_table_geometry
from app.ingestion.tables import validate_reconstructed_table

HUAWEI_PDF = PROJECT_ROOT / "huawei.pdf"


@pytest.mark.skipif(not HUAWEI_PDF.exists(), reason="huawei.pdf not found")
def test_huawei_page9_geometry_reconstruction():
    with pdfplumber.open(HUAWEI_PDF) as doc:
        page = doc.pages[8]
        bbox = page.find_tables()[0].bbox
        rows = reconstruct_table_geometry(page, bbox)

    assert rows is not None
    ok, validation = validate_reconstructed_table(rows)
    assert ok, validation

    headers = rows[0]
    assert headers[0] == "Metric"
    assert any("2025" in h and "USD" in h for h in headers[1:])
    assert any("2025" in h and "CNY" in h for h in headers[1:])
    assert any("2024" in h for h in headers[1:])

    revenue_row = next(row for row in rows[1:] if row[0] == "Revenue")
    assert revenue_row[1] == "126,018"
    assert revenue_row[2] == "880,941"
    assert revenue_row[3] == "862,072"
    assert revenue_row[4] == "704,174"
