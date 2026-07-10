"""Helpers for building sample XLSX files in tests."""

from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook
from openpyxl.worksheet.table import Table


def build_sample_xlsx(path: Path) -> Path:
    workbook = Workbook()
    revenue = workbook.active
    revenue.title = "Revenue"
    revenue.append(["Year", "Revenue"])
    revenue.append(["2023", 100])
    revenue.append(["2024", 120])
    revenue.add_table(Table(displayName="RevenueTable", ref="A1:B3"))

    hidden = workbook.create_sheet("Hidden")
    hidden.append(["Secret", "Value"])
    hidden.append(["X", 1])
    hidden.sheet_state = "hidden"

    bands = workbook.create_sheet("Bands")
    bands.append(["Metric", "Value"])
    for index in range(1, 6):
        bands.append([f"Row {index}", index * 10])

    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)
    workbook.close()
    return path
