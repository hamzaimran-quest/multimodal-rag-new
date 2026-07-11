"""Helpers for building sample XLSX files in tests."""

from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook
from openpyxl.worksheet.table import Table


def build_star_schema_xlsx(path: Path) -> Path:
    """Workbook with one primary sheet and 1:1 + 1:many satellites for schema tests."""
    workbook = Workbook()
    shows = workbook.active
    shows.title = "Shows"
    shows.append(["show_id", "title", "year"])
    shows.append(["1", "Alpha", "2020"])
    shows.append(["2", "Beta", "2021"])

    countries = workbook.create_sheet("Countries")
    countries.append(["show_id", "country"])
    countries.append(["1", "France"])
    countries.append(["2", "Germany"])

    cast = workbook.create_sheet("Cast")
    cast.append(["show_id", "actor"])
    cast.append(["1", "Alice"])
    cast.append(["1", "Bob"])
    cast.append(["2", "Carol"])

    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)
    workbook.close()
    return path


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
