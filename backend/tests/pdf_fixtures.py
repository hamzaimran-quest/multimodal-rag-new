"""Helpers for generating test PDFs."""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF


def build_sample_pdf(path: Path) -> Path:
    """Create a small PDF with paragraph text suitable for ingestion tests."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    pdf.multi_cell(
        w=180,
        h=8,
        text=(
            "Five-Year Financial Highlights. Revenue grew steadily across regions. "
            "Operating profit improved year over year with disciplined cost control. "
            "Net income increased due to stronger margins and stable demand."
        ),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))
    return path
