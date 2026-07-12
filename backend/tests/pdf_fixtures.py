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


def build_two_column_pdf(path: Path) -> Path:
    """Create a two-column page where row-wise extraction interleaves columns."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    left_text = (
        "LEFTCOLUMN_START alpha beta gamma delta epsilon zeta eta theta iota kappa "
        "lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega END_LEFT"
    )
    right_text = (
        "RIGHTCOLUMN_START one two three four five six seven eight nine ten eleven "
        "twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen END_RIGHT"
    )

    pdf.set_xy(10, 20)
    pdf.multi_cell(85, 5, left_text)
    pdf.set_xy(110, 20)
    pdf.multi_cell(85, 5, right_text)

    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))
    return path


def build_two_column_headings_pdf(path: Path) -> Path:
    """Two-column page with distinct headings in each column band."""
    pdf = FPDF()
    pdf.add_page()

    left_body = (
        "Left column body begins here with enough words to pass the minimum chunk threshold easily. "
        "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho."
    )
    right_body = (
        "Right column body begins here with enough words to pass the minimum chunk threshold easily. "
        "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen."
    )

    pdf.set_xy(10, 20)
    pdf.set_font("Helvetica", size=18)
    pdf.multi_cell(85, 7, "LEFT TITLE HEADING")
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(85, 5, left_body)

    pdf.set_xy(110, 20)
    pdf.set_font("Helvetica", size=18)
    pdf.multi_cell(85, 7, "RIGHT TITLE HEADING")
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(85, 5, right_body)

    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))
    return path
