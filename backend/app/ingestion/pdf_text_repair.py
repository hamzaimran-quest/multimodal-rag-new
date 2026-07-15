"""Mojibake repair for text pulled from PDFs.

Some PDFs use fonts without a proper ToUnicode CMap, which causes pdfminer/pdfplumber
to mis-decode glyphs for curly quotes, em-dashes, etc. (e.g. "Bank's" extracted as
"Bankâ€™s"). This is noise specific to PDF text extraction, not DOCX/XLSX ingestion
(those formats already carry proper Unicode strings), so this helper is only wired
into the PDF extraction paths (app.ingestion.text, pdf_tables, table_geometry).
"""

from __future__ import annotations

import ftfy


def repair_pdf_text(value: str) -> str:
    if not value:
        return value
    return ftfy.fix_text(value)
