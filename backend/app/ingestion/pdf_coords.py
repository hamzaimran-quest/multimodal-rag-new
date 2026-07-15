"""Shifts pdfplumber's raw coordinates to be relative to the page's own CropBox.

pdfplumber reports word/table positions in the PDF's absolute native coordinate
space (flipped top-down using the page's MediaBox height) -- it is not aware of
CropBox and does not normalize to a (0, 0) origin. Most PDFs have CropBox equal
to MediaBox with a (0, 0) origin, so this is invisible. PDFs where CropBox is
smaller than MediaBox (e.g. print bleed/trim marks outside the visible page,
common in professionally typeset filings) need this shift: any PDF renderer,
including pdf.js in the frontend viewer, renders and reports geometry relative
to CropBox, not MediaBox -- without this shift, stored highlight boxes land off
by exactly the crop margin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pdfplumber

Box = tuple[float, float, float, float]


def crop_offset(page: "pdfplumber.page.Page") -> tuple[float, float]:
    """(x, top) of the page's own CropBox, in pdfplumber's coordinate space."""
    cropbox = page.cropbox
    return float(cropbox[0]), float(cropbox[1])


def shift_bbox(bbox: Box, offset: tuple[float, float]) -> list[float]:
    dx, dy = offset
    x0, top, x1, bottom = bbox
    return [float(x0) - dx, float(top) - dy, float(x1) - dx, float(bottom) - dy]
